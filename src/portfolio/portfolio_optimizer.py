"""
Portfolio Optimizer

Implements Markowitz mean-variance optimization with transaction costs and constraints.
Provides multiple optimization objectives and robust covariance estimation.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""
    transaction_cost: float = 0.0004  # 0.04% default transaction cost
    use_shrinkage: bool = True
    max_iterations: int = 1000
    tolerance: float = 1e-9
    risk_free_rate: float = 0.0

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.transaction_cost < 0:
            raise ValueError("Transaction cost must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")


@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    transaction_cost: float = 0.0
    error_message: Optional[str] = None
    optimization_iterations: int = 0
    objective_value: float = 0.0

    @property
    def net_expected_return(self) -> float:
        """Expected return after transaction costs"""
        return self.expected_return - self.transaction_cost

    @property
    def total_leverage(self) -> float:
        """Total leverage (sum of absolute weights)"""
        return np.sum(np.abs(self.weights))


class PortfolioOptimizer:
    """
    Markowitz Portfolio Optimizer with Transaction Costs

    Implements mean-variance optimization with:
    - Multiple optimization objectives (max Sharpe, min volatility)
    - Transaction cost modeling
    - Ledoit-Wolf shrinkage covariance estimation
    - Various portfolio constraints
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize portfolio optimizer

        Args:
            config: Optimization configuration. Uses defaults if None.
        """
        self.config = config or OptimizationConfig()

        # Extract config for easier access
        self.transaction_cost = self.config.transaction_cost
        self.use_shrinkage = self.config.use_shrinkage
        self.max_iterations = self.config.max_iterations
        self.tolerance = self.config.tolerance
        self.risk_free_rate = self.config.risk_free_rate

    def optimize_weights(
        self,
        returns_data: pd.DataFrame,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, Any]] = None,
        objective: str = 'max_sharpe'
    ) -> OptimizationResult:
        """
        Optimize portfolio weights

        Args:
            returns_data: DataFrame of asset returns
            current_weights: Current portfolio weights (for transaction costs)
            constraints: Portfolio constraints dictionary
            objective: Optimization objective ('max_sharpe', 'min_volatility')

        Returns:
            OptimizationResult: Optimization results and metrics
        """
        # Input validation (raise exceptions for invalid inputs)
        self._validate_inputs(returns_data, current_weights)

        # Validate objective before try block
        valid_objectives = ['max_sharpe', 'min_volatility']
        if objective not in valid_objectives:
            raise ValueError(f"Unsupported optimization objective: {objective}. Valid options: {valid_objectives}")

        try:
            n_assets = len(returns_data.columns)

            # Handle single asset case
            if n_assets == 1:
                return self._handle_single_asset_case(returns_data, current_weights)

            # Set default current weights if not provided
            if current_weights is None:
                current_weights = np.zeros(n_assets)

            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean().values
            cov_matrix = self._calculate_covariance_matrix(returns_data)

            # Setup optimization problem
            objective_func = self._create_objective_function(
                expected_returns, cov_matrix, current_weights, objective
            )

            constraints_list = self._create_constraints(constraints, n_assets)
            bounds = self._create_bounds(constraints, n_assets)

            # Initial guess
            x0 = self._create_initial_guess(current_weights, n_assets)

            # Run optimization
            result = minimize(
                objective_func,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'disp': False
                }
            )

            # Process results
            if result.success:
                return self._create_success_result(
                    result.x, expected_returns, cov_matrix, current_weights,
                    result.nit, result.fun
                )
            else:
                return self._create_failure_result(result.message)

        except Exception as e:
            return self._create_failure_result(str(e))

    def _handle_single_asset_case(
        self,
        returns_data: pd.DataFrame,
        current_weights: Optional[np.ndarray]
    ) -> OptimizationResult:
        """Handle single asset portfolio case"""
        weights = np.array([1.0])
        returns = returns_data.iloc[:, 0]

        expected_return = returns.mean()
        volatility = returns.std()

        if volatility > 0:
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0

        # Transaction cost
        if current_weights is not None and len(current_weights) == 1:
            turnover = abs(weights[0] - current_weights[0])
        else:
            turnover = 1.0  # Full position
        transaction_cost = turnover * self.transaction_cost

        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            success=True,
            transaction_cost=transaction_cost,
            optimization_iterations=0,
            objective_value=0.0
        )

    def _validate_inputs(
        self,
        returns_data: pd.DataFrame,
        current_weights: Optional[np.ndarray]
    ) -> None:
        """Validate input parameters"""
        if not isinstance(returns_data, pd.DataFrame):
            raise TypeError("Returns data must be a pandas DataFrame")

        if returns_data.empty:
            raise ValueError("Returns data cannot be empty")

        if len(returns_data.columns) == 0:
            raise ValueError("Returns data must have at least one asset")

        if current_weights is not None:
            if len(current_weights) != len(returns_data.columns):
                raise ValueError(
                    f"Current weights dimension ({len(current_weights)}) "
                    f"must match number of assets ({len(returns_data.columns)})"
                )

    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate covariance matrix using sample or shrinkage estimation

        Args:
            returns_data: Asset returns data

        Returns:
            np.ndarray: Covariance matrix
        """
        if not self.use_shrinkage:
            return returns_data.cov().values

        return self._ledoit_wolf_shrinkage(returns_data)

    def _ledoit_wolf_shrinkage(self, returns_data: pd.DataFrame) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage covariance estimation

        Args:
            returns_data: Asset returns data

        Returns:
            np.ndarray: Shrinkage covariance matrix
        """
        returns = returns_data.values
        n, p = returns.shape

        # Handle edge cases
        if p == 1:
            # Single asset case
            variance = np.var(returns, ddof=1)
            if variance <= 0 or np.isnan(variance):
                variance = 1e-6
            return np.array([[variance]])

        # Demean returns with NaN handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            returns_mean = np.nanmean(returns, axis=0)
            returns_demeaned = returns - returns_mean

        # Replace NaN values with 0
        returns_demeaned = np.nan_to_num(returns_demeaned, nan=0.0, posinf=0.0, neginf=0.0)

        # Sample covariance
        sample_cov = np.cov(returns_demeaned.T)

        # Handle degenerate covariance matrix
        if np.any(np.diag(sample_cov) <= 0) or np.any(np.isnan(sample_cov)) or np.any(np.isinf(sample_cov)):
            # Use identity matrix scaled by mean variance
            mean_var = np.nanmean(np.diag(sample_cov))
            if mean_var <= 0 or np.isnan(mean_var):
                mean_var = 1e-4
            return np.eye(p) * mean_var

        # Shrinkage target (scaled identity matrix)
        mu = np.trace(sample_cov) / p
        if mu <= 0 or np.isnan(mu):
            mu = 1e-4
        target = np.eye(p) * mu

        # Calculate shrinkage intensity
        if n <= p + 1:
            # High-dimensional case: use maximum shrinkage
            shrinkage_intensity = 1.0
        else:
            # Standard Ledoit-Wolf formula
            y = returns_demeaned
            y2 = y**2

            # Asymptotic variance of sample covariance
            try:
                phi_matrix = np.zeros((p, p))
                for i in range(p):
                    for j in range(p):
                        phi_matrix[i, j] = np.mean(y2[:, i] * y2[:, j]) - sample_cov[i, j]**2

                phi = np.sum(phi_matrix)

                # Misspecification
                gamma = np.linalg.norm(sample_cov - target, 'fro')**2

                # Shrinkage intensity
                if gamma > 1e-10 and not np.isnan(phi):
                    kappa = phi / gamma
                    shrinkage_intensity = max(0, min(1, kappa / n))
                else:
                    shrinkage_intensity = 0.5  # Default moderate shrinkage
            except:
                shrinkage_intensity = 0.5  # Fallback

        # Shrunk covariance matrix
        shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target

        # Ensure positive definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(shrunk_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        shrunk_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        return shrunk_cov

    def _create_objective_function(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray,
        objective: str
    ):
        """Create optimization objective function"""
        if objective == 'max_sharpe':
            return self._max_sharpe_objective(expected_returns, cov_matrix, current_weights)
        elif objective == 'min_volatility':
            return self._min_volatility_objective(cov_matrix)
        else:
            # This should not happen due to validation in optimize_weights
            raise ValueError(f"Invalid objective: {objective}")

    def _max_sharpe_objective(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray
    ):
        """Objective function for maximizing Sharpe ratio"""
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)

            # Transaction cost
            turnover = np.sum(np.abs(weights - current_weights))
            transaction_cost = turnover * self.transaction_cost

            # Net return after costs
            net_return = portfolio_return - transaction_cost

            # Sharpe ratio (negative for minimization)
            if portfolio_std > 1e-10:
                return -(net_return - self.risk_free_rate) / portfolio_std
            else:
                return 1e6  # Large penalty for zero volatility

        return objective

    def _min_volatility_objective(self, cov_matrix: np.ndarray):
        """Objective function for minimizing volatility"""
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return portfolio_variance

        return objective

    def _create_constraints(self, constraints: Optional[Dict[str, Any]], n_assets: int) -> List[Dict]:
        """Create optimization constraints"""
        constraints_list = []

        # Portfolio weights sum to 1
        constraints_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })

        if constraints:
            # Maximum leverage constraint
            if 'max_leverage' in constraints:
                max_leverage = constraints['max_leverage']
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: max_leverage - np.sum(np.abs(w))
                })

            # Maximum position size constraint
            if 'max_position' in constraints:
                max_position = constraints['max_position']
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: max_position - np.max(np.abs(w))
                })

        return constraints_list

    def _create_bounds(self, constraints: Optional[Dict[str, Any]], n_assets: int) -> List[tuple]:
        """Create variable bounds"""
        if constraints and constraints.get('long_only', False):
            # Long-only constraint
            bounds = [(0, 1) for _ in range(n_assets)]
        else:
            # Allow long/short positions (default: -50% to +50% per position for flexibility)
            max_position = 0.5
            if constraints and 'max_position' in constraints:
                max_position = constraints['max_position']
            bounds = [(-max_position, max_position) for _ in range(n_assets)]

        return bounds

    def _create_initial_guess(self, current_weights: np.ndarray, n_assets: int) -> np.ndarray:
        """Create initial guess for optimization"""
        if np.sum(np.abs(current_weights)) > 1e-6:
            # Use current weights as starting point
            return current_weights
        else:
            # Equal weights as starting point
            return np.ones(n_assets) / n_assets

    def _create_success_result(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray,
        iterations: int,
        objective_value: float
    ) -> OptimizationResult:
        """Create successful optimization result"""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Transaction cost
        turnover = np.sum(np.abs(weights - current_weights))
        transaction_cost = turnover * self.transaction_cost

        # Sharpe ratio
        if portfolio_std > 1e-10:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        else:
            sharpe_ratio = 0.0

        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_std,
            sharpe_ratio=sharpe_ratio,
            success=True,
            transaction_cost=transaction_cost,
            optimization_iterations=iterations,
            objective_value=objective_value
        )

    def _create_failure_result(self, error_message: str) -> OptimizationResult:
        """Create failed optimization result"""
        return OptimizationResult(
            weights=np.array([]),
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            error_message=error_message
        )