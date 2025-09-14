"""
Correlation Analyzer

Implements cross-strategy correlation analysis and risk decomposition.
Provides dynamic correlation matrices and diversification metrics.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import warnings
from datetime import datetime


@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis"""
    window_size: int = 60  # Rolling window size
    min_periods: int = 30  # Minimum periods for calculation
    method: str = 'pearson'  # Correlation method
    decay_factor: float = 0.94  # For exponential weighting

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.window_size <= 0:
            raise ValueError("Window size must be positive")
        if self.min_periods <= 0:
            raise ValueError("Min periods must be positive")

        valid_methods = ['pearson', 'spearman', 'kendall']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid correlation method: {self.method}. Valid options: {valid_methods}")

        if not 0 < self.decay_factor <= 1:
            raise ValueError("Decay factor must be between 0 and 1")


@dataclass
class CorrelationMatrix:
    """Correlation matrix with metadata"""
    matrix: pd.DataFrame
    method: str = 'pearson'
    timestamp: Optional[pd.Timestamp] = None
    window_size: Optional[int] = None
    decay_factor: Optional[float] = None

    @property
    def average_correlation(self) -> float:
        """Average correlation excluding diagonal"""
        n = len(self.matrix)
        if n <= 1:
            return 0.0

        # Get upper triangle excluding diagonal
        upper_triangle = np.triu(self.matrix.values, k=1)
        valid_correlations = upper_triangle[upper_triangle != 0]

        if len(valid_correlations) == 0:
            return 0.0

        return np.nanmean(valid_correlations)

    @property
    def max_correlation(self) -> float:
        """Maximum correlation excluding diagonal"""
        n = len(self.matrix)
        if n <= 1:
            return 0.0

        upper_triangle = np.triu(self.matrix.values, k=1)
        valid_correlations = upper_triangle[upper_triangle != 0]

        if len(valid_correlations) == 0:
            return 0.0

        return np.nanmax(valid_correlations)

    @property
    def min_correlation(self) -> float:
        """Minimum correlation excluding diagonal"""
        n = len(self.matrix)
        if n <= 1:
            return 0.0

        upper_triangle = np.triu(self.matrix.values, k=1)
        valid_correlations = upper_triangle[upper_triangle != 0]

        if len(valid_correlations) == 0:
            return 0.0

        return np.nanmin(valid_correlations)


@dataclass
class RiskDecomposition:
    """Portfolio risk decomposition results"""
    total_portfolio_risk: float
    individual_risks: Dict[str, float]
    marginal_contributions: Dict[str, float]
    component_contributions: Dict[str, float]
    correlation_matrix: pd.DataFrame


@dataclass
class DiversificationMetrics:
    """Portfolio diversification metrics"""
    diversification_ratio: float
    effective_number_of_strategies: float
    concentration_ratio: float
    herfindahl_index: float


class CorrelationAnalyzer:
    """
    Cross-Strategy Correlation Analysis System

    Provides comprehensive correlation analysis and risk decomposition for
    multi-strategy portfolios, including:
    - Dynamic correlation matrices
    - Rolling correlations
    - Risk decomposition
    - Diversification metrics
    """

    def __init__(self, config: Optional[CorrelationConfig] = None):
        """
        Initialize correlation analyzer

        Args:
            config: Correlation analysis configuration. Uses defaults if None.
        """
        self.config = config or CorrelationConfig()

        # Extract config for easier access
        self.window_size = self.config.window_size
        self.min_periods = self.config.min_periods
        self.method = self.config.method
        self.decay_factor = self.config.decay_factor

        # Strategy return data storage
        self.strategy_returns: Dict[str, pd.Series] = {}

    def add_strategy_returns(
        self,
        strategy_name: str,
        returns: pd.Series,
        replace: bool = True
    ) -> None:
        """
        Add or update strategy return data

        Args:
            strategy_name: Name of the strategy
            returns: Series of strategy returns
            replace: If True, replace existing data. If False, append.
        """
        # Validate input
        if not isinstance(returns, pd.Series):
            raise TypeError("Returns must be a pandas Series")

        if len(returns) == 0:
            raise ValueError("Returns series cannot be empty")

        if replace or strategy_name not in self.strategy_returns:
            # Replace or create new entry
            self.strategy_returns[strategy_name] = returns.copy()
        else:
            # Append to existing returns
            existing_returns = self.strategy_returns[strategy_name]
            combined_returns = pd.concat([existing_returns, returns])
            # Remove duplicates, keeping the latest
            combined_returns = combined_returns[~combined_returns.index.duplicated(keep='last')]
            self.strategy_returns[strategy_name] = combined_returns.sort_index()

    def calculate_correlation_matrix(
        self,
        method: Optional[str] = None,
        decay_factor: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> CorrelationMatrix:
        """
        Calculate correlation matrix for all strategies

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall', 'exponential')
            decay_factor: Decay factor for exponential weighting
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)

        Returns:
            CorrelationMatrix: Correlation matrix with metadata
        """
        if not self.strategy_returns:
            raise ValueError("No strategy returns data available")

        method = method or self.method
        decay_factor = decay_factor or self.decay_factor

        # Get aligned returns data
        aligned_returns = self._get_aligned_returns(start_date, end_date)

        if len(aligned_returns.columns) == 0:
            raise ValueError("No overlapping data found for correlation calculation")

        # Calculate correlation based on method
        if method == 'exponential':
            correlation_df = self._calculate_exponential_correlation(aligned_returns, decay_factor)
        else:
            correlation_df = aligned_returns.corr(method=method, min_periods=self.min_periods)

        # Handle NaN values - diagonal should be 1.0, off-diagonal can be 0.0
        if len(correlation_df) == 1:
            # Single strategy case - correlation with itself is 1.0
            correlation_df = correlation_df.fillna(1.0)
        else:
            # Multiple strategies - fill diagonal with 1.0, off-diagonal with 0.0
            np.fill_diagonal(correlation_df.values, 1.0)
            correlation_df = correlation_df.fillna(0.0)

        return CorrelationMatrix(
            matrix=correlation_df,
            method=method,
            timestamp=pd.Timestamp.now(),
            window_size=None,  # Full period
            decay_factor=decay_factor if method == 'exponential' else None
        )

    def calculate_rolling_correlation_matrix(
        self,
        window: Optional[int] = None,
        step: int = 1,
        method: Optional[str] = None
    ) -> List[CorrelationMatrix]:
        """
        Calculate rolling correlation matrices

        Args:
            window: Rolling window size. Uses config default if None.
            step: Step size between windows
            method: Correlation method

        Returns:
            List[CorrelationMatrix]: List of correlation matrices for each period
        """
        window = window or self.window_size
        method = method or self.method

        # Get aligned returns data
        aligned_returns = self._get_aligned_returns()

        if len(aligned_returns) < window:
            return []

        rolling_correlations = []

        for i in range(0, len(aligned_returns) - window + 1, step):
            window_data = aligned_returns.iloc[i:i + window]

            try:
                if method == 'exponential':
                    corr_matrix = self._calculate_exponential_correlation(window_data, self.decay_factor)
                else:
                    corr_matrix = window_data.corr(method=method, min_periods=self.min_periods)

                # Handle NaN values
                corr_matrix = corr_matrix.fillna(0.0)

                correlation_result = CorrelationMatrix(
                    matrix=corr_matrix,
                    method=method,
                    timestamp=window_data.index[-1],
                    window_size=window,
                    decay_factor=self.decay_factor if method == 'exponential' else None
                )

                rolling_correlations.append(correlation_result)

            except Exception:
                # Skip periods with insufficient data or calculation errors
                continue

        return rolling_correlations

    def calculate_risk_decomposition(
        self,
        weights: Dict[str, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> RiskDecomposition:
        """
        Calculate portfolio risk decomposition

        Args:
            weights: Strategy weights dictionary
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            RiskDecomposition: Risk decomposition results
        """
        # Validate weights
        self._validate_weights(weights)

        # Get aligned returns and correlation matrix
        aligned_returns = self._get_aligned_returns(start_date, end_date)
        correlation_matrix = self.calculate_correlation_matrix(
            start_date=start_date,
            end_date=end_date
        ).matrix

        # Calculate individual strategy risks (volatilities)
        individual_risks = {}
        for strategy in aligned_returns.columns:
            volatility = aligned_returns[strategy].std() * np.sqrt(252)  # Annualized
            individual_risks[strategy] = volatility

        # Create weight vector aligned with correlation matrix
        weight_vector = np.array([weights[strategy] for strategy in correlation_matrix.index])

        # Create volatility vector
        vol_vector = np.array([individual_risks[strategy] for strategy in correlation_matrix.index])

        # Create covariance matrix
        vol_matrix = np.outer(vol_vector, vol_vector)
        covariance_matrix = vol_matrix * correlation_matrix.values

        # Calculate portfolio variance and risk
        portfolio_variance = np.dot(weight_vector.T, np.dot(covariance_matrix, weight_vector))
        portfolio_risk = np.sqrt(portfolio_variance)

        # Calculate marginal contributions
        marginal_contributions = {}
        if portfolio_risk > 1e-10:
            marginal_contrib_vector = np.dot(covariance_matrix, weight_vector) / portfolio_risk
            for i, strategy in enumerate(correlation_matrix.index):
                marginal_contributions[strategy] = marginal_contrib_vector[i]
        else:
            marginal_contributions = {strategy: 0.0 for strategy in correlation_matrix.index}

        # Calculate component contributions
        component_contributions = {}
        for i, strategy in enumerate(correlation_matrix.index):
            component_contributions[strategy] = weights[strategy] * marginal_contributions[strategy]

        return RiskDecomposition(
            total_portfolio_risk=portfolio_risk,
            individual_risks=individual_risks,
            marginal_contributions=marginal_contributions,
            component_contributions=component_contributions,
            correlation_matrix=correlation_matrix
        )

    def calculate_diversification_metrics(
        self,
        weights: Dict[str, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> DiversificationMetrics:
        """
        Calculate portfolio diversification metrics

        Args:
            weights: Strategy weights dictionary
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            DiversificationMetrics: Diversification metrics
        """
        # Validate weights
        self._validate_weights(weights)

        # Get risk decomposition
        risk_decomp = self.calculate_risk_decomposition(weights, start_date, end_date)

        # Calculate diversification ratio
        weighted_avg_vol = sum(weights[strategy] * risk_decomp.individual_risks[strategy]
                              for strategy in weights.keys())

        if risk_decomp.total_portfolio_risk > 1e-10:
            diversification_ratio = weighted_avg_vol / risk_decomp.total_portfolio_risk
        else:
            diversification_ratio = 1.0

        # Calculate effective number of strategies (inverse of Herfindahl index of weights)
        herfindahl_index = sum(w**2 for w in weights.values())
        effective_number = 1.0 / herfindahl_index if herfindahl_index > 0 else 0.0

        # Calculate concentration ratio (weight of largest position)
        concentration_ratio = max(abs(w) for w in weights.values())

        return DiversificationMetrics(
            diversification_ratio=diversification_ratio,
            effective_number_of_strategies=effective_number,
            concentration_ratio=concentration_ratio,
            herfindahl_index=herfindahl_index
        )

    def _get_aligned_returns(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get aligned returns data for all strategies"""
        if not self.strategy_returns:
            return pd.DataFrame()

        # Combine all strategy returns into DataFrame
        returns_df = pd.DataFrame(self.strategy_returns)

        # Filter by date range if specified
        if start_date is not None:
            returns_df = returns_df.loc[start_date:]
        if end_date is not None:
            returns_df = returns_df.loc[:end_date]

        # Drop rows with all NaN values
        returns_df = returns_df.dropna(how='all')

        return returns_df

    def _calculate_exponential_correlation(
        self,
        returns: pd.DataFrame,
        decay_factor: float
    ) -> pd.DataFrame:
        """Calculate exponentially weighted correlation matrix"""
        if len(returns) < 2:
            # Not enough data for correlation
            n_strategies = len(returns.columns)
            return pd.DataFrame(
                np.eye(n_strategies),
                index=returns.columns,
                columns=returns.columns
            )

        # Calculate exponentially weighted covariance matrix
        ewm_cov = returns.ewm(alpha=1-decay_factor, min_periods=self.min_periods).cov()

        # Get the most recent covariance matrix
        if len(ewm_cov) > 0:
            # EWM cov returns a multi-index DataFrame, get the last time period
            last_date = ewm_cov.index.get_level_values(0)[-1]
            cov_matrix = ewm_cov.loc[last_date]
        else:
            # Fallback to standard covariance
            cov_matrix = returns.cov(min_periods=self.min_periods)

        # Convert covariance to correlation
        correlation_matrix = self._covariance_to_correlation(cov_matrix)

        return correlation_matrix

    def _covariance_to_correlation(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Convert covariance matrix to correlation matrix"""
        # Extract standard deviations (diagonal elements)
        std_devs = np.sqrt(np.diag(cov_matrix))

        # Handle zero variance case
        std_devs = np.where(std_devs == 0, 1e-10, std_devs)

        # Calculate correlation matrix
        correlation_values = cov_matrix.values / np.outer(std_devs, std_devs)

        # Handle numerical issues
        correlation_values = np.clip(correlation_values, -1.0, 1.0)

        return pd.DataFrame(
            correlation_values,
            index=cov_matrix.index,
            columns=cov_matrix.columns
        )

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """Validate portfolio weights"""
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")

        # Check if all strategies are present
        strategy_names = set(self.strategy_returns.keys())
        weight_names = set(weights.keys())

        if not weight_names.issubset(strategy_names):
            missing = weight_names - strategy_names
            raise ValueError(f"Unknown strategies in weights: {missing}")

        if not strategy_names.issubset(weight_names):
            missing = strategy_names - weight_names
            raise ValueError(f"Weights must be provided for all strategies: {missing}")

        # Check if weights sum to approximately 1
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1, got {weight_sum}")

    def get_correlation_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics of correlation analysis

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dict: Summary statistics
        """
        if not self.strategy_returns:
            return {}

        correlation_matrix = self.calculate_correlation_matrix(start_date, end_date)

        summary = {
            'number_of_strategies': len(correlation_matrix.matrix),
            'average_correlation': correlation_matrix.average_correlation,
            'max_correlation': correlation_matrix.max_correlation,
            'min_correlation': correlation_matrix.min_correlation,
            'method': correlation_matrix.method,
            'analysis_period': (start_date, end_date)
        }

        # Add pairwise correlation statistics
        n = len(correlation_matrix.matrix)
        if n > 1:
            upper_triangle = np.triu(correlation_matrix.matrix.values, k=1)
            valid_correlations = upper_triangle[upper_triangle != 0]

            if len(valid_correlations) > 0:
                summary.update({
                    'correlation_std': np.std(valid_correlations),
                    'positive_correlations': np.sum(valid_correlations > 0.1),
                    'negative_correlations': np.sum(valid_correlations < -0.1),
                    'high_correlations': np.sum(np.abs(valid_correlations) > 0.7)
                })

        return summary

    def detect_correlation_clusters(
        self,
        correlation_matrix: Optional[CorrelationMatrix] = None,
        threshold: float = 0.7
    ) -> Dict[str, List[str]]:
        """
        Detect clusters of highly correlated strategies

        Args:
            correlation_matrix: Correlation matrix to analyze. Uses latest if None.
            threshold: Correlation threshold for clustering

        Returns:
            Dict: Strategy clusters
        """
        if correlation_matrix is None:
            correlation_matrix = self.calculate_correlation_matrix()

        corr_matrix = correlation_matrix.matrix
        n_strategies = len(corr_matrix)

        if n_strategies <= 1:
            return {'cluster_1': list(corr_matrix.index)}

        # Simple clustering based on correlation threshold
        clusters = {}
        clustered_strategies = set()
        cluster_id = 1

        for i in range(n_strategies):
            strategy_i = corr_matrix.index[i]

            if strategy_i in clustered_strategies:
                continue

            # Start new cluster
            cluster = [strategy_i]
            clustered_strategies.add(strategy_i)

            # Find highly correlated strategies
            for j in range(i + 1, n_strategies):
                strategy_j = corr_matrix.index[j]

                if strategy_j in clustered_strategies:
                    continue

                correlation = abs(corr_matrix.iloc[i, j])
                if correlation >= threshold:
                    cluster.append(strategy_j)
                    clustered_strategies.add(strategy_j)

            clusters[f'cluster_{cluster_id}'] = cluster
            cluster_id += 1

        return clusters