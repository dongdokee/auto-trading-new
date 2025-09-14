"""
No-Lookahead Regime Detection System

Implements market regime detection using Hidden Markov Models (HMM) and GARCH volatility forecasting.
Prevents lookahead bias by only using historical data up to the current point.
Includes whipsaw prevention through sticky transitions and minimum regime duration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings from statistical models
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class NoLookAheadRegimeDetector:
    """
    Regime detection system that prevents lookahead bias

    Uses HMM to identify market regimes (BULL/BEAR/SIDEWAYS) and GARCH for volatility forecasting.
    Implements sticky transitions and minimum regime duration to prevent excessive switching.
    """

    def __init__(
        self,
        retrain_interval: int = 180,
        min_train_samples: int = 500,
        transition_penalty: float = 0.9,
        min_regime_duration: int = 5
    ):
        """
        Initialize regime detector

        Args:
            retrain_interval: Days between model retraining
            min_train_samples: Minimum samples needed for training
            transition_penalty: Self-transition probability (higher = more sticky)
            min_regime_duration: Minimum days before regime change allowed
        """
        self.retrain_interval = retrain_interval
        self.min_train_samples = min_train_samples
        self.transition_penalty = transition_penalty
        self.min_regime_duration = min_regime_duration

        # Model state
        self.hmm_model = None
        self.garch_model = None
        self.garch_result = None
        self.last_train_index = -1

        # Regime tracking
        self.current_regime: Optional[str] = None
        self.regime_duration = 0
        self.state_label_map: Dict[int, str] = {}

    def fit(self, historical_data: pd.DataFrame, end_index: int) -> bool:
        """
        Train models using data up to end_index (prevents lookahead bias)

        Args:
            historical_data: Complete historical DataFrame
            end_index: Last index to use for training (exclusive)

        Returns:
            bool: True if training successful, False otherwise
        """
        if end_index < self.min_train_samples:
            return False

        try:
            # Use only historical data up to end_index
            train_data = historical_data.iloc[:end_index].copy()

            # Prepare features
            features = self._prepare_features(train_data)
            if features.size == 0:
                return False

            # Train HMM with sticky transitions
            self._fit_hmm(features)

            # Train GARCH model
            self._fit_garch(train_data)

            # Relabel states based on returns
            self.state_label_map = self._relabel_states_after_training(features)

            self.last_train_index = end_index
            return True

        except Exception as e:
            print(f"Model fitting failed: {e}")
            return False

    def _fit_hmm(self, features: np.ndarray) -> None:
        """Fit HMM model with sticky transitions"""
        try:
            from hmmlearn.hmm import GaussianHMM

            self.hmm_model = GaussianHMM(
                n_components=3,
                covariance_type="full",
                n_iter=100,
                init_params="mc"
            )

            # Set sticky transition matrix
            trans_mat = np.full((3, 3), (1 - self.transition_penalty) / 2)
            np.fill_diagonal(trans_mat, self.transition_penalty)
            self.hmm_model.transmat_ = trans_mat
            self.hmm_model.startprob_ = np.array([1/3, 1/3, 1/3])

            # Fit model
            self.hmm_model.fit(features)

        except ImportError:
            print("hmmlearn not available, using mock HMM")
            self.hmm_model = None
        except Exception as e:
            print(f"HMM fitting failed: {e}")
            self.hmm_model = None

    def _fit_garch(self, train_data: pd.DataFrame) -> None:
        """Fit GARCH model for volatility forecasting"""
        try:
            from arch import arch_model

            returns = np.diff(np.log(train_data['close']))

            self.garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
            self.garch_result = self.garch_model.fit(disp='off')

        except ImportError:
            print("arch not available, using historical volatility")
            self.garch_model = None
            self.garch_result = None
        except Exception as e:
            print(f"GARCH fitting failed: {e}")
            self.garch_model = None
            self.garch_result = None

    def detect_regime(self, data: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """
        Detect current market regime (no lookahead bias)

        Args:
            data: Complete price data
            current_index: Current time index

        Returns:
            dict: Regime information including regime, confidence, volatility forecast
        """
        # Check if retraining is needed
        if current_index - self.last_train_index > self.retrain_interval:
            self.fit(data, current_index)

        # Prepare features using only current and historical data
        current_data = data.iloc[:current_index + 1].copy()
        features = self._prepare_features(current_data)

        if features.size == 0:
            return self._default_regime_info()

        # Get regime prediction
        regime_info = self._predict_regime(features)

        # Apply whipsaw prevention
        final_regime = self._apply_whipsaw_prevention(regime_info['regime'], regime_info['confidence'])

        # Get volatility forecast
        volatility_forecast = self._get_volatility_forecast()

        return {
            'regime': final_regime,
            'confidence': regime_info['confidence'],
            'regime_probabilities': regime_info['probabilities'],
            'volatility_forecast': volatility_forecast,
            'duration': self.regime_duration
        }

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract and standardize features for regime detection

        Args:
            data: Price data DataFrame

        Returns:
            np.ndarray: Standardized features [returns, volume_ratio, rsi]
        """
        if len(data) < 21:  # Need minimum data for indicators
            return np.array([])

        try:
            # Calculate returns
            returns = np.diff(np.log(data['close']))

            # Volume profile
            volume_ma = data['volume'].rolling(20).mean()
            volume_ratio = data['volume'] / (volume_ma + 1e-10)

            # RSI
            rsi = self._calculate_rsi(data['close'])

            # Align all features
            min_length = min(len(returns), len(volume_ratio) - 1, len(rsi) - 1)
            if min_length < 20:
                return np.array([])

            features = np.column_stack([
                returns[-min_length:],
                volume_ratio.values[-min_length-1:-1],
                rsi.values[-min_length-1:-1]
            ])

            # Remove any NaN or infinite values
            mask = np.isfinite(features).all(axis=1)
            features = features[mask]

            if len(features) == 0:
                return np.array([])

            # Z-score standardization
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            return features

        except Exception as e:
            print(f"Feature preparation failed: {e}")
            return np.array([])

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _predict_regime(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict regime using HMM model"""
        if self.hmm_model is None:
            return {
                'regime': 'NEUTRAL',
                'confidence': 0.5,
                'probabilities': [0.33, 0.33, 0.34]
            }

        try:
            # Get state probabilities
            state_probs = self.hmm_model.predict_proba(features[-1:])
            predicted_state = np.argmax(state_probs[0])
            confidence = float(state_probs[0, predicted_state])

            # Map state to regime label
            candidate_regime = self.state_label_map.get(predicted_state, 'NEUTRAL')

            return {
                'regime': candidate_regime,
                'confidence': confidence,
                'probabilities': state_probs[0].tolist()
            }

        except Exception as e:
            print(f"Regime prediction failed: {e}")
            return {
                'regime': 'NEUTRAL',
                'confidence': 0.5,
                'probabilities': [0.33, 0.33, 0.34]
            }

    def _apply_whipsaw_prevention(self, candidate_regime: str, confidence: float) -> str:
        """Apply whipsaw prevention logic"""
        if self.current_regime is None:
            self.current_regime = candidate_regime
            self.regime_duration = 1
            return candidate_regime

        if candidate_regime == self.current_regime:
            self.regime_duration += 1
            return self.current_regime

        # Regime change candidate
        # Require sufficient duration and high confidence
        if (self.regime_duration >= self.min_regime_duration and confidence > 0.7):
            self.current_regime = candidate_regime
            self.regime_duration = 1
            return candidate_regime

        # Stay in current regime
        self.regime_duration += 1
        return self.current_regime

    def _get_volatility_forecast(self) -> float:
        """Get volatility forecast from GARCH model"""
        if self.garch_result is not None:
            try:
                forecast = self.garch_result.forecast(horizon=1)
                volatility_forecast = float(forecast.variance.values[-1, 0] ** 0.5)
                return volatility_forecast
            except Exception as e:
                print(f"GARCH forecast failed: {e}")

        # Default volatility estimate
        return 0.02

    def _relabel_states_after_training(self, features: np.ndarray) -> Dict[int, str]:
        """
        Relabel HMM states based on mean returns to ensure consistency

        Args:
            features: Training features used for HMM

        Returns:
            dict: Mapping from HMM state to regime label
        """
        if self.hmm_model is None:
            return {}

        try:
            states = self.hmm_model.predict(features)
            returns = features[:, 0]  # First feature is returns

            # Calculate mean return for each state
            state_returns = {}
            for state in range(self.hmm_model.n_components):
                mask = (states == state)
                if mask.sum() > 0:
                    state_returns[state] = returns[mask].mean()
                else:
                    state_returns[state] = 0.0

            # Sort states by mean return
            sorted_states = sorted(state_returns.items(), key=lambda x: x[1], reverse=True)

            # Assign labels based on return ranking
            label_map = {}
            if len(sorted_states) >= 3:
                label_map[sorted_states[0][0]] = 'BULL'      # Highest returns
                label_map[sorted_states[2][0]] = 'BEAR'      # Lowest returns
                label_map[sorted_states[1][0]] = 'SIDEWAYS'  # Middle returns
            else:
                # Fallback mapping
                for i, (state, _) in enumerate(sorted_states):
                    labels = ['BULL', 'SIDEWAYS', 'BEAR']
                    label_map[state] = labels[min(i, 2)]

            return label_map

        except Exception as e:
            print(f"State relabeling failed: {e}")
            return {0: 'BULL', 1: 'BEAR', 2: 'SIDEWAYS'}

    def _default_regime_info(self) -> Dict[str, Any]:
        """Return default regime information when detection fails"""
        return {
            'regime': 'NEUTRAL',
            'confidence': 0.5,
            'regime_probabilities': [0.33, 0.33, 0.34],
            'volatility_forecast': 0.02,
            'duration': 0
        }

    def get_regime_history(self) -> Dict[str, Any]:
        """Get historical regime information for analysis"""
        return {
            'current_regime': self.current_regime,
            'regime_duration': self.regime_duration,
            'last_train_index': self.last_train_index,
            'state_label_map': self.state_label_map,
            'model_fitted': self.hmm_model is not None
        }

    def reset(self) -> None:
        """Reset detector state for new analysis"""
        self.current_regime = None
        self.regime_duration = 0
        self.last_train_index = -1
        self.hmm_model = None
        self.garch_model = None
        self.garch_result = None
        self.state_label_map = {}