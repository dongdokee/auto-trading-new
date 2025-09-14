"""
Unit tests for NoLookAheadRegimeDetector
Tests the HMM/GARCH-based market regime detection system with whipsaw prevention.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.strategy_engine.regime_detector import NoLookAheadRegimeDetector


class TestNoLookAheadRegimeDetector:
    """Test NoLookAheadRegimeDetector functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.detector = NoLookAheadRegimeDetector()

        # Create synthetic market data for testing
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2024-01-01', periods=200, freq='D')

        # Generate synthetic OHLCV data with different regimes
        base_price = 50000
        returns = np.random.normal(0, 0.02, 200)  # 2% daily volatility

        # Add regime patterns
        returns[50:100] = np.random.normal(0.01, 0.015, 50)   # Bull market
        returns[100:150] = np.random.normal(-0.01, 0.025, 50)  # Bear market
        returns[150:] = np.random.normal(0, 0.01, 50)         # Low volatility

        prices = base_price * np.exp(np.cumsum(returns))

        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, 200)),
            'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, 200))),
            'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, 200))),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 200)
        })

    def test_should_initialize_with_default_parameters(self):
        """Test detector initialization with default parameters"""
        assert self.detector.hmm_model is None
        assert self.detector.garch_model is None
        assert self.detector.last_train_index == -1
        assert self.detector.retrain_interval == 180
        assert self.detector.min_train_samples == 500
        assert self.detector.transition_penalty == 0.9
        assert self.detector.min_regime_duration == 5
        assert self.detector.current_regime is None

    def test_should_initialize_with_custom_parameters(self):
        """Test detector initialization with custom parameters"""
        detector = NoLookAheadRegimeDetector(
            retrain_interval=90,
            min_train_samples=300,
            transition_penalty=0.8,
            min_regime_duration=10
        )

        assert detector.retrain_interval == 90
        assert detector.min_train_samples == 300
        assert detector.transition_penalty == 0.8
        assert detector.min_regime_duration == 10

    def test_should_prepare_features_with_sufficient_data(self):
        """Test feature preparation with sufficient data"""
        features = self.detector._prepare_features(self.test_data)

        assert features.shape[0] > 0
        assert features.shape[1] == 3  # returns, volume_ratio, rsi
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_should_handle_insufficient_data_in_features(self):
        """Test feature preparation with insufficient data"""
        small_data = self.test_data.head(10)  # Too small for features
        features = self.detector._prepare_features(small_data)

        assert features.size == 0

    def test_should_calculate_rsi_correctly(self):
        """Test RSI calculation"""
        prices = pd.Series([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                           46.03, 46.83, 46.69, 46.45, 46.59, 46.3, 46.02, 46.74, 46.01, 46.688])

        rsi = self.detector._calculate_rsi(prices, period=14)

        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)
        assert len(valid_rsi) > 0

    def test_should_fit_models_with_sufficient_data(self):
        """Test model fitting with sufficient training data"""
        # Create larger dataset for training
        large_data = pd.concat([self.test_data] * 3, ignore_index=True)  # 600 samples

        result = self.detector.fit(large_data, end_index=550)

        # Should succeed even without the optional libraries
        # The detector handles missing libraries gracefully
        assert result is True
        assert self.detector.last_train_index == 550

    def test_should_fail_fit_with_insufficient_data(self):
        """Test that fit fails with insufficient training data"""
        result = self.detector.fit(self.test_data, end_index=100)  # Less than min_train_samples

        assert result is False
        assert self.detector.hmm_model is None
        assert self.detector.garch_model is None

    def test_should_relabel_states_after_training(self):
        """Test state relabeling based on mean returns"""
        # Setup mock HMM model with predict method
        mock_hmm_instance = Mock()
        mock_hmm_instance.n_components = 3
        mock_hmm_instance.predict.return_value = np.array([0, 1, 2, 0, 1, 2] * 10)  # Cycling states

        self.detector.hmm_model = mock_hmm_instance

        # Create features with known return patterns
        features = np.array([
            [0.02, 0.5, 60],    # Positive returns -> BULL
            [-0.02, 0.5, 40],   # Negative returns -> BEAR
            [0.001, 0.5, 50]    # Small returns -> SIDEWAYS
        ] * 20)  # Repeat pattern

        label_map = self.detector._relabel_states_after_training(features)

        assert isinstance(label_map, dict)
        assert len(label_map) <= 3
        assert all(label in ['BULL', 'BEAR', 'SIDEWAYS'] for label in label_map.values())

    def test_should_detect_regime_without_models(self):
        """Test regime detection when models are not fitted"""
        result = self.detector.detect_regime(self.test_data, current_index=100)

        assert result['regime'] == 'NEUTRAL'
        assert result['confidence'] == 0.5

    def test_should_detect_regime_with_fitted_models(self):
        """Test regime detection with properly fitted models"""
        # Setup mock HMM model
        mock_hmm_instance = Mock()
        mock_hmm_instance.n_components = 3
        mock_hmm_instance.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])  # High confidence in state 1

        # Setup mock GARCH model
        mock_garch_result = Mock()
        mock_forecast = Mock()
        mock_forecast.variance.values = np.array([[0.0004]])  # 2% volatility
        mock_garch_result.forecast.return_value = mock_forecast

        # Set up detector state
        self.detector.hmm_model = mock_hmm_instance
        self.detector.garch_result = mock_garch_result
        self.detector.state_label_map = {0: 'BULL', 1: 'BEAR', 2: 'SIDEWAYS'}

        result = self.detector.detect_regime(self.test_data, current_index=100)

        assert result['regime'] in ['BULL', 'BEAR', 'SIDEWAYS', 'NEUTRAL']
        assert 0 <= result['confidence'] <= 1
        assert 'volatility_forecast' in result
        assert result['volatility_forecast'] == pytest.approx(0.02, rel=1e-2)

    def test_should_prevent_whipsaw_with_min_duration(self):
        """Test whipsaw prevention with minimum regime duration"""
        # Setup detector with short regime duration for testing
        self.detector.min_regime_duration = 3
        self.detector.current_regime = 'BULL'
        self.detector.regime_duration = 2  # Less than minimum

        # Mock models to suggest regime change
        mock_hmm_instance = Mock()
        mock_hmm_instance.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # High confidence in SIDEWAYS
        self.detector.hmm_model = mock_hmm_instance
        self.detector.state_label_map = {0: 'BULL', 1: 'BEAR', 2: 'SIDEWAYS'}

        result = self.detector.detect_regime(self.test_data, current_index=100)

        # Should remain in BULL because duration is too short
        assert result['regime'] == 'BULL'
        assert self.detector.regime_duration == 3  # Incremented

    def test_should_change_regime_with_sufficient_duration_and_confidence(self):
        """Test regime change with sufficient duration and confidence"""
        # Setup detector
        self.detector.min_regime_duration = 3
        self.detector.current_regime = 'BULL'
        self.detector.regime_duration = 5  # Greater than minimum

        # Mock models to suggest regime change with high confidence
        mock_hmm_instance = Mock()
        mock_hmm_instance.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # High confidence
        self.detector.hmm_model = mock_hmm_instance
        self.detector.state_label_map = {0: 'BULL', 1: 'BEAR', 2: 'SIDEWAYS'}

        result = self.detector.detect_regime(self.test_data, current_index=100)

        # Should change to SIDEWAYS
        assert result['regime'] == 'SIDEWAYS'
        assert self.detector.regime_duration == 1  # Reset

    def test_should_handle_retrain_trigger(self):
        """Test automatic retraining when interval is exceeded"""
        self.detector.last_train_index = 50
        self.detector.retrain_interval = 100
        current_index = 200  # Exceeds retrain interval

        # Mock the fit method
        original_fit = self.detector.fit
        self.detector.fit = Mock(return_value=True)

        self.detector.detect_regime(self.test_data, current_index=current_index)

        self.detector.fit.assert_called_once_with(self.test_data, current_index)

    def test_should_handle_edge_case_empty_features(self):
        """Test handling of empty features"""
        # Create detector that will return empty features
        self.detector._prepare_features = Mock(return_value=np.array([]))

        result = self.detector.detect_regime(self.test_data, current_index=100)

        assert result['regime'] == 'NEUTRAL'
        assert result['confidence'] == 0.5

    def test_should_return_complete_regime_info(self):
        """Test that detect_regime returns all expected information"""
        result = self.detector.detect_regime(self.test_data, current_index=100)

        required_keys = ['regime', 'confidence', 'regime_probabilities', 'volatility_forecast', 'duration']
        for key in required_keys:
            assert key in result

        assert isinstance(result['regime'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['regime_probabilities'], list)
        assert isinstance(result['volatility_forecast'], float)
        assert isinstance(result['duration'], int)


class TestRegimeDetectorIntegration:
    """Integration tests for regime detector with real-like data"""

    def test_should_work_with_trending_data(self):
        """Test regime detection with clearly trending data"""
        # Create strong uptrend data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        base_price = 50000
        trend_returns = np.random.normal(0.01, 0.005, 100)  # Strong positive trend
        prices = base_price * np.exp(np.cumsum(trend_returns))

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })

        detector = NoLookAheadRegimeDetector(min_train_samples=50)
        result = detector.detect_regime(data, current_index=90)

        # Should detect some regime (not necessarily BULL due to insufficient training data)
        assert result['regime'] in ['BULL', 'BEAR', 'SIDEWAYS', 'NEUTRAL']
        assert 0 <= result['confidence'] <= 1

    def test_should_work_with_sideways_data(self):
        """Test regime detection with sideways market data"""
        # Create sideways market data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        base_price = 50000
        sideways_returns = np.random.normal(0, 0.01, 100)  # No trend, low volatility
        prices = base_price + np.cumsum(sideways_returns * base_price * 0.0001)  # Small absolute changes

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        })

        detector = NoLookAheadRegimeDetector(min_train_samples=50)
        result = detector.detect_regime(data, current_index=90)

        assert result['regime'] in ['BULL', 'BEAR', 'SIDEWAYS', 'NEUTRAL']
        assert result['volatility_forecast'] > 0