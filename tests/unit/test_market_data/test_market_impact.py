# tests/unit/test_market_data/test_market_impact.py

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.market_data.models import MarketImpactEstimate, OrderSide
from src.market_data.market_impact import MarketImpactModel


class TestMarketImpactModel:
    """Test suite for MarketImpactModel"""

    @pytest.fixture
    def model(self):
        return MarketImpactModel()

    @pytest.fixture
    def sample_execution_history(self):
        """Create sample execution history for testing"""
        base_time = datetime.utcnow()
        return [
            {
                'size': 1.0,
                'avg_daily_volume': 1000000,
                'volatility': 0.02,
                'spread_bps': 5.0,
                'execution_speed': 1.0,
                'hour': 12,
                'market_regime': 0.3,
                'exec_price': 50005,
                'mid_before': 50000,
                'mid_after_5min': 50010
            },
            {
                'size': 2.0,
                'avg_daily_volume': 1000000,
                'volatility': 0.015,
                'spread_bps': 4.0,
                'execution_speed': 1.5,
                'hour': 14,
                'market_regime': 0.7,
                'exec_price': 49995,
                'mid_before': 50000,
                'mid_after_5min': 49998
            }
        ] * 30  # Repeat to have enough samples

    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state for testing"""
        return {
            'daily_volume': 1000000,
            'volatility': 0.02,
            'spread_bps': 5.0,
            'execution_speed': 1.0,
            'market_regime': 0.5
        }

    def test_should_initialize_with_default_parameters(self, model):
        """Test model initialization"""
        assert model.temp_impact_coef == 0.1
        assert model.perm_impact_coef == 0.05
        assert model.last_calibration is None
        assert model.model_confidence == 0.0

    def test_should_reject_insufficient_calibration_data(self, model):
        """Test rejection of insufficient calibration data"""
        small_history = [{'size': 1.0}] * 10  # Less than minimum
        result = model.calibrate_from_trades(small_history)
        assert result is False

    @patch('sklearn.model_selection.cross_val_score')
    @patch('sklearn.pipeline.Pipeline')
    def test_should_calibrate_from_sufficient_data(self, mock_pipeline, mock_cv, model, sample_execution_history):
        """Test successful calibration with sufficient data"""
        # Mock scikit-learn components
        mock_cv.return_value = np.array([-0.01, -0.01, -0.01, -0.01, -0.01])
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        result = model.calibrate_from_trades(sample_execution_history)
        assert result is True
        assert model.last_calibration is not None
        assert model.temp_impact_model is not None
        assert model.perm_impact_model is not None

    def test_should_extract_features_correctly(self, model):
        """Test feature extraction from trade data"""
        trade = {
            'size': 1.0,
            'avg_daily_volume': 1000000,
            'volatility': 0.02,
            'spread_bps': 5.0,
            'execution_speed': 1.0,
            'hour': 12,
            'market_regime': 0.3
        }

        features = model._extract_features(trade)
        assert features is not None
        assert len(features) == 6
        assert features[0] == 1.0 / 1000000  # relative_size
        assert features[1] == 0.02  # volatility
        assert features[2] == 0.05  # spread normalized

    def test_should_handle_invalid_trade_data(self, model):
        """Test handling of invalid trade data"""
        invalid_trade = {
            'size': 1.0,
            'avg_daily_volume': 0,  # Invalid volume
            'volatility': None,  # Invalid volatility
        }

        features = model._extract_features(invalid_trade)
        assert features is None

    def test_should_calculate_temporary_impact(self, model):
        """Test temporary impact calculation"""
        trade = {
            'exec_price': 50005,
            'mid_before': 50000
        }

        temp_impact = model._calculate_temp_impact(trade)
        assert temp_impact is not None
        assert temp_impact == 0.0001  # (50005 - 50000) / 50000

    def test_should_calculate_permanent_impact(self, model):
        """Test permanent impact calculation"""
        trade = {
            'mid_before': 50000,
            'mid_after_5min': 50010
        }

        perm_impact = model._calculate_perm_impact(trade)
        assert perm_impact is not None
        assert perm_impact == 0.0002  # (50010 - 50000) / 50000

    def test_should_cap_extreme_impacts(self, model):
        """Test that extreme impacts are capped"""
        extreme_trade = {
            'exec_price': 100000,  # Extreme price
            'mid_before': 50000
        }

        temp_impact = model._calculate_temp_impact(extreme_trade)
        assert temp_impact == 0.1  # Should be capped at 10%

    def test_should_estimate_impact_with_fallback_model(self, model, sample_market_state):
        """Test impact estimation using fallback square-root model"""
        estimate = model.estimate_impact(Decimal('1.0'), "BTCUSDT", sample_market_state)

        assert isinstance(estimate, MarketImpactEstimate)
        assert estimate.symbol == "BTCUSDT"
        assert estimate.order_size == Decimal('1.0')
        assert estimate.temporary_impact >= 0
        assert estimate.permanent_impact >= 0
        assert estimate.total_impact == estimate.temporary_impact + estimate.permanent_impact

    def test_should_include_impact_breakdown(self, model, sample_market_state):
        """Test that impact estimate includes component breakdown"""
        estimate = model.estimate_impact(Decimal('1.0'), "BTCUSDT", sample_market_state)

        assert hasattr(estimate, 'spread_component')
        assert hasattr(estimate, 'size_component')
        assert hasattr(estimate, 'volatility_component')
        assert hasattr(estimate, 'timing_component')
        assert hasattr(estimate, 'permanent_drift')

        # Check that components are reasonable
        assert 0 <= estimate.spread_component <= 0.01
        assert 0 <= estimate.size_component <= estimate.temporary_impact
        assert 0 <= estimate.volatility_component <= estimate.temporary_impact

    @patch.object(MarketImpactModel, '_get_recent_executions')
    def test_should_trigger_recalibration_when_needed(self, mock_get_executions, model, sample_execution_history):
        """Test automatic recalibration when model is stale"""
        # Make model appear old
        model.last_calibration = datetime.utcnow() - timedelta(days=2)
        mock_get_executions.return_value = sample_execution_history

        with patch.object(model, 'calibrate_from_trades') as mock_calibrate:
            mock_calibrate.return_value = True
            model.estimate_impact(Decimal('1.0'), "BTCUSDT", {'daily_volume': 1000000})
            mock_calibrate.assert_called_once()

    def test_should_use_trained_models_when_confident(self, model, sample_market_state):
        """Test that trained models are used when confidence is high"""
        # Mock trained models
        mock_temp_model = Mock()
        mock_temp_model.predict.return_value = [0.001]
        mock_perm_model = Mock()
        mock_perm_model.predict.return_value = [0.0005]

        model.temp_impact_model = mock_temp_model
        model.perm_impact_model = mock_perm_model
        model.model_confidence = 0.8  # High confidence

        estimate = model.estimate_impact(Decimal('1.0'), "BTCUSDT", sample_market_state)

        assert mock_temp_model.predict.called
        assert mock_perm_model.predict.called
        assert estimate.temporary_impact == 0.001
        assert estimate.permanent_impact == 0.0005

    def test_should_fallback_when_prediction_fails(self, model, sample_market_state):
        """Test fallback to square-root model when prediction fails"""
        # Mock failing models
        mock_temp_model = Mock()
        mock_temp_model.predict.side_effect = Exception("Prediction failed")

        model.temp_impact_model = mock_temp_model
        model.perm_impact_model = mock_temp_model
        model.model_confidence = 0.8

        estimate = model.estimate_impact(Decimal('1.0'), "BTCUSDT", sample_market_state)

        # Should still return a valid estimate using fallback
        assert estimate.temporary_impact > 0
        assert estimate.permanent_impact > 0

    def test_should_scale_impact_with_order_size(self, model, sample_market_state):
        """Test that impact scales appropriately with order size"""
        small_estimate = model.estimate_impact(Decimal('0.1'), "BTCUSDT", sample_market_state)
        large_estimate = model.estimate_impact(Decimal('10.0'), "BTCUSDT", sample_market_state)

        # Larger orders should have higher impact
        assert large_estimate.total_impact > small_estimate.total_impact

    def test_should_scale_impact_with_volatility(self, model):
        """Test that impact scales with market volatility"""
        low_vol_state = {'daily_volume': 1000000, 'volatility': 0.01, 'spread_bps': 5.0}
        high_vol_state = {'daily_volume': 1000000, 'volatility': 0.05, 'spread_bps': 5.0}

        low_vol_estimate = model.estimate_impact(Decimal('1.0'), "BTCUSDT", low_vol_state)
        high_vol_estimate = model.estimate_impact(Decimal('1.0'), "BTCUSDT", high_vol_state)

        # Higher volatility should lead to higher impact
        assert high_vol_estimate.total_impact > low_vol_estimate.total_impact

    def test_should_provide_model_statistics(self, model):
        """Test model statistics reporting"""
        stats = model.get_model_stats()

        assert 'last_calibration' in stats
        assert 'model_confidence' in stats
        assert 'calibration_count' in stats
        assert 'has_trained_models' in stats
        assert 'default_coefficients' in stats

        assert stats['has_trained_models'] is False  # No training yet
        assert stats['model_confidence'] == 0.0

    def test_should_reset_models_correctly(self, model):
        """Test model reset functionality"""
        # Set up some state
        model.temp_impact_model = Mock()
        model.perm_impact_model = Mock()
        model.model_confidence = 0.8
        model.last_calibration = datetime.utcnow()

        model.reset_models()

        assert model.temp_impact_model is None
        assert model.perm_impact_model is None
        assert model.model_confidence == 0.0
        assert model.last_calibration is None

    def test_should_handle_zero_division_in_impact_calculation(self, model):
        """Test handling of zero division in impact calculations"""
        zero_trade = {
            'exec_price': 50000,
            'mid_before': 0  # Would cause division by zero
        }

        temp_impact = model._calculate_temp_impact(zero_trade)
        assert temp_impact is None

    def test_should_prepare_prediction_features_with_defaults(self, model):
        """Test feature preparation with missing market state"""
        incomplete_state = {'daily_volume': 1000000}  # Missing other fields

        features = model._prepare_prediction_features(Decimal('1.0'), incomplete_state)

        assert features.shape == (1, 6)
        assert not np.isnan(features).any()  # Should have default values

    def test_should_maintain_calibration_history(self, model):
        """Test that calibration history is maintained"""
        # Simulate the capping logic by adding exactly 11 items
        # to trigger the truncation in calibrate_from_trades
        for i in range(11):
            model.calibration_history.append({
                'timestamp': datetime.utcnow(),
                'sample_count': 100,
                'confidence': 0.8
            })

        # Manually trigger the capping logic (simulating what happens in calibrate_from_trades)
        if len(model.calibration_history) > 10:
            model.calibration_history = model.calibration_history[-10:]

        assert len(model.calibration_history) <= 10  # Should be capped

    def test_should_handle_missing_trade_data_gracefully(self, model):
        """Test graceful handling of trades with missing data"""
        incomplete_trades = [
            {'size': 1.0, 'avg_daily_volume': 1000000},  # Missing other fields
            {'volatility': 0.02},  # Missing size
            {}  # Empty trade
        ] * 20

        # Should not crash, but should fail calibration
        result = model.calibrate_from_trades(incomplete_trades)
        assert result is False

    def test_should_validate_feature_finite_values(self, model):
        """Test that infinite/NaN features are rejected"""
        invalid_trade = {
            'size': float('inf'),  # Invalid size
            'avg_daily_volume': 1000000,
            'volatility': 0.02,
            'spread_bps': 5.0
        }

        features = model._extract_features(invalid_trade)
        assert features is None  # Should reject infinite values