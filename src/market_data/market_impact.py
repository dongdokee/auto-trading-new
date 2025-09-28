# src/market_data/market_impact.py

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

from .models import MarketImpactEstimate, OrderSide


class MarketImpactModel:
    """Dynamic calibration-based market impact model"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Default impact coefficients (fallback values)
        self.temp_impact_coef = 0.1
        self.perm_impact_coef = 0.05

        # Calibration state
        self.last_calibration: Optional[datetime] = None
        self.calibration_interval = 86400  # 24 hours in seconds
        self.min_calibration_samples = 50

        # Trained models (None means fallback to square-root model)
        self.temp_impact_model: Optional[Tuple[Any, Any]] = None
        self.perm_impact_model: Optional[Tuple[Any, Any]] = None

        # Model metadata
        self.model_confidence = 0.0
        self.calibration_history: List[Dict] = []

    def calibrate_from_trades(self, execution_history: List[Dict]) -> bool:
        """
        Calibrate impact coefficients from execution history

        Args:
            execution_history: List of historical execution records

        Returns:
            bool: True if calibration succeeded, False otherwise
        """
        if len(execution_history) < self.min_calibration_samples:
            self.logger.warning(
                f"Insufficient data for calibration: {len(execution_history)} < {self.min_calibration_samples}"
            )
            return False

        try:
            # Prepare feature matrix and target vectors
            X, y_temp, y_perm = self._prepare_calibration_data(execution_history)

            if len(X) < self.min_calibration_samples:
                self.logger.warning("Insufficient valid samples after data preparation")
                return False

            # Train models
            success = self._train_impact_models(X, y_temp, y_perm)

            if success:
                self.last_calibration = datetime.utcnow()
                self.logger.info(f"Successfully calibrated impact models on {len(X)} samples")

                # Store calibration metadata
                self.calibration_history.append({
                    'timestamp': self.last_calibration,
                    'sample_count': len(X),
                    'confidence': self.model_confidence
                })

                # Keep only last 10 calibrations
                if len(self.calibration_history) > 10:
                    self.calibration_history = self.calibration_history[-10:]

            return success

        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False

    def _prepare_calibration_data(self, execution_history: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and targets for model training

        Args:
            execution_history: Historical execution data

        Returns:
            Tuple: (features, temp_impacts, perm_impacts)
        """
        X, y_temp, y_perm = [], [], []

        for trade in execution_history:
            try:
                # Extract features
                features = self._extract_features(trade)
                if features is None:
                    continue

                # Calculate impact measures
                temp_impact = self._calculate_temp_impact(trade)
                perm_impact = self._calculate_perm_impact(trade)

                if temp_impact is not None and perm_impact is not None:
                    X.append(features)
                    y_temp.append(temp_impact)
                    y_perm.append(perm_impact)

            except (KeyError, ValueError, TypeError) as e:
                self.logger.debug(f"Skipping invalid trade record: {e}")
                continue

        return np.array(X), np.array(y_temp), np.array(y_perm)

    def _extract_features(self, trade: Dict) -> Optional[List[float]]:
        """
        Extract features for impact prediction

        Args:
            trade: Single trade record

        Returns:
            Optional[List[float]]: Feature vector or None if invalid
        """
        try:
            # Feature 1: Relative order size (order_size / daily_volume)
            daily_volume = trade.get('avg_daily_volume', 0)
            if daily_volume <= 0:
                return None

            relative_size = float(trade['size']) / daily_volume

            # Feature 2: Market volatility (recent price volatility)
            volatility = trade.get('volatility', 0.01)

            # Feature 3: Spread component (spread in basis points)
            spread_bps = trade.get('spread_bps', 5.0)

            # Feature 4: Execution speed (1.0 = immediate, higher = slower)
            execution_speed = trade.get('execution_speed', 1.0)

            # Feature 5: Time of day factor (normalized hour)
            hour = trade.get('hour', 12)
            time_factor = np.sin(2 * np.pi * hour / 24)

            # Feature 6: Market regime (0 = trending, 1 = mean-reverting)
            market_regime = trade.get('market_regime', 0.5)

            features = [
                relative_size,
                volatility,
                spread_bps / 100.0,  # Normalize
                execution_speed,
                time_factor,
                market_regime
            ]

            # Validate features
            if any(not np.isfinite(f) for f in features):
                return None

            return features

        except (KeyError, ValueError, TypeError):
            return None

    def _calculate_temp_impact(self, trade: Dict) -> Optional[float]:
        """Calculate temporary impact from trade data"""
        try:
            exec_price = float(trade['exec_price'])
            mid_before = float(trade['mid_before'])

            if mid_before <= 0:
                return None

            temp_impact = abs(exec_price - mid_before) / mid_before
            return min(temp_impact, 0.1)  # Cap at 10%

        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return None

    def _calculate_perm_impact(self, trade: Dict) -> Optional[float]:
        """Calculate permanent impact from trade data"""
        try:
            mid_before = float(trade['mid_before'])
            mid_after_5min = float(trade.get('mid_after_5min', mid_before))

            if mid_before <= 0:
                return None

            perm_impact = abs(mid_after_5min - mid_before) / mid_before
            return min(perm_impact, 0.05)  # Cap at 5%

        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return None

    def _train_impact_models(self, X: np.ndarray, y_temp: np.ndarray,
                           y_perm: np.ndarray) -> bool:
        """
        Train polynomial regression models for impact prediction

        Args:
            X: Feature matrix
            y_temp: Temporary impact targets
            y_perm: Permanent impact targets

        Returns:
            bool: True if training succeeded
        """
        try:
            from sklearn.preprocessing import PolynomialFeatures, StandardScaler
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import mean_squared_error

            # Create polynomial features
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

            # Create pipelines with scaling and regularization
            temp_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('ridge', Ridge(alpha=0.1))
            ])

            perm_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('ridge', Ridge(alpha=0.1))
            ])

            # Train models
            temp_pipeline.fit(X, y_temp)
            perm_pipeline.fit(X, y_perm)

            # Validate models using cross-validation
            temp_scores = cross_val_score(temp_pipeline, X, y_temp, cv=5, scoring='neg_mean_squared_error')
            perm_scores = cross_val_score(perm_pipeline, X, y_perm, cv=5, scoring='neg_mean_squared_error')

            # Calculate model confidence based on cross-validation scores
            temp_rmse = np.sqrt(-temp_scores.mean())
            perm_rmse = np.sqrt(-perm_scores.mean())

            # Confidence based on prediction accuracy (lower RMSE = higher confidence)
            self.model_confidence = max(0.0, 1.0 - (temp_rmse + perm_rmse) / 2.0)

            # Store trained models
            self.temp_impact_model = temp_pipeline
            self.perm_impact_model = perm_pipeline

            self.logger.info(
                f"Model training completed - Temp RMSE: {temp_rmse:.4f}, "
                f"Perm RMSE: {perm_rmse:.4f}, Confidence: {self.model_confidence:.3f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False

    def estimate_impact(self, order_size: Decimal, symbol: str,
                       market_state: Dict) -> MarketImpactEstimate:
        """
        Estimate market impact for a given order

        Args:
            order_size: Size of the order
            symbol: Trading symbol
            market_state: Current market conditions

        Returns:
            MarketImpactEstimate: Comprehensive impact estimate
        """
        # Check if recalibration is needed
        if self._needs_recalibration():
            self.logger.info("Attempting to recalibrate impact models")
            recent_executions = self._get_recent_executions(symbol)
            if recent_executions:
                self.calibrate_from_trades(recent_executions)

        # Prepare features for prediction
        features = self._prepare_prediction_features(order_size, market_state)

        # Get impact estimates
        temp_impact, perm_impact = self._predict_impact(features)

        # Calculate impact breakdown
        breakdown = self._calculate_impact_breakdown(
            temp_impact, perm_impact, market_state
        )

        # Determine order side for impact direction
        side = OrderSide.BUY  # Default, should be passed as parameter

        return MarketImpactEstimate(
            symbol=symbol,
            order_size=order_size,
            side=side,
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            total_impact=temp_impact + perm_impact,
            spread_component=breakdown['spread_component'],
            size_component=breakdown['size_component'],
            volatility_component=breakdown['volatility_component'],
            timing_component=breakdown['timing_component'],
            permanent_drift=breakdown['permanent_drift'],
            model_confidence=self.model_confidence,
            calibration_time=self.last_calibration or datetime.utcnow(),
            features_used=['relative_size', 'volatility', 'spread', 'execution_speed', 'time_factor', 'market_regime']
        )

    def _needs_recalibration(self) -> bool:
        """Check if model needs recalibration"""
        if self.last_calibration is None:
            return True

        age = (datetime.utcnow() - self.last_calibration).total_seconds()
        return age > self.calibration_interval

    def _prepare_prediction_features(self, order_size: Decimal,
                                   market_state: Dict) -> np.ndarray:
        """Prepare features for impact prediction"""
        try:
            # Extract market state variables
            daily_volume = market_state.get('daily_volume', 1000000)
            volatility = market_state.get('volatility', 0.01)
            spread_bps = market_state.get('spread_bps', 5.0)
            execution_speed = market_state.get('execution_speed', 1.0)
            current_hour = datetime.utcnow().hour
            market_regime = market_state.get('market_regime', 0.5)

            # Calculate features
            relative_size = float(order_size) / max(daily_volume, 1e-10)
            time_factor = np.sin(2 * np.pi * current_hour / 24)

            features = np.array([[
                relative_size,
                volatility,
                spread_bps / 100.0,
                execution_speed,
                time_factor,
                market_regime
            ]])

            return features

        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            # Return default features
            return np.array([[0.001, 0.01, 0.05, 1.0, 0.0, 0.5]])

    def _predict_impact(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict impact using trained models or fallback"""
        try:
            # Use trained models if available and confident
            if (self.temp_impact_model is not None and
                self.perm_impact_model is not None and
                self.model_confidence > 0.3):

                temp = float(self.temp_impact_model.predict(features)[0])
                perm = float(self.perm_impact_model.predict(features)[0])

                # Ensure reasonable bounds
                temp = max(0.0, min(temp, 0.1))  # 0-10%
                perm = max(0.0, min(perm, 0.05))  # 0-5%

                return temp, perm

            else:
                # Fallback to square-root model
                return self._square_root_fallback(features)

        except Exception as e:
            self.logger.error(f"Impact prediction failed: {e}")
            return self._square_root_fallback(features)

    def _square_root_fallback(self, features: np.ndarray) -> Tuple[float, float]:
        """Fallback square-root impact model"""
        relative_size = features[0, 0]  # First feature is relative size
        volatility = features[0, 1]    # Second feature is volatility

        # Square-root impact model
        temp = self.temp_impact_coef * np.sqrt(relative_size) * (1 + volatility)
        perm = self.perm_impact_coef * relative_size * (1 + volatility)

        return float(temp), float(perm)

    def _calculate_impact_breakdown(self, temp_impact: float,
                                  perm_impact: float,
                                  market_state: Dict) -> Dict[str, float]:
        """Break down impact into components"""
        spread_bps = market_state.get('spread_bps', 5.0)

        return {
            'spread_component': spread_bps / 20000,  # Half spread in decimal
            'size_component': temp_impact * 0.6,
            'volatility_component': temp_impact * 0.3,
            'timing_component': temp_impact * 0.1,
            'permanent_drift': perm_impact
        }

    def _get_recent_executions(self, symbol: str) -> List[Dict]:
        """Get recent execution history for calibration"""
        # TODO: Implement database query to get recent executions
        # This should integrate with the execution history database
        return []

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        return {
            'last_calibration': self.last_calibration,
            'model_confidence': self.model_confidence,
            'calibration_count': len(self.calibration_history),
            'has_trained_models': self.temp_impact_model is not None,
            'calibration_history': self.calibration_history[-5:],  # Last 5 calibrations
            'default_coefficients': {
                'temp_impact_coef': self.temp_impact_coef,
                'perm_impact_coef': self.perm_impact_coef
            }
        }

    def reset_models(self) -> None:
        """Reset models to use default coefficients"""
        self.temp_impact_model = None
        self.perm_impact_model = None
        self.model_confidence = 0.0
        self.last_calibration = None
        self.calibration_history.clear()
        self.logger.info("Market impact models reset to defaults")