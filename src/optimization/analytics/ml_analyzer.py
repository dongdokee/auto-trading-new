"""
Machine learning model training and evaluation.

This module provides automated ML pipeline for predictive analytics including:
- Model training and evaluation
- Feature importance analysis
- Prediction generation
- Model performance comparison
"""

from typing import Dict, List, Tuple

from .core import BaseAnalyzer, AnalyticsResult, AnalyticsError

try:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    np = None
    train_test_split = None
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None
    StandardScaler = None
    LinearRegression = None
    RandomForestRegressor = None


class MachineLearningAnalyzer(BaseAnalyzer):
    """
    Machine learning model training and evaluation.

    Provides automated ML pipeline for predictive analytics.
    """

    def __init__(self):
        """Initialize ML analyzer."""
        super().__init__("MachineLearningAnalyzer")
        self.models = {}
        self.scalers = {}

    def analyze(self, features: List[List[float]], targets: List[float], **kwargs) -> AnalyticsResult:
        """Train and evaluate ML models."""
        if LinearRegression is None:
            raise AnalyticsError("scikit-learn is required for machine learning analysis")

        if len(features) != len(targets) or len(features) < 3:
            raise AnalyticsError("Insufficient or mismatched data for ML analysis")

        result = AnalyticsResult(
            analysis_type="machine_learning_analysis",
            data_points=len(features)
        )

        # Train multiple models
        models_performance = {}

        # Linear Regression
        lr_performance = self._train_and_evaluate_model(
            LinearRegression(), features, targets, "linear_regression"
        )
        models_performance['linear_regression'] = lr_performance

        # Random Forest (if available)
        if RandomForestRegressor is not None:
            rf_performance = self._train_and_evaluate_model(
                RandomForestRegressor(n_estimators=50, random_state=42),
                features, targets, "random_forest"
            )
            models_performance['random_forest'] = rf_performance

        result.results['models'] = models_performance

        # Select best model
        best_model = max(models_performance.items(), key=lambda x: x[1]['r2_score'])
        result.results['best_model'] = {
            'name': best_model[0],
            'performance': best_model[1]
        }
        result.metrics.update(best_model[1])

        # Generate insights
        result.insights = self._generate_ml_insights(models_performance, best_model)
        result.confidence = best_model[1]['r2_score']

        return result

    def predict(self, model_name: str, features: List[List[float]]) -> AnalyticsResult:
        """Make predictions using trained model."""
        if model_name not in self.models:
            raise AnalyticsError(f"Model {model_name} not found")

        model = self.models[model_name]
        scaler = self.scalers.get(model_name)

        result = AnalyticsResult(
            analysis_type=f"prediction_{model_name}",
            data_points=len(features)
        )

        # Scale features if scaler available
        if scaler:
            features = scaler.transform(features)

        # Make predictions
        predictions = model.predict(features).tolist()
        result.results['predictions'] = predictions
        result.metrics['prediction_count'] = len(predictions)

        result.add_insight(f"Generated {len(predictions)} predictions using {model_name} model")
        result.confidence = 0.85

        return result

    def feature_importance(self, model_name: str) -> AnalyticsResult:
        """Analyze feature importance for the specified model."""
        if model_name not in self.models:
            raise AnalyticsError(f"Model {model_name} not found")

        model = self.models[model_name]
        result = AnalyticsResult(
            analysis_type=f"feature_importance_{model_name}"
        )

        # Get feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_.tolist()
            result.results['feature_importances'] = importances
            result.metrics['top_feature_importance'] = max(importances)

            # Generate insights
            top_feature_idx = importances.index(max(importances))
            result.add_insight(f"Most important feature: Feature {top_feature_idx} (importance: {max(importances):.3f})")

        elif hasattr(model, 'coef_'):
            coefficients = model.coef_.tolist()
            result.results['coefficients'] = coefficients
            result.metrics['max_coefficient'] = max(abs(c) for c in coefficients)

            # Generate insights
            max_coef_idx = coefficients.index(max(coefficients, key=abs))
            result.add_insight(f"Highest coefficient: Feature {max_coef_idx} (coefficient: {coefficients[max_coef_idx]:.3f})")

        result.confidence = 0.90

        return result

    def _train_and_evaluate_model(self, model, features: List[List[float]], targets: List[float], model_name: str) -> Dict[str, float]:
        """Train and evaluate a single model."""
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler

        return {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'rmse': np.sqrt(mse)
        }

    def _generate_ml_insights(self, models_performance: Dict, best_model: Tuple) -> List[str]:
        """Generate insights from ML analysis."""
        insights = []

        best_name, best_perf = best_model
        insights.append(f"Best performing model: {best_name} (R² = {best_perf['r2_score']:.3f})")

        if best_perf['r2_score'] > 0.8:
            insights.append("Excellent model performance achieved")
        elif best_perf['r2_score'] > 0.6:
            insights.append("Good model performance achieved")
        elif best_perf['r2_score'] > 0.3:
            insights.append("Moderate model performance - consider feature engineering")
        else:
            insights.append("Poor model performance - data may not be suitable for prediction")

        # Compare models if multiple available
        if len(models_performance) > 1:
            performances = [perf['r2_score'] for perf in models_performance.values()]
            if max(performances) - min(performances) > 0.1:
                insights.append("Significant performance differences between models detected")

        return insights