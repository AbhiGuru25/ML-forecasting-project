"""
Multivariate ML Model
Implements Explainable Boosting Machine (EBM) for step count forecasting.
"""

import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultivariateForecaster:
    """Multivariate forecasting using Explainable Boosting Machine."""
    
    def __init__(self):
        """Initialize multivariate forecaster."""
        self.model = None
        self.feature_names = []
        self.metrics = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for modeling.
        
        Args:
            df: DataFrame with all engineered features
        
        Returns:
            Tuple of (feature_df, feature_names)
        """
        logger.info("Preparing features for multivariate model...")
        
        # Exclude target and date columns
        exclude_cols = ['Date', 'Daily_Step_Count']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any columns with all NaN or constant values
        valid_features = []
        for col in feature_cols:
            if df[col].notna().sum() > 0 and df[col].nunique() > 1:
                valid_features.append(col)
        
        logger.info(f"  Total features: {len(valid_features)}")
        logger.info(f"  Feature categories:")
        
        # Categorize features
        temporal = [f for f in valid_features if any(x in f for x in ['day_', 'week_', 'month', 'weekend'])]
        lag = [f for f in valid_features if 'steps_t_minus' in f]
        rolling = [f for f in valid_features if 'rolling' in f]
        clinical = [f for f in valid_features if any(x in f for x in ['therapy', 'side_effect', 'diagnosis', 'event'])]
        demo = [f for f in valid_features if any(x in f for x in ['gender', 'age', 'disease'])]
        
        logger.info(f"    - Temporal: {len(temporal)}")
        logger.info(f"    - Lag features: {len(lag)}")
        logger.info(f"    - Rolling features: {len(rolling)}")
        logger.info(f"    - Clinical: {len(clinical)}")
        logger.info(f"    - Demographics: {len(demo)}")
        
        self.feature_names = valid_features
        
        return df[valid_features], valid_features
    
    def prepare_data(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple:
        """
        Prepare train/test split.
        
        Args:
            df: DataFrame with all features
            train_ratio: Ratio for train split
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing train/test split...")
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        y = df['Daily_Step_Count']
        
        # Remove rows with NaN in features (from lag features)
        valid_idx = X.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"  Valid samples after removing NaN: {len(X)}")
        
        # Time-based split (not random, to preserve temporal order)
        split_idx = int(len(X) * train_ratio)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"  Train size: {len(X_train)}")
        logger.info(f"  Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train EBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Training Explainable Boosting Machine...")
        
        # Initialize EBM
        self.model = ExplainableBoostingRegressor(
            max_bins=256,
            max_interaction_bins=32,
            interactions=10,
            outer_bags=8,
            inner_bags=0,
            learning_rate=0.01,
            min_samples_leaf=2,
            max_leaves=3,
            random_state=42
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        logger.info("✓ EBM model trained successfully")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary with metrics
        """
        logger.info("Evaluating multivariate model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Ensure non-negative
        y_pred = np.maximum(y_pred, 0)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        self.metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Test_Size': len(y_test)
        }
        
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        
        return self.metrics
    
    def forecast_future(self, df: pd.DataFrame, periods: int = 365) -> pd.DataFrame:
        """
        Generate future forecast using recursive prediction.
        
        Args:
            df: Full feature dataframe
            periods: Number of days to forecast
        
        Returns:
            Forecast dataframe
        """
        logger.info(f"Generating {periods}-day forecast...")
        
        # Get the last known data point
        last_date = df['Date'].max()
        last_features = df.iloc[-1:].copy()
        
        # Prepare features
        X_features, _ = self.prepare_features(df)
        last_X = X_features.iloc[-1:].copy()
        
        # Initialize forecast list
        forecast_dates = []
        forecast_steps = []
        
        # Recursive forecasting
        for i in range(periods):
            # Predict next day
            pred = self.model.predict(last_X)[0]
            pred = max(0, pred)  # Ensure non-negative
            
            # Store prediction
            next_date = last_date + pd.Timedelta(days=i+1)
            forecast_dates.append(next_date)
            forecast_steps.append(pred)
            
            # Update features for next prediction
            # This is simplified - in practice, you'd update all time-dependent features
            # For now, we'll use the last known feature values
            # In a real scenario, you'd update lags, rolling averages, etc.
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted_Steps': forecast_steps,
            'Trend_Component': forecast_steps,  # Simplified
            'Exogenous_Impact': 0  # Placeholder
        })
        
        logger.info("✓ Forecast generated")
        logger.info(f"  Average predicted steps: {forecast_df['Predicted_Steps'].mean():.0f}")
        
        return forecast_df
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from EBM.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        logger.info(f"Extracting top {top_n} feature importances...")
        
        # Get global explanation
        ebm_global = self.model.explain_global()
        
        # Extract feature importance
        importance_df = pd.DataFrame({
            'Feature': ebm_global.data()['names'],
            'Importance': ebm_global.data()['scores']
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        logger.info("✓ Feature importance extracted")
        
        return importance_df.head(top_n)


def run_multivariate_model(features_df: pd.DataFrame) -> Tuple[MultivariateForecaster, pd.DataFrame]:
    """
    Complete multivariate model pipeline.
    
    Args:
        features_df: DataFrame with all engineered features
    
    Returns:
        Tuple of (trained model, forecast dataframe)
    """
    logger.info("\n" + "="*50)
    logger.info("MULTIVARIATE MODEL (EBM)")
    logger.info("="*50)
    
    # Initialize model
    forecaster = MultivariateForecaster()
    
    # Prepare data
    X_train, X_test, y_train, y_test = forecaster.prepare_data(features_df)
    
    # Train
    forecaster.train(X_train, y_train)
    
    # Evaluate
    metrics = forecaster.evaluate(X_test, y_test)
    
    # Get feature importance
    importance = forecaster.get_feature_importance()
    logger.info("\nTop 10 Most Important Features:")
    for idx, row in importance.head(10).iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Forecast future
    forecast_df = forecaster.forecast_future(features_df, periods=365)
    
    logger.info("="*50)
    logger.info("MULTIVARIATE MODEL COMPLETE")
    logger.info("="*50 + "\n")
    
    return forecaster, forecast_df


if __name__ == "__main__":
    # Test multivariate model
    import sys
    sys.path.append('.')
    
    from src.data_loader import DataLoader
    from src.preprocessing import TimeSeriesPreprocessor
    from src.feature_engineering import FeatureEngineer
    
    loader = DataLoader()
    timeseries_df, categorical_data = loader.load_all()
    
    preprocessor = TimeSeriesPreprocessor(timeseries_df)
    daily_data = preprocessor.preprocess()
    
    engineer = FeatureEngineer(daily_data, categorical_data)
    features_df = engineer.engineer_all_features()
    
    forecaster, forecast = run_multivariate_model(features_df)
    
    print("\nForecast Summary:")
    print(forecast.head(10))
    print(f"\nMetrics: {forecaster.metrics}")
