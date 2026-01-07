"""
Baseline Time Series Model
Implements univariate forecasting using Prophet.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineForecaster:
    """Baseline univariate time series forecasting model using Prophet."""
    
    def __init__(self):
        """Initialize baseline forecaster."""
        self.model = None
        self.metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).
        
        Args:
            df: DataFrame with Date and Daily_Step_Count
            train_ratio: Ratio of data to use for training
        
        Returns:
            Tuple of (train_df, test_df) in Prophet format
        """
        logger.info("Preparing data for Prophet...")
        
        # Convert to Prophet format
        prophet_df = df[['Date', 'Daily_Step_Count']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Split train/test
        split_idx = int(len(prophet_df) * train_ratio)
        train_df = prophet_df[:split_idx]
        test_df = prophet_df[split_idx:]
        
        logger.info(f"  Train size: {len(train_df)} days")
        logger.info(f"  Test size: {len(test_df)} days")
        logger.info(f"  Train date range: {train_df['ds'].min()} to {train_df['ds'].max()}")
        logger.info(f"  Test date range: {test_df['ds'].min()} to {test_df['ds'].max()}")
        
        return train_df, test_df
    
    def train(self, train_df: pd.DataFrame) -> None:
        """
        Train Prophet model.
        
        Args:
            train_df: Training data in Prophet format (ds, y)
        """
        logger.info("Training Prophet model...")
        
        # Initialize Prophet with sensible defaults
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # Fit model
        self.model.fit(train_df)
        
        logger.info("✓ Model trained successfully")
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_df: Test data in Prophet format
        
        Returns:
            Dictionary with RMSE and MAE metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Make predictions
        forecast = self.model.predict(test_df[['ds']])
        
        # Calculate metrics
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        self.metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Test_Size': len(test_df)
        }
        
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        
        return self.metrics
    
    def forecast_future(self, periods: int = 365) -> pd.DataFrame:
        """
        Generate future forecast.
        
        Args:
            periods: Number of days to forecast (default: 365)
        
        Returns:
            DataFrame with forecast
        """
        logger.info(f"Generating {periods}-day forecast...")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Extract only future predictions
        future_forecast = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
        future_forecast.columns = ['Date', 'Predicted_Steps', 'Lower_Bound', 'Upper_Bound', 'Trend_Component']
        
        # Ensure non-negative predictions
        future_forecast['Predicted_Steps'] = future_forecast['Predicted_Steps'].clip(lower=0)
        future_forecast['Lower_Bound'] = future_forecast['Lower_Bound'].clip(lower=0)
        
        logger.info("✓ Forecast generated")
        logger.info(f"  Forecast date range: {future_forecast['Date'].min()} to {future_forecast['Date'].max()}")
        logger.info(f"  Average predicted steps: {future_forecast['Predicted_Steps'].mean():.0f}")
        
        return future_forecast
    
    def plot_forecast(self, forecast_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot forecast results.
        
        Args:
            forecast_df: Forecast dataframe
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(14, 6))
        
        plt.plot(forecast_df['Date'], forecast_df['Predicted_Steps'], label='Predicted Steps', linewidth=2)
        plt.fill_between(
            forecast_df['Date'],
            forecast_df['Lower_Bound'],
            forecast_df['Upper_Bound'],
            alpha=0.3,
            label='Confidence Interval'
        )
        
        plt.xlabel('Date')
        plt.ylabel('Daily Step Count')
        plt.title('Baseline Model: 365-Day Step Count Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  Plot saved to {save_path}")
        
        plt.close()


def run_baseline_model(daily_data: pd.DataFrame) -> Tuple[BaselineForecaster, pd.DataFrame]:
    """
    Complete baseline model pipeline.
    
    Args:
        daily_data: DataFrame with Date and Daily_Step_Count
    
    Returns:
        Tuple of (trained model, forecast dataframe)
    """
    logger.info("\n" + "="*50)
    logger.info("BASELINE MODEL (PROPHET)")
    logger.info("="*50)
    
    # Initialize model
    forecaster = BaselineForecaster()
    
    # Prepare data
    train_df, test_df = forecaster.prepare_data(daily_data)
    
    # Train
    forecaster.train(train_df)
    
    # Evaluate
    metrics = forecaster.evaluate(test_df)
    
    # Forecast future
    forecast_df = forecaster.forecast_future(periods=365)
    
    logger.info("="*50)
    logger.info("BASELINE MODEL COMPLETE")
    logger.info("="*50 + "\n")
    
    return forecaster, forecast_df


if __name__ == "__main__":
    # Test baseline model
    import sys
    sys.path.append('.')
    
    from src.data_loader import DataLoader
    from src.preprocessing import TimeSeriesPreprocessor
    
    loader = DataLoader()
    timeseries_df, _ = loader.load_all()
    
    preprocessor = TimeSeriesPreprocessor(timeseries_df)
    daily_data = preprocessor.preprocess()
    
    forecaster, forecast = run_baseline_model(daily_data)
    
    print("\nForecast Summary:")
    print(forecast.head(10))
    print(f"\nMetrics: {forecaster.metrics}")
