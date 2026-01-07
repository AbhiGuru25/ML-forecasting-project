"""
Preprocessing Module
Handles timestamp conversion, timezone standardization, and daily aggregation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Preprocesses time series data for daily aggregation."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor with timeseries dataframe.
        
        Args:
            df: DataFrame with columns: metric, count, start, end
        """
        self.df = df.copy()
    
    def convert_timestamps(self) -> pd.DataFrame:
        """
        Convert timestamp strings to datetime objects.
        
        Returns:
            DataFrame with datetime columns
        """
        logger.info("Converting timestamps to datetime objects...")
        
        # Convert start and end to datetime
        self.df['start'] = pd.to_datetime(self.df['start'])
        self.df['end'] = pd.to_datetime(self.df['end'])
        
        logger.info(f"✓ Timestamps converted")
        logger.info(f"  Date range: {self.df['start'].min()} to {self.df['end'].max()}")
        
        return self.df
    
    def standardize_timezones(self, target_tz: str = 'UTC') -> pd.DataFrame:
        """
        Standardize timezones across all timestamps.
        
        Args:
            target_tz: Target timezone (default: UTC)
        
        Returns:
            DataFrame with standardized timezones
        """
        logger.info(f"Standardizing timezones to {target_tz}...")
        
        # Convert to UTC if timezone-aware, otherwise localize
        for col in ['start', 'end']:
            if self.df[col].dt.tz is None:
                # Assume UTC if no timezone info
                self.df[col] = self.df[col].dt.tz_localize('UTC')
            else:
                # Convert to UTC
                self.df[col] = self.df[col].dt.tz_convert('UTC')
        
        logger.info("✓ Timezones standardized to UTC")
        
        return self.df
    
    def aggregate_daily_steps(self) -> pd.DataFrame:
        """
        Aggregate step counts into daily totals.
        
        Returns:
            DataFrame with daily step counts (Date, Daily_Step_Count)
        """
        logger.info("Aggregating step counts to daily totals...")
        
        # Extract date from start timestamp
        self.df['date'] = self.df['start'].dt.date
        
        # Sum steps by date
        daily_steps = self.df.groupby('date')['count'].sum().reset_index()
        daily_steps.columns = ['Date', 'Daily_Step_Count']
        
        # Convert Date back to datetime for consistency
        daily_steps['Date'] = pd.to_datetime(daily_steps['Date'])
        
        logger.info(f"✓ Aggregated to {len(daily_steps)} daily records")
        logger.info(f"  Date range: {daily_steps['Date'].min()} to {daily_steps['Date'].max()}")
        logger.info(f"  Average daily steps: {daily_steps['Daily_Step_Count'].mean():.0f}")
        logger.info(f"  Min daily steps: {daily_steps['Daily_Step_Count'].min():.0f}")
        logger.info(f"  Max daily steps: {daily_steps['Daily_Step_Count'].max():.0f}")
        
        return daily_steps
    
    def create_continuous_timeline(self, daily_steps: pd.DataFrame) -> pd.DataFrame:
        """
        Create a continuous daily timeline with no gaps.
        
        Args:
            daily_steps: DataFrame with Date and Daily_Step_Count
        
        Returns:
            DataFrame with continuous daily timeline (gaps filled with 0 or interpolated)
        """
        logger.info("Creating continuous daily timeline...")
        
        # Get date range
        min_date = daily_steps['Date'].min()
        max_date = daily_steps['Date'].max()
        
        # Create complete date range
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        continuous_df = pd.DataFrame({'Date': date_range})
        
        # Merge with actual data
        continuous_df = continuous_df.merge(daily_steps, on='Date', how='left')
        
        # Count missing days
        missing_days = continuous_df['Daily_Step_Count'].isna().sum()
        
        if missing_days > 0:
            logger.warning(f"  Found {missing_days} missing days in timeline")
            logger.info("  Filling missing values with 0 (assuming no activity)")
            continuous_df['Daily_Step_Count'].fillna(0, inplace=True)
        else:
            logger.info("✓ No gaps found in timeline")
        
        logger.info(f"✓ Continuous timeline created: {len(continuous_df)} days")
        
        return continuous_df
    
    def preprocess(self) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.
        
        Returns:
            DataFrame with continuous daily step counts
        """
        logger.info("\n" + "="*50)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*50)
        
        # Step 1: Convert timestamps
        self.convert_timestamps()
        
        # Step 2: Standardize timezones
        self.standardize_timezones()
        
        # Step 3: Aggregate to daily
        daily_steps = self.aggregate_daily_steps()
        
        # Step 4: Create continuous timeline
        continuous_timeline = self.create_continuous_timeline(daily_steps)
        
        logger.info("="*50)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*50 + "\n")
        
        return continuous_timeline


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import DataLoader
    
    loader = DataLoader()
    timeseries_df, _ = loader.load_all()
    
    preprocessor = TimeSeriesPreprocessor(timeseries_df)
    daily_data = preprocessor.preprocess()
    
    print("\nDaily Step Count Summary:")
    print(daily_data.describe())
    print("\nFirst 10 days:")
    print(daily_data.head(10))
    print("\nLast 10 days:")
    print(daily_data.tail(10))
