"""
Feature Engineering Module
Engineers clinical and time-series features for forecasting model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features from clinical data and time series."""
    
    def __init__(self, daily_timeline: pd.DataFrame, categorical_data: Dict):
        """
        Initialize feature engineer.
        
        Args:
            daily_timeline: DataFrame with Date and Daily_Step_Count
            categorical_data: Dictionary with clinical data
        """
        self.timeline = daily_timeline.copy()
        self.clinical_data = categorical_data
        
    def add_demographics(self) -> pd.DataFrame:
        """
        Add demographic features (gender, age, disease).
        
        Returns:
            DataFrame with demographic features
        """
        logger.info("Adding demographic features...")
        
        # Gender (one-hot encode)
        gender = self.clinical_data.get('gender', 'UNKNOWN')
        self.timeline[f'gender_{gender}'] = 1
        
        # Age (calculate from birth year)
        birth_year = self.clinical_data.get('birthYear')
        if birth_year:
            current_year = datetime.now().year
            age = current_year - birth_year
            self.timeline['age'] = age
            logger.info(f"  Age: {age}")
        
        # Disease (encode)
        disease = self.clinical_data.get('disease', 'UNKNOWN')
        self.timeline[f'disease_{disease}'] = 1
        logger.info(f"  Disease: {disease}")
        
        logger.info("✓ Demographics added")
        return self.timeline
    
    def add_therapy_features(self) -> pd.DataFrame:
        """
        Add therapy-related features.
        
        Returns:
            DataFrame with therapy features
        """
        logger.info("Adding therapy features...")
        
        therapies = self.clinical_data.get('therapies', [])
        
        if not therapies:
            logger.warning("  No therapy data found")
            self.timeline['active_therapy_count'] = 0
            return self.timeline
        
        # Initialize therapy columns
        self.timeline['active_therapy_count'] = 0
        therapy_durations = []
        
        for idx, therapy in enumerate(therapies):
            therapy_id = therapy.get('therapyId', f'therapy_{idx}')
            start_date = pd.to_datetime(therapy.get('startDate'))
            end_date = therapy.get('endDate')
            
            if end_date:
                end_date = pd.to_datetime(end_date)
            else:
                # If no end date, assume ongoing
                end_date = self.timeline['Date'].max()
            
            # Create binary flag for this therapy
            col_name = f'is_on_therapy_{therapy_id}'
            
            # Ensure dates are timezone-naive for comparison
            if self.timeline['Date'].dt.tz is not None:
                timeline_dates = self.timeline['Date'].dt.tz_localize(None)
            else:
                timeline_dates = self.timeline['Date']
            
            if start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            
            self.timeline[col_name] = (
                (timeline_dates >= start_date) & 
                (timeline_dates <= end_date)
            ).astype(int)
            
            # Count active therapies
            self.timeline['active_therapy_count'] += self.timeline[col_name]
            
            # Track duration
            duration = (end_date - start_date).days
            therapy_durations.append(duration)
        
        # Add average therapy duration as a feature
        if therapy_durations:
            self.timeline['avg_therapy_duration'] = np.mean(therapy_durations)
        
        logger.info(f"  Processed {len(therapies)} therapies")
        logger.info(f"  Max concurrent therapies: {self.timeline['active_therapy_count'].max()}")
        
        return self.timeline
    
    def add_side_effect_features(self) -> pd.DataFrame:
        """
        Add side effect features.
        
        Returns:
            DataFrame with side effect features
        """
        logger.info("Adding side effect features...")
        
        side_effects = self.clinical_data.get('sideEffects', [])
        
        if not side_effects:
            logger.warning("  No side effect data found")
            self.timeline['active_side_effect_count'] = 0
            self.timeline['max_side_effect_intensity'] = 0
            self.timeline['avg_side_effect_intensity'] = 0
            return self.timeline
        
        # Initialize columns
        self.timeline['active_side_effect_count'] = 0
        self.timeline['max_side_effect_intensity'] = 0
        self.timeline['total_intensity'] = 0
        
        for idx, se in enumerate(side_effects):
            start_date = pd.to_datetime(se.get('startDate'))
            end_date = se.get('endDate')
            intensity = se.get('intensity', 1)
            
            if end_date:
                end_date = pd.to_datetime(end_date)
            else:
                end_date = self.timeline['Date'].max()
            
            # Ensure timezone-naive dates for comparison
            timeline_dates = self.timeline['Date']
            if timeline_dates.dt.tz is not None:
                timeline_dates = timeline_dates.dt.tz_localize(None)
            if start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            
            # Create mask for active period
            mask = (timeline_dates >= start_date) & (timeline_dates <= end_date)
            
            # Count active side effects
            self.timeline.loc[mask, 'active_side_effect_count'] += 1
            
            # Track max intensity
            self.timeline.loc[mask, 'max_side_effect_intensity'] = self.timeline.loc[mask, 'max_side_effect_intensity'].apply(
                lambda x: max(x, intensity)
            )
            
            # Track total intensity for averaging
            self.timeline.loc[mask, 'total_intensity'] += intensity
        
        # Calculate average intensity
        self.timeline['avg_side_effect_intensity'] = np.where(
            self.timeline['active_side_effect_count'] > 0,
            self.timeline['total_intensity'] / self.timeline['active_side_effect_count'],
            0
        )
        
        # Drop temporary column
        self.timeline.drop('total_intensity', axis=1, inplace=True)
        
        logger.info(f"  Processed {len(side_effects)} side effects")
        logger.info(f"  Max concurrent side effects: {self.timeline['active_side_effect_count'].max()}")
        logger.info(f"  Max intensity: {self.timeline['max_side_effect_intensity'].max()}")
        
        return self.timeline
    
    def add_diagnosis_features(self) -> pd.DataFrame:
        """
        Add diagnosis-related features.
        
        Returns:
            DataFrame with diagnosis features
        """
        logger.info("Adding diagnosis features...")
        
        diagnoses = self.clinical_data.get('diagnoses', [])
        
        if not diagnoses:
            logger.warning("  No diagnosis data found")
            self.timeline['active_diagnosis_count'] = 0
            return self.timeline
        
        self.timeline['active_diagnosis_count'] = 0
        
        for idx, diag in enumerate(diagnoses):
            diag_id = diag.get('diagnosisOptionsId', f'diagnosis_{idx}')
            start_date = pd.to_datetime(diag.get('startDate'))
            end_date = diag.get('endDate')
            
            if end_date:
                end_date = pd.to_datetime(end_date)
            else:
                end_date = self.timeline['Date'].max()
            
            # Ensure timezone-naive dates
            timeline_dates = self.timeline['Date']
            if timeline_dates.dt.tz is not None:
                timeline_dates = timeline_dates.dt.tz_localize(None)
            if start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            
            # Create binary flag
            col_name = f'diagnosis_active_{diag_id}'
            self.timeline[col_name] = (
                (timeline_dates >= start_date) & 
                (timeline_dates <= end_date)
            ).astype(int)
            
            self.timeline['active_diagnosis_count'] += self.timeline[col_name]
        
        logger.info(f"  Processed {len(diagnoses)} diagnoses")
        
        return self.timeline
    
    def add_event_features(self) -> pd.DataFrame:
        """
        Add clinical event features (e.g., relapses).
        
        Returns:
            DataFrame with event features
        """
        logger.info("Adding event features...")
        
        events = self.clinical_data.get('events', [])
        
        if not events:
            logger.warning("  No event data found")
            self.timeline['days_since_last_event'] = 9999
            return self.timeline
        
        # Initialize
        self.timeline['days_since_last_event'] = 9999
        
        # Convert event dates
        event_dates = []
        for event in events:
            event_date = pd.to_datetime(event.get('startDate'))
            event_dates.append(event_date)
        
        # Ensure timezone-naive dates
        timeline_dates = self.timeline['Date']
        if timeline_dates.dt.tz is not None:
            timeline_dates = timeline_dates.dt.tz_localize(None)
        event_dates = [ed.tz_localize(None) if hasattr(ed, 'tz') and ed.tz is not None else ed for ed in event_dates]
        
        # For each day, calculate days since last event
        for idx, row in self.timeline.iterrows():
            current_date = timeline_dates.iloc[idx]
            
            # Find most recent event before current date
            past_events = [ed for ed in event_dates if ed <= current_date]
            
            if past_events:
                most_recent = max(past_events)
                days_since = (current_date - most_recent).days
                self.timeline.at[idx, 'days_since_last_event'] = days_since
        
        logger.info(f"  Processed {len(events)} events")
        logger.info(f"  Min days since event: {self.timeline['days_since_last_event'].min()}")
        
        return self.timeline
    
    def add_temporal_features(self) -> pd.DataFrame:
        """
        Add time-based features (day of week, week of year, etc.).
        
        Returns:
            DataFrame with temporal features
        """
        logger.info("Adding temporal features...")
        
        self.timeline['day_of_week'] = self.timeline['Date'].dt.dayofweek
        self.timeline['week_of_year'] = self.timeline['Date'].dt.isocalendar().week
        self.timeline['month'] = self.timeline['Date'].dt.month
        self.timeline['is_weekend'] = (self.timeline['day_of_week'] >= 5).astype(int)
        
        logger.info("✓ Temporal features added")
        
        return self.timeline
    
    def add_lag_features(self, lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        Add lag features for step counts.
        
        Args:
            lags: List of lag periods (default: [1, 7, 30])
        
        Returns:
            DataFrame with lag features
        """
        logger.info(f"Adding lag features: {lags}")
        
        for lag in lags:
            self.timeline[f'steps_t_minus_{lag}'] = self.timeline['Daily_Step_Count'].shift(lag)
        
        logger.info("✓ Lag features added")
        
        return self.timeline
    
    def add_rolling_features(self, windows: List[int] = [7, 30]) -> pd.DataFrame:
        """
        Add rolling average features.
        
        Args:
            windows: List of window sizes (default: [7, 30])
        
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Adding rolling features: {windows}")
        
        for window in windows:
            self.timeline[f'rolling_avg_{window}d'] = (
                self.timeline['Daily_Step_Count']
                .rolling(window=window, min_periods=1)
                .mean()
            )
            
            self.timeline[f'rolling_std_{window}d'] = (
                self.timeline['Daily_Step_Count']
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)
            )
        
        logger.info("✓ Rolling features added")
        
        return self.timeline
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("\n" + "="*50)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("="*50)
        
        # Demographics
        self.add_demographics()
        
        # Clinical features
        self.add_therapy_features()
        self.add_side_effect_features()
        self.add_diagnosis_features()
        self.add_event_features()
        
        # Temporal features
        self.add_temporal_features()
        
        # Time series features
        self.add_lag_features()
        self.add_rolling_features()
        
        logger.info("="*50)
        logger.info(f"FEATURE ENGINEERING COMPLETE")
        logger.info(f"Total features: {len(self.timeline.columns)}")
        logger.info("="*50 + "\n")
        
        return self.timeline


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader
    from preprocessing import TimeSeriesPreprocessor
    
    loader = DataLoader()
    timeseries_df, categorical_data = loader.load_all()
    
    preprocessor = TimeSeriesPreprocessor(timeseries_df)
    daily_data = preprocessor.preprocess()
    
    engineer = FeatureEngineer(daily_data, categorical_data)
    features_df = engineer.engineer_all_features()
    
    print("\nFeature Summary:")
    print(f"Total rows: {len(features_df)}")
    print(f"Total columns: {len(features_df.columns)}")
    print("\nColumn names:")
    print(features_df.columns.tolist())
    print("\nFirst 5 rows:")
    print(features_df.head())
