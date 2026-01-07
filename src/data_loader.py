"""
Data Loader Module
Loads and validates JSON datasets for time series forecasting project.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial validation of JSON data files."""
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing the JSON data files
        """
        self.data_dir = Path(data_dir)
        self.timeseries_file = self.data_dir / "timeseries-data.json"
        self.categorical_file = self.data_dir / "categorical-data.json"
    
    def load_timeseries_data(self) -> pd.DataFrame:
        """
        Load time series step count data from JSON.
        
        Returns:
            DataFrame with columns: metric, count, start, end
        """
        logger.info(f"Loading timeseries data from {self.timeseries_file}")
        
        with open(self.timeseries_file, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"Loaded {len(df)} timeseries records")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Date range: {df['start'].min()} to {df['end'].max()}")
        
        # Validate metric field
        if 'metric' in df.columns:
            unique_metrics = df['metric'].unique()
            logger.info(f"Unique metrics: {unique_metrics}")
            if len(unique_metrics) == 1 and unique_metrics[0] == 'STEPS':
                logger.info("✓ Metric validation passed - all records are STEPS")
        
        return df
    
    def load_categorical_data(self) -> Dict:
        """
        Load categorical clinical data from JSON.
        
        Returns:
            Dictionary containing demographics and clinical events
        """
        logger.info(f"Loading categorical data from {self.categorical_file}")
        
        with open(self.categorical_file, 'r') as f:
            data = json.load(f)
        
        logger.info("Categorical data structure:")
        logger.info(f"  - Top-level keys: {list(data.keys())}")
        
        # Log demographics if present
        if 'gender' in data:
            logger.info(f"  - Gender: {data.get('gender')}")
        if 'birthYear' in data:
            logger.info(f"  - Birth Year: {data.get('birthYear')}")
        if 'disease' in data:
            logger.info(f"  - Disease: {data.get('disease')}")
        
        # Log clinical event counts
        for key in ['therapies', 'sideEffects', 'diagnoses', 'events']:
            if key in data and isinstance(data[key], list):
                logger.info(f"  - {key}: {len(data[key])} records")
        
        return data
    
    def load_all(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load both datasets.
        
        Returns:
            Tuple of (timeseries_df, categorical_dict)
        """
        timeseries_df = self.load_timeseries_data()
        categorical_data = self.load_categorical_data()
        
        return timeseries_df, categorical_data
    
    def validate_files_exist(self) -> bool:
        """
        Check if required data files exist.
        
        Returns:
            True if both files exist, False otherwise
        """
        timeseries_exists = self.timeseries_file.exists()
        categorical_exists = self.categorical_file.exists()
        
        if not timeseries_exists:
            logger.error(f"Timeseries file not found: {self.timeseries_file}")
        if not categorical_exists:
            logger.error(f"Categorical file not found: {self.categorical_file}")
        
        return timeseries_exists and categorical_exists


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    if loader.validate_files_exist():
        timeseries_df, categorical_data = loader.load_all()
        print("\n" + "="*50)
        print("DATA LOADING SUMMARY")
        print("="*50)
        print(f"Timeseries records: {len(timeseries_df)}")
        print(f"Categorical data keys: {list(categorical_data.keys())}")
        print("\nFirst few timeseries records:")
        print(timeseries_df.head())
    else:
        print("ERROR: Required data files not found!")
