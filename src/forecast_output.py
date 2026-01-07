"""
Forecast Output Module
Generates final forecast output in required format.
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_forecast_output(
    forecast_df: pd.DataFrame,
    model_name: str = 'multivariate'
) -> pd.DataFrame:
    """
    Create final forecast output in required format.
    
    Required schema:
    - Date
    - Predicted_Steps
    - Trend_Component
    - Exogenous_Impact
    
    Args:
        forecast_df: Forecast dataframe from model
        model_name: Name of the model used
    
    Returns:
        Formatted forecast dataframe
    """
    logger.info(f"Creating forecast output for {model_name} model...")
    
    # Ensure required columns exist
    required_cols = ['Date', 'Predicted_Steps', 'Trend_Component', 'Exogenous_Impact']
    
    output_df = forecast_df.copy()
    
    # Add missing columns if needed
    if 'Exogenous_Impact' not in output_df.columns:
        # Calculate as difference between prediction and trend
        if 'Trend_Component' in output_df.columns:
            output_df['Exogenous_Impact'] = (
                output_df['Predicted_Steps'] - output_df['Trend_Component']
            )
        else:
            output_df['Exogenous_Impact'] = 0
    
    if 'Trend_Component' not in output_df.columns:
        # Use predicted steps as trend if not available
        output_df['Trend_Component'] = output_df['Predicted_Steps']
        output_df['Exogenous_Impact'] = 0
    
    # Select and order columns
    output_df = output_df[required_cols]
    
    # Round to integers
    output_df['Predicted_Steps'] = output_df['Predicted_Steps'].round(0).astype(int)
    output_df['Trend_Component'] = output_df['Trend_Component'].round(0).astype(int)
    output_df['Exogenous_Impact'] = output_df['Exogenous_Impact'].round(0).astype(int)
    
    logger.info(f"✓ Forecast output created: {len(output_df)} rows")
    logger.info(f"  Date range: {output_df['Date'].min()} to {output_df['Date'].max()}")
    logger.info(f"  Avg predicted steps: {output_df['Predicted_Steps'].mean():.0f}")
    
    return output_df


def save_forecast_output(output_df: pd.DataFrame, file_path: str) -> None:
    """
    Save forecast output to CSV.
    
    Args:
        output_df: Forecast dataframe
        file_path: Path to save CSV
    """
    logger.info(f"Saving forecast output to {file_path}...")
    
    output_df.to_csv(file_path, index=False)
    
    logger.info(f"✓ Forecast saved successfully")


def display_forecast_sample(output_df: pd.DataFrame, n: int = 10) -> None:
    """
    Display sample of forecast output.
    
    Args:
        output_df: Forecast dataframe
        n: Number of rows to display
    """
    print("\n" + "="*70)
    print(f"FORECAST OUTPUT SAMPLE (First {n} days)")
    print("="*70)
    print(output_df.head(n).to_string(index=False))
    print("\n" + "="*70)
    print(f"FORECAST OUTPUT SAMPLE (Last {n} days)")
    print("="*70)
    print(output_df.tail(n).to_string(index=False))
    print("\n")


if __name__ == "__main__":
    # Demo
    demo_forecast = pd.DataFrame({
        'Date': pd.date_range('2025-12-12', periods=365),
        'Predicted_Steps': [4500 + i*10 for i in range(365)],
        'Trend_Component': [4200 + i*10 for i in range(365)]
    })
    
    output = create_forecast_output(demo_forecast)
    display_forecast_sample(output)
