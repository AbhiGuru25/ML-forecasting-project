"""
Cloud Utilities
Provides cloud integration functions (AWS S3 upload simulation).
"""

import pandas as pd
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_forecast_to_s3(
    dataframe: pd.DataFrame,
    bucket_name: str,
    file_name: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: str = 'us-east-1',
    execute: bool = False
) -> None:
    """
    Upload forecast dataframe to AWS S3.
    
    IMPORTANT: This function is for demonstration purposes.
    Actual execution is disabled by default (execute=False).
    
    Args:
        dataframe: Forecast dataframe to upload
        bucket_name: S3 bucket name
        file_name: File name in S3
        aws_access_key_id: AWS access key (should use IAM roles instead)
        aws_secret_access_key: AWS secret key (should use IAM roles instead)
        region_name: AWS region
        execute: Set to True to actually execute upload (default: False)
    
    Security Best Practices:
        1. NEVER hard-code AWS credentials in notebooks or code
        2. Use IAM roles when running on AWS infrastructure (EC2, Lambda, etc.)
        3. Use environment variables for local development
        4. Use AWS Secrets Manager for sensitive credentials
        5. Implement least-privilege access policies
    
    Example Usage:
        # Using environment variables (RECOMMENDED)
        import os
        upload_forecast_to_s3(
            dataframe=forecast_df,
            bucket_name='my-forecast-bucket',
            file_name='forecasts/2024-01-01.csv',
            execute=False  # Keep False for demo
        )
        
        # In production with IAM roles (BEST PRACTICE)
        # No credentials needed - boto3 automatically uses IAM role
        upload_forecast_to_s3(
            dataframe=forecast_df,
            bucket_name='my-forecast-bucket',
            file_name='forecasts/2024-01-01.csv',
            execute=True
        )
    """
    
    logger.info("="*60)
    logger.info("S3 UPLOAD FUNCTION (SIMULATION)")
    logger.info("="*60)
    
    if not execute:
        logger.info("⚠️  Upload execution is DISABLED (execute=False)")
        logger.info("   This is a simulation for demonstration purposes")
        logger.info("")
        logger.info("Upload Configuration:")
        logger.info(f"  Bucket: {bucket_name}")
        logger.info(f"  File: {file_name}")
        logger.info(f"  Region: {region_name}")
        logger.info(f"  Dataframe shape: {dataframe.shape}")
        logger.info(f"  Dataframe size: {dataframe.memory_usage(deep=True).sum() / 1024:.2f} KB")
        logger.info("")
        logger.info("Security Best Practices:")
        logger.info("  ✓ Never hard-code AWS credentials")
        logger.info("  ✓ Use IAM roles for AWS infrastructure")
        logger.info("  ✓ Use environment variables for local dev")
        logger.info("  ✓ Use AWS Secrets Manager for sensitive data")
        logger.info("  ✓ Implement least-privilege IAM policies")
        logger.info("")
        logger.info("Example IAM Policy for S3 Upload:")
        logger.info("""
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:PutObject",
                        "s3:PutObjectAcl"
                    ],
                    "Resource": "arn:aws:s3:::BUCKET_NAME/forecasts/*"
                }
            ]
        }
        """)
        logger.info("="*60)
        return
    
    # Actual upload code (only executes if execute=True)
    try:
        import boto3
        import io
        
        logger.info("Executing S3 upload...")
        
        # Initialize S3 client
        # If running on AWS infrastructure, boto3 automatically uses IAM role
        # Otherwise, it uses credentials from environment variables or ~/.aws/credentials
        s3_client = boto3.client(
            's3',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,  # None if using IAM role
            aws_secret_access_key=aws_secret_access_key  # None if using IAM role
        )
        
        # Convert dataframe to CSV
        csv_buffer = io.StringIO()
        dataframe.to_csv(csv_buffer, index=False)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        logger.info(f"✓ Successfully uploaded to s3://{bucket_name}/{file_name}")
        
    except ImportError:
        logger.error("boto3 not installed. Install with: pip install boto3")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error("Check your AWS credentials and permissions")


def get_credentials_from_secrets_manager(secret_name: str, region_name: str = 'us-east-1') -> dict:
    """
    Retrieve credentials from AWS Secrets Manager.
    
    This is a BEST PRACTICE for managing sensitive credentials.
    
    Args:
        secret_name: Name of the secret in Secrets Manager
        region_name: AWS region
    
    Returns:
        Dictionary with credentials
    
    Example:
        credentials = get_credentials_from_secrets_manager('my-app/credentials')
        api_key = credentials['api_key']
    """
    try:
        import boto3
        import json
        
        client = boto3.client('secretsmanager', region_name=region_name)
        response = client.get_secret_value(SecretId=secret_name)
        
        if 'SecretString' in response:
            return json.loads(response['SecretString'])
        else:
            # Binary secret
            import base64
            return json.loads(base64.b64decode(response['SecretBinary']))
            
    except Exception as e:
        logger.error(f"Failed to retrieve secret: {str(e)}")
        return {}


if __name__ == "__main__":
    # Demo
    demo_df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=365),
        'Predicted_Steps': [5000] * 365,
        'Trend_Component': [4800] * 365,
        'Exogenous_Impact': [200] * 365
    })
    
    upload_forecast_to_s3(
        dataframe=demo_df,
        bucket_name='my-forecast-bucket',
        file_name='forecasts/demo.csv',
        execute=False  # Simulation only
    )
