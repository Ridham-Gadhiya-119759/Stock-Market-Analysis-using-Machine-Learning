"""
File operations utilities
"""
import os
import json
import joblib
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

def save_json(data, filepath):
    """Save dictionary to JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"✅ JSON saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save JSON: {e}")
        return False

def load_json(filepath):
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"✅ JSON loaded: {filepath}")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load JSON: {e}")
        return None

def save_model(model, filepath):
    """Save sklearn model using joblib"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"✅ Model saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return False

def load_model(filepath):
    """Load sklearn model"""
    try:
        model = joblib.load(filepath)
        logger.info(f"✅ Model loaded: {filepath}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

def save_dataframe(df, filepath, format='csv'):
    """Save pandas DataFrame"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if format == 'csv':
            df.to_csv(filepath)
        elif format == 'parquet':
            df.to_parquet(filepath)
        elif format == 'excel':
            df.to_excel(filepath)
        logger.info(f"✅ DataFrame saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save DataFrame: {e}")
        return False

def load_dataframe(filepath, format='csv'):
    """Load pandas DataFrame"""
    try:
        if format == 'csv':
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format == 'parquet':
            df = pd.read_parquet(filepath)
        elif format == 'excel':
            df = pd.read_excel(filepath, index_col=0, parse_dates=True)
        logger.info(f"✅ DataFrame loaded: {filepath}")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load DataFrame: {e}")
        return None
