"""
Data preprocessing module - Handles NaNs, scaling, and splitting
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """Handle data cleaning, scaling, and train-test splitting"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def handle_missing_values(self, df):
        """
        Handle missing values in DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("🧹 Handling missing values...")
        
        initial_shape = df.shape
        
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Drop remaining NaNs
        df = df.dropna()
        
        final_shape = df.shape
        removed = initial_shape[0] - final_shape[0]
        
        logger.info(f"✅ Removed {removed} rows with NaNs. Final shape: {final_shape}")
        
        return df
    
    def split_data(self, df, target_col, feature_cols):
        """
        Split data into train and test sets
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature column names
            
        Returns:
            dict with X_train, X_test, y_train, y_test
        """
        logger.info(f"✂️ Splitting data: {int((1-self.test_size)*100)}% train, {int(self.test_size*100)}% test")
        
        # Time-series split (no shuffling)
        split_idx = int(len(df) * (1 - self.test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        logger.info(f"✅ Train: {len(X_train)} samples | Test: {len(X_test)} samples")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_dates': train_df.index,
            'test_dates': test_df.index
        }
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            dict with scaled X_train and X_test
        """
        logger.info("⚖️ Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("✅ Features scaled successfully")
        
        return {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': self.scaler
        }
    
    def prepare_data(self, df, target_col, feature_cols):
        """
        Complete data preparation pipeline
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature column names
            
        Returns:
            dict with all prepared data
        """
        logger.info("🔧 Starting data preparation pipeline...")
        
        # Clean data
        df_clean = self.handle_missing_values(df)
        
        # Split data
        split_data = self.split_data(df_clean, target_col, feature_cols)
        
        # Scale features
        scaled_data = self.scale_features(split_data['X_train'], split_data['X_test'])
        
        # Combine all results
        result = {**split_data, **scaled_data}
        
        logger.info("✅ Data preparation complete!")
        
        return result
