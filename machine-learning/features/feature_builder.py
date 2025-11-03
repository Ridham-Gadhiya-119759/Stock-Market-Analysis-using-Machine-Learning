"""
Feature builder - Combines all features and creates target variables
"""
import numpy as np
import pandas as pd
from features.technical_indicators import TechnicalIndicators
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureBuilder:
    """Build complete feature set for modeling"""
    
    def __init__(self, target_configs):
        self.target_configs = target_configs
        
    def create_targets(self, df):
        """
        Create target variables for prediction
        
        Args:
            df: Input DataFrame with stock data
            
        Returns:
            DataFrame with added target columns
        """
        logger.info("🎯 Creating target variables...")
        
        for target_name, config in self.target_configs.items():
            if config['type'] == 'regression':
                shift = config['shift']
                
                if 'Close' in target_name:
                    # Price targets
                    df[target_name] = df['close'].shift(shift)
                    
                elif 'Return' in target_name:
                    # Return targets
                    df[target_name] = df['close'].pct_change(periods=abs(shift)).shift(shift)
                    
                elif 'Volatility' in target_name:
                    # Volatility targets
                    window = abs(shift)
                    df[target_name] = df['ret'].rolling(window=window).std().shift(shift) * np.sqrt(252)
        
        logger.info(f"✅ Created {len(self.target_configs)} target variables")
        
        return df
    
    def create_additional_features(self, df, index_df=None):
        """
        Create additional derived features
        
        Args:
            df: Stock DataFrame
            index_df: Market index DataFrame (optional)
            
        Returns:
            DataFrame with additional features
        """
        logger.info("🔧 Creating additional features...")
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'ret_lag_{lag}'] = df['ret'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'ret_mean_{window}'] = df['ret'].rolling(window=window).mean()
            df[f'ret_std_{window}'] = df['ret'].rolling(window=window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume indicators
        df['volume_change'] = df['volume'].pct_change()
        df['price_volume'] = df['close'] * df['volume']
        
        # Trend indicators
        df['trend_5'] = np.where(df['close'] > df['sma_5'], 1, -1)
        df['trend_20'] = np.where(df['close'] > df['sma_20'], 1, -1)
        
        # Add market index features if available
        if index_df is not None and not index_df.empty:
            index_df = index_df.rename(columns={col: f'index_{col}' for col in index_df.columns})
            df = df.join(index_df[['index_close', 'index_volume']], how='left')
            
            if 'index_close' in df.columns:
                df['index_ret'] = df['index_close'].pct_change()
                df['beta'] = df['ret'].rolling(window=63).cov(df['index_ret']) / df['index_ret'].rolling(window=63).var()
        
        logger.info(f"✅ Added additional features. Total columns: {len(df.columns)}")
        
        return df
    
    def build_features(self, stock_df, index_df=None):
        """
        Complete feature building pipeline
        
        Args:
            stock_df: Stock price DataFrame
            index_df: Market index DataFrame (optional)
            
        Returns:
            DataFrame with all features and targets
        """
        logger.info("🏗️ Starting feature building pipeline...")
        
        # Add technical indicators
        df = TechnicalIndicators.add_all_indicators(stock_df.copy())
        
        # Add additional features
        df = self.create_additional_features(df, index_df)
        
        # Create targets
        df = self.create_targets(df)
        
        # Remove NaN rows
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)
        
        logger.info(f"✅ Feature building complete! Removed {removed} rows with NaNs")
        logger.info(f"📊 Final dataset shape: {df.shape}")
        
        return df
    
    def get_feature_names(self, df):
        """
        Get list of feature column names (excluding targets and basic OHLCV)
        
        Args:
            df: DataFrame with all features
            
        Returns:
            list of feature column names
        """
        # Exclude basic columns and targets
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close'] + list(self.target_configs.keys())
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"📋 Identified {len(feature_cols)} feature columns")
        
        return feature_cols
