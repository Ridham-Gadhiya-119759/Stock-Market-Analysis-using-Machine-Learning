"""
Technical indicators calculation
"""
import numpy as np
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TechnicalIndicators:
    """Calculate various technical indicators for stock analysis"""
    
    @staticmethod
    def add_returns(df):
        """Add return columns"""
        df['ret'] = df['close'].pct_change()
        df['log_ret'] = np.log(df['close']).diff()
        return df
    
    @staticmethod
    def add_moving_averages(df, windows=[5, 10, 20, 50, 200]):
        """Add Simple and Exponential Moving Averages"""
        for w in windows:
            df[f'sma_{w}'] = df['close'].rolling(window=w).mean()
            df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()
        return df
    
    @staticmethod
    def add_rsi(df, period=14):
        """Add Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """Add MACD indicator"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    @staticmethod
    def add_bollinger_bands(df, window=20, num_std=2):
        """Add Bollinger Bands"""
        sma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        df['bb_upper'] = sma + (std * num_std)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * num_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        return df
    
    @staticmethod
    def add_volatility(df, windows=[5, 10, 21, 63]):
        """Add rolling volatility"""
        for w in windows:
            df[f'volatility_{w}'] = df['ret'].rolling(window=w).std() * np.sqrt(252)
        return df
    
    @staticmethod
    def add_atr(df, period=14):
        """Add Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_stochastic(df, k_period=14, d_period=3):
        """Add Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        return df
    
    @staticmethod
    def add_obv(df):
        """Add On-Balance Volume"""
        obv = np.where(df['close'] > df['close'].shift(), df['volume'],
                      np.where(df['close'] < df['close'].shift(), -df['volume'], 0))
        df['obv'] = pd.Series(obv, index=df.index).cumsum()
        return df
    
    @staticmethod
    def add_momentum(df, periods=[5, 10, 20]):
        """Add momentum indicators"""
        for p in periods:
            df[f'momentum_{p}'] = df['close'].pct_change(periods=p)
        return df
    
    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators"""
        logger.info("📊 Calculating technical indicators...")
        
        df = TechnicalIndicators.add_returns(df)
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_volatility(df)
        df = TechnicalIndicators.add_atr(df)
        df = TechnicalIndicators.add_stochastic(df)
        df = TechnicalIndicators.add_obv(df)
        df = TechnicalIndicators.add_momentum(df)
        
        logger.info(f"✅ Added technical indicators. Total columns: {len(df.columns)}")
        
        return df
