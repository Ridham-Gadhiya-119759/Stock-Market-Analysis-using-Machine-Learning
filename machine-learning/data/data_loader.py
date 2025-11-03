"""
Data loading module - Downloads stock data from Yahoo Finance
"""
import pandas as pd
import yfinance as yf
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    """Download and clean stock data from Yahoo Finance"""
    
    def __init__(self, ticker, start_date, end_date, index_ticker='^NSEI'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.index_ticker = index_ticker
        
    def download_data(self, ticker, name="Stock"):
        """
        Download data from Yahoo Finance with robust error handling
        
        Args:
            ticker: Stock ticker symbol
            name: Name for logging
            
        Returns:
            pandas.DataFrame with OHLCV data
        """
        logger.info(f"📥 Downloading {name} data: {ticker}")
        
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=False,
                threads=False
            )
            
            # Handle tuple return
            if isinstance(data, tuple):
                data = data[0]
            
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(1)
            
            # Validate data
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError(f"No valid data for {ticker}")
            
            # Clean column names
            data = data.rename(columns={c: str(c).lower().replace(' ', '_') for c in data.columns})
            data.index = pd.to_datetime(data.index)
            
            # Fix adj_close column
            if 'adj_close' not in data.columns:
                if 'adjclose' in data.columns:
                    data.rename(columns={'adjclose': 'adj_close'}, inplace=True)
                elif 'close' in data.columns:
                    data['adj_close'] = data['close']
            
            logger.info(f"✅ Downloaded {ticker}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error downloading {ticker}: {e}")
            return pd.DataFrame()
    
    def load_stock_data(self):
        """Load main stock data"""
        return self.download_data(self.ticker, "Stock")
    
    def load_index_data(self):
        """Load market index data"""
        return self.download_data(self.index_ticker, "Index")
    
    def load_all(self):
        """
        Load both stock and index data
        
        Returns:
            dict with 'stock' and 'index' DataFrames
        """
        stock_df = self.load_stock_data()
        index_df = self.load_index_data()
        
        return {
            'stock': stock_df,
            'index': index_df
        }
