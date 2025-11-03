"""
Helper functions
"""
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_returns(prices):
    """Calculate percentage returns"""
    return prices.pct_change()

def calculate_log_returns(prices):
    """Calculate log returns"""
    return np.log(prices).diff()

def calculate_volatility(returns, window=21):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_sharpe_ratio(returns, risk_free_rate=0.06):
    """Calculate Sharpe ratio"""
    excess_returns = returns.mean() - risk_free_rate/252
    return excess_returns / returns.std() * np.sqrt(252)

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def format_number(num, decimals=2):
    """Format number with thousands separator"""
    return f"{num:,.{decimals}f}"

def format_percentage(num, decimals=2):
    """Format number as percentage"""
    return f"{num:.{decimals}f}%"

def get_date_range_info(start_date, end_date):
    """Get information about date range"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = (end - start).days
    years = days / 365.25
    return {
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'days': days,
        'years': round(years, 2)
    }

def validate_dataframe(df, required_columns):
    """Validate DataFrame has required columns"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
