"""
Configuration settings for stock forecasting system
"""
import os
from datetime import datetime, timedelta

# ============================================================================
# TOP 10 INDIAN STOCKS FOR PREDICTION
# ============================================================================
TOP_STOCKS = {
    '1': {'ticker': 'RELIANCE.NS', 'name': 'Reliance Industries'},
    '2': {'ticker': 'TCS.NS', 'name': 'Tata Consultancy Services'},
    '3': {'ticker': 'INFY.NS', 'name': 'Infosys'},
    '4': {'ticker': 'HDFCBANK.NS', 'name': 'HDFC Bank'},
    '5': {'ticker': 'ICICIBANK.NS', 'name': 'ICICI Bank'},
    '6': {'ticker': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever'},
    '7': {'ticker': 'BAJFINANCE.NS', 'name': 'Bajaj Finance'},
    '8': {'ticker': 'BHARTIARTL.NS', 'name': 'Bharti Airtel'},
    '9': {'ticker': 'ITC.NS', 'name': 'ITC Limited'},
    '10': {'ticker': 'KOTAKBANK.NS', 'name': 'Kotak Mahindra Bank'}
}

# ============================================================================
# DATA PARAMETERS
# ============================================================================
START_DATE = '2010-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
INDEX_TICKER = '^NSEI'  # NIFTY 50 Index

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 80-20 train-test split
INITIAL_TRAIN_SIZE = 500
RETRAIN_FREQUENCY = 20

# Prediction targets
TARGET_CONFIGS = {
    'Next_Month_Close': {
        'type': 'regression',
        'shift': -21,
        'description': 'Price 1 month ahead'
    },
    'Next_Quarter_Close': {
        'type': 'regression',
        'shift': -63,
        'description': 'Price 3 months ahead'
    },
    'Next_Year_Close': {
        'type': 'regression',
        'shift': -252,
        'description': 'Price 1 year ahead'
    },
    'Cumulative_Return': {
        'type': 'regression',
        'shift': -21,
        'description': '1-month cumulative return'
    },
    'Volatility_Next_Month': {
        'type': 'regression',
        'shift': -21,
        'description': '1-month forward volatility'
    }
}

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================
OPTIMIZED_PARAMS = {
    'RandomForest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_SEED
    },
    'GradientBoosting': {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'subsample': 0.8,
        'random_state': RANDOM_SEED
    },
    'ExtraTrees': {
        'n_estimators': 200,
        'max_depth': 25,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_SEED
    }
}

# ============================================================================
# TRADING STRATEGY PARAMETERS
# ============================================================================
CONFIDENCE_THRESHOLD = 0.60
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10
MAX_POSITION_SIZE = 0.30
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.06

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')

# Create directories
for directory in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300
COLOR_PALETTE = 'viridis'
