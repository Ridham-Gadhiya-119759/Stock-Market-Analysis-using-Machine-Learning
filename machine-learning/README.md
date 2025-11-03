# 📊 Stock Long-Term Forecasting System

A comprehensive Python-based stock forecasting system that predicts multiple targets (price, returns, volatility) using ensemble machine learning models.

## 🚀 Features

- **Interactive Stock Selection**: Choose from top 10 Indian stocks
- **Multiple Predictions**: Forecasts 5 different targets:
  - Next Month Close Price
  - Next Quarter Close Price
  - Next Year Close Price
  - Cumulative Returns
  - Volatility (Next Month)
- **Advanced ML Models**: Ensemble methods (Stacking, Voting) with optimized hyperparameters
- **Comprehensive Metrics**: R², MAE, RMSE, MAPE, Directional Accuracy
- **Professional Visualizations**: Auto-generated plots saved in separate folders
- **JSON Output**: All results exported in structured JSON format
- **Modular Architecture**: Clean, maintainable code structure

## 📁 Project Structure

```
stock_long_term_forecast/
│
├── main.py                      # Run this file!
│
├── config/
│   └── settings.py              # Configuration parameters
│
├── data/
│   ├── data_loader.py           # Yahoo Finance data fetcher
│   └── data_preprocessor.py     # Data cleaning and scaling
│
├── features/
│   ├── technical_indicators.py  # Technical indicators (SMA, RSI, MACD, etc.)
│   └── feature_builder.py       # Feature engineering pipeline
│
├── model/
│   ├── model_trainer.py         # Model training (RF, GB, ET, Ensembles)
│   ├── model_predictor.py       # Inference and prediction
│   └── model_utils.py           # Save/load, hyperparameter tuning
│
├── evaluation/
│   ├── metrics.py               # Performance metrics
│   └── visualizer.py            # Plotting and visualization
│
├── utils/
│   ├── logger.py                # Logging setup
│   ├── file_ops.py              # File operations
│   └── helpers.py               # Helper functions
│
├── outputs/                     # Generated outputs (auto-created)
│   ├── {STOCK_NAME}/
│   │   ├── models/             # Trained models (.pkl)
│   │   ├── plots/              # PNG visualizations
│   │   ├── data/               # Processed datasets
│   │   └── results/            # JSON results
│   └── logs/                   # System logs
│
└── requirements.txt            # Dependencies
```

## 🔧 Installation

### 1. Clone or Download

```bash
cd d:\Project\stock_long_term_forecast
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the System

```bash
python main.py
```

## 💻 Usage

### Simple Interactive Mode

1. Run `python main.py`
2. Select a stock from the menu (1-10)
3. Wait for the system to:
   - Download data
   - Engineer features
   - Train models
   - Make predictions
   - Generate visualizations
4. Check the `outputs/{STOCK}/` folder for results

### Example Output

```
================================================================================
🎉 FORECASTING COMPLETE!
================================================================================

📈 Reliance Industries (RELIANCE.NS)

📊 PREDICTIONS:

  Next_Month_Close:
    Current Value:  1234.50
    Predicted:      1289.75
    Change:         +55.25 (+4.48%)
    R² Score:       0.8523
    Dir. Accuracy:  91.2%

  Next_Quarter_Close:
    Current Value:  1234.50
    Predicted:      1356.80
    Change:         +122.30 (+9.91%)
    R² Score:       0.7891
    Dir. Accuracy:  87.6%

📁 OUTPUT LOCATIONS:
    Results (JSON):  outputs/RELIANCE/results/forecast_results.json
    Models:          outputs/RELIANCE/models/
    Plots:           outputs/RELIANCE/plots/
    Data:            outputs/RELIANCE/data/
```

## 📊 Output Files

### JSON Results (`forecast_results.json`)

```json
{
  "timestamp": "2025-11-04 10:30:45",
  "stock": {
    "ticker": "RELIANCE.NS",
    "name": "Reliance Industries"
  },
  "predictions": {
    "Next_Month_Close": {
      "prediction": 1289.75,
      "current_value": 1234.50,
      "change": 55.25,
      "change_pct": 4.48,
      "metrics": {
        "r2": 0.8523,
        "mae": 23.45,
        "rmse": 31.78,
        "directional_accuracy": 91.2
      }
    }
  }
}
```

### Generated Plots

- `{target}_predictions.png` - Actual vs Predicted time series and scatter plots
- `{target}_error_analysis.png` - Error distribution and Q-Q plots
- `{target}_feature_importance.png` - Top features by importance
- `complete_dashboard.png` - Comprehensive dashboard with all targets

## 🎯 Supported Stocks

1. Reliance Industries (RELIANCE.NS)
2. Tata Consultancy Services (TCS.NS)
3. Infosys (INFY.NS)
4. HDFC Bank (HDFCBANK.NS)
5. ICICI Bank (ICICIBANK.NS)
6. Hindustan Unilever (HINDUNILVR.NS)
7. Bajaj Finance (BAJFINANCE.NS)
8. Bharti Airtel (BHARTIARTL.NS)
9. ITC Limited (ITC.NS)
10. Kotak Mahindra Bank (KOTAKBANK.NS)

## 🔬 Technical Details

### Models Used

- **Random Forest Regressor**: Ensemble of decision trees
- **Gradient Boosting Regressor**: Sequential boosting
- **Extra Trees Regressor**: Extremely randomized trees
- **Stacking Ensemble**: Combination of above with Ridge meta-learner

### Features Engineered (60+)

- Price & Volume: OHLCV, ratios, changes
- Moving Averages: SMA, EMA (5, 10, 20, 50, 200)
- Momentum: RSI, MACD, Stochastic
- Volatility: ATR, Bollinger Bands
- Trends: Direction indicators
- Market: Index correlation, Beta

### Performance Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Directional Accuracy (%)
- Correlation
- Within ±5% and ±10% accuracy

## 🛠️ Configuration

Edit `config/settings.py` to customize:

- Date ranges
- Model parameters
- Trading thresholds
- Output directories
- Stock list

## 📝 Logging

- Console logs: Real-time progress
- File logs: `outputs/logs/forecast_YYYYMMDD_HHMMSS.log`

## 🤝 Contributing

This is a production-ready forecasting system. Feel free to:

- Add more stocks to `TOP_STOCKS` in `config/settings.py`
- Add custom technical indicators in `features/technical_indicators.py`
- Experiment with different models in `model/model_trainer.py`
- Create custom visualizations in `evaluation/visualizer.py`

## 📄 License

MIT License - Feel free to use for personal or commercial projects

## 👨‍💻 Author

**Ridham Gadhiya**
- Software Engineer
- Stock Market Analysis using Machine Learning

## 🙏 Acknowledgments

- Yahoo Finance for data
- Scikit-learn for ML models
- Matplotlib/Seaborn for visualizations

---

⭐ **Star this project if you find it useful!**
