# 🚀 QUICK START GUIDE

## Installation & First Run

### Step 1: Navigate to Project
```powershell
cd d:\Project\stock_long_term_forecast
```

### Step 2: Install Dependencies (First Time Only)
```powershell
pip install -r requirements.txt
```

### Step 3: Test Installation (Optional)
```powershell
python test_installation.py
```

### Step 4: Run the System
```powershell
python main.py
```

## What Happens When You Run

1. **Interactive Menu Appears**
   ```
   ================================================================================
   🚀 STOCK PRICE FORECASTING SYSTEM
   ================================================================================
   
   📈 Select a stock to predict:
   
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
   
     0. Exit
   ================================================================================
   
   Enter your choice (1-10 or 0 to exit):
   ```

2. **Select a Stock** - Enter a number (1-10)

3. **System Processes** (Takes 2-5 minutes)
   - Downloads 15+ years of stock data
   - Calculates 60+ technical indicators
   - Trains 5 ensemble ML models
   - Makes predictions for 5 targets
   - Generates professional visualizations

4. **Results Generated**
   ```
   outputs/
   └── RELIANCE/              # (or your selected stock)
       ├── models/           # Trained models (.pkl files)
       │   ├── Next_Month_Close_stacking_model.pkl
       │   ├── Next_Month_Close_stacking_scaler.pkl
       │   └── ... (10 files total)
       │
       ├── plots/            # PNG visualizations
       │   ├── Next_Month_Close_predictions.png
       │   ├── Next_Month_Close_error_analysis.png
       │   ├── Next_Quarter_Close_predictions.png
       │   ├── complete_dashboard.png
       │   └── ... (15+ plots)
       │
       ├── data/             # Processed datasets
       │   ├── raw_stock_data.csv
       │   └── processed_data.csv
       │
       └── results/          # JSON output
           └── forecast_results.json  # 👈 MAIN OUTPUT
   ```

5. **View Results**
   - Open `forecast_results.json` for structured data
   - Open PNG files in `plots/` folder to see visualizations

## Understanding the Output

### JSON Result Structure

```json
{
  "timestamp": "2025-11-04 10:30:45",
  "stock": {
    "ticker": "RELIANCE.NS",
    "name": "Reliance Industries"
  },
  "data_info": {
    "start_date": "2010-01-01",
    "end_date": "2025-11-04",
    "total_samples": 3908,
    "training_samples": 3126,
    "test_samples": 782
  },
  "predictions": {
    "Next_Month_Close": {
      "prediction": 1289.75,        // Predicted price
      "current_value": 1234.50,     // Current price
      "change": 55.25,              // Absolute change
      "change_pct": 4.48,           // Percentage change
      "description": "Price 1 month ahead",
      "model_type": "stacking_ensemble",
      "metrics": {
        "r2": 0.8523,              // Higher is better (max 1.0)
        "mae": 23.45,              // Lower is better
        "rmse": 31.78,             // Lower is better
        "mape": 1.89,              // Lower is better
        "directional_accuracy": 91.2,  // Higher is better (%)
        "correlation": 0.9234,
        "within_5pct": 78.5,       // % predictions within ±5%
        "within_10pct": 94.3       // % predictions within ±10%
      }
    },
    "Next_Quarter_Close": { ... },
    "Next_Year_Close": { ... },
    "Cumulative_Return": { ... },
    "Volatility_Next_Month": { ... }
  },
  "output_folders": { ... }
}
```

### Key Metrics Explained

- **R² Score**: 0 to 1, measures how well predictions fit actual data
  - 0.9+ = Excellent
  - 0.8-0.9 = Very Good
  - 0.7-0.8 = Good
  - <0.7 = Needs improvement

- **MAE (Mean Absolute Error)**: Average prediction error in same units as price
  - Lower is better
  - E.g., MAE of 25 means average error of ₹25

- **MAPE (Mean Absolute Percentage Error)**: Average error as percentage
  - <5% = Excellent
  - 5-10% = Very Good
  - 10-20% = Good
  - >20% = Needs improvement

- **Directional Accuracy**: How often it predicts up/down correctly
  - >90% = Excellent for trading signals
  - 80-90% = Very Good
  - 70-80% = Good
  - <70% = Not reliable for trading

## Example Session

```powershell
PS D:\Project\stock_long_term_forecast> python main.py

================================================================================
🚀 STOCK PRICE FORECASTING SYSTEM
================================================================================

📈 Select a stock to predict:

  1. Reliance Industries (RELIANCE.NS)
  ...

Enter your choice (1-10 or 0 to exit): 1

================================================================================
🎯 Selected: Reliance Industries (RELIANCE.NS)
================================================================================

📥 STEP 1: Loading data...
✅ Downloaded RELIANCE.NS: 3908 rows

🔧 STEP 2: Building features...
✅ Created 67 features

🤖 STEP 3: Training models and making predictions...
─────────────────────────────────────────────────────────────
🎯 Processing target: Next_Month_Close
   Description: Price 1 month ahead
─────────────────────────────────────────────────────────────
✅ Model training complete!

============================================================
📊 Next_Month_Close Performance
============================================================
R² Score:               0.8523
Mean Absolute Error:    23.4500
Root Mean Squared Error: 31.7800
MAPE:                   1.89%
Directional Accuracy:   91.20%
Correlation:            0.9234
Predictions within ±5%: 78.5%
Predictions within ±10%: 94.3%
============================================================

... (continues for all 5 targets)

📊 STEP 4: Creating visualizations...
✅ All visualizations created!

💾 STEP 5: Saving results...
✅ JSON saved: outputs\RELIANCE\results\forecast_results.json

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
    Results (JSON):  outputs\RELIANCE\results\forecast_results.json
    Models:          outputs\RELIANCE\models\
    Plots:           outputs\RELIANCE\plots\
    Data:            outputs\RELIANCE\data\

================================================================================

🔄 Would you like to predict another stock? (y/n):
```

## Tips & Tricks

### 1. Running for Multiple Stocks
Just say "y" when asked if you want to predict another stock!

### 2. Viewing Plots
Navigate to `outputs/{STOCK_NAME}/plots/` and open PNG files

### 3. Comparing Stocks
Run for multiple stocks, then compare their JSON files

### 4. Reusing Models
Trained models are saved in `models/` folder - you can load them later without retraining

### 5. Customizing
Edit `config/settings.py` to:
- Change date ranges
- Add more stocks
- Modify model parameters
- Adjust prediction targets

## Troubleshooting

### "No module named 'xyz'"
```powershell
pip install -r requirements.txt
```

### "Failed to download data"
- Check internet connection
- Ticker might be delisted - try another stock

### "Out of memory"
- Reduce date range in `config/settings.py`
- Close other applications

### Slow performance
- First run is slow (downloads 15+ years data)
- Subsequent runs with same stock are faster (uses cache)

## Next Steps

1. ✅ Run `test_installation.py` to verify setup
2. ✅ Run `main.py` and select stock #1 (Reliance)
3. ✅ Wait 2-5 minutes for completion
4. ✅ Open `outputs/RELIANCE/results/forecast_results.json`
5. ✅ View plots in `outputs/RELIANCE/plots/`
6. ✅ Try other stocks!

## Need Help?

- Check `outputs/logs/` for detailed logs
- Review `README.md` for full documentation
- Ensure Python 3.8+ is installed

---

**🎉 Happy Forecasting!**
