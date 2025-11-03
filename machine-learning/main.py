"""
Main pipeline for stock forecasting system
Run this file to execute the complete forecasting workflow
"""
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import *
from utils.logger import setup_logger
from utils.file_ops import save_json, save_dataframe
from data import DataLoader, DataPreprocessor
from features import FeatureBuilder
from model import ModelTrainer, ModelPredictor, ModelUtils
from evaluation import MetricsCalculator, Visualizer

logger = setup_logger('main')

def display_stock_menu():
    """Display stock selection menu"""
    print("\n" + "="*70)
    print("🚀 STOCK PRICE FORECASTING SYSTEM")
    print("="*70)
    print("\n📈 Select a stock to predict:\n")
    
    for key, stock_info in TOP_STOCKS.items():
        print(f"  {key}. {stock_info['name']} ({stock_info['ticker']})")
    
    print(f"\n  0. Exit")
    print("="*70)

def get_user_choice():
    """Get user's stock selection"""
    while True:
        choice = input("\nEnter your choice (1-10 or 0 to exit): ").strip()
        
        if choice == '0':
            print("\n👋 Thank you for using the forecasting system!")
            sys.exit(0)
        
        if choice in TOP_STOCKS:
            return TOP_STOCKS[choice]
        
        print("❌ Invalid choice. Please select a number between 1-10.")

def create_output_folders(ticker):
    """Create output folders for the selected stock"""
    stock_dir = os.path.join(OUTPUT_DIR, ticker.replace('.NS', ''))
    folders = {
        'main': stock_dir,
        'models': os.path.join(stock_dir, 'models'),
        'plots': os.path.join(stock_dir, 'plots'),
        'data': os.path.join(stock_dir, 'data'),
        'results': os.path.join(stock_dir, 'results')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    logger.info(f"📁 Created output folders for {ticker}")
    
    return folders

def main():
    """Main forecasting pipeline"""
    
    # Display menu and get selection
    display_stock_menu()
    stock_info = get_user_choice()
    
    ticker = stock_info['ticker']
    stock_name = stock_info['name']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 Selected: {stock_name} ({ticker})")
    logger.info(f"{'='*70}\n")
    
    # Create output folders
    folders = create_output_folders(ticker)
    
    # Update global paths
    global MODEL_DIR, PLOTS_DIR, DATA_DIR
    MODEL_DIR = folders['models']
    PLOTS_DIR = folders['plots']
    DATA_DIR = folders['data']
    
    try:
        # =================================================================
        # STEP 1: DATA LOADING
        # =================================================================
        logger.info("\n📥 STEP 1: Loading data...")
        
        data_loader = DataLoader(ticker, START_DATE, END_DATE)
        data = data_loader.load_all()
        
        stock_df = data['stock']
        index_df = data['index']
        
        if stock_df.empty:
            logger.error(f"❌ Failed to load data for {ticker}")
            return
        
        logger.info(f"✅ Loaded {len(stock_df)} days of data")
        
        # Save raw data
        save_dataframe(stock_df, os.path.join(DATA_DIR, 'raw_stock_data.csv'))
        
        # =================================================================
        # STEP 2: FEATURE ENGINEERING
        # =================================================================
        logger.info("\n🔧 STEP 2: Building features...")
        
        feature_builder = FeatureBuilder(TARGET_CONFIGS)
        df = feature_builder.build_features(stock_df, index_df)
        
        feature_cols = feature_builder.get_feature_names(df)
        
        logger.info(f"✅ Created {len(feature_cols)} features")
        
        # Save processed data
        save_dataframe(df, os.path.join(DATA_DIR, 'processed_data.csv'))
        
        # =================================================================
        # STEP 3: MODEL TRAINING & PREDICTION
        # =================================================================
        logger.info("\n🤖 STEP 3: Training models and making predictions...")
        
        all_results = {}
        final_predictions = {}
        
        for target_name, target_config in TARGET_CONFIGS.items():
            logger.info(f"\n{'─'*70}")
            logger.info(f"🎯 Processing target: {target_name}")
            logger.info(f"   Description: {target_config['description']}")
            logger.info(f"{'─'*70}")
            
            # Prepare data for this target
            df_target = df[[target_name] + feature_cols].dropna()
            
            preprocessor = DataPreprocessor(test_size=TEST_SIZE, random_state=RANDOM_SEED)
            prepared_data = preprocessor.prepare_data(df_target, target_name, feature_cols)
            
            X_train_scaled = prepared_data['X_train_scaled']
            X_test_scaled = prepared_data['X_test_scaled']
            y_train = prepared_data['y_train']
            y_test = prepared_data['y_test']
            test_dates = prepared_data['test_dates']
            scaler = prepared_data['scaler']
            
            # Train model (using ensemble for best performance)
            trainer = ModelTrainer(model_type='stacking')
            model = trainer.train(X_train_scaled, y_train.values)
            
            # Make predictions
            y_pred = trainer.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_all_metrics(y_test.values, y_pred)
            MetricsCalculator.print_metrics(metrics, f"{target_name} Performance")
            
            # Save model and scaler
            model_paths = ModelUtils.save_model_and_scaler(
                model, scaler, target_name, 'stacking'
            )
            
            # Store results
            all_results[target_name] = {
                'y_true': y_test.values,
                'y_pred': y_pred,
                'dates': test_dates,
                'metrics': metrics
            }
            
            # Predict next value
            latest_features = prepared_data['X_test'].iloc[-1:].values
            latest_scaled = scaler.transform(latest_features)
            next_prediction = float(model.predict(latest_scaled)[0])
            
            final_predictions[target_name] = {
                'prediction': next_prediction,
                'current_value': float(y_test.iloc[-1]),
                'change': next_prediction - float(y_test.iloc[-1]),
                'change_pct': ((next_prediction - float(y_test.iloc[-1])) / float(y_test.iloc[-1])) * 100,
                'description': target_config['description'],
                'model_type': 'stacking_ensemble',
                'metrics': metrics
            }
        
        # =================================================================
        # STEP 4: VISUALIZATION
        # =================================================================
        logger.info("\n📊 STEP 4: Creating visualizations...")
        
        visualizer = Visualizer(stock_name)
        
        for target_name, result in all_results.items():
            # Prediction plots
            visualizer.plot_predictions(
                result['y_true'], result['y_pred'], result['dates'],
                target_name, save_dir=PLOTS_DIR
            )
            
            # Error analysis
            visualizer.plot_error_analysis(
                result['y_true'], result['y_pred'],
                target_name, save_dir=PLOTS_DIR
            )
        
        # Create comprehensive dashboard
        visualizer.create_summary_dashboard(all_results, save_dir=PLOTS_DIR)
        
        logger.info("✅ All visualizations created!")
        
        # =================================================================
        # STEP 5: SAVE RESULTS AS JSON
        # =================================================================
        logger.info("\n💾 STEP 5: Saving results...")
        
        # Prepare JSON output
        output_json = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock': {
                'ticker': ticker,
                'name': stock_name
            },
            'data_info': {
                'start_date': START_DATE,
                'end_date': END_DATE,
                'total_samples': len(df),
                'training_samples': len(prepared_data['X_train']),
                'test_samples': len(prepared_data['X_test'])
            },
            'predictions': final_predictions,
            'output_folders': {
                'models': folders['models'],
                'plots': folders['plots'],
                'data': folders['data'],
                'results': folders['results']
            }
        }
        
        # Save JSON
        json_path = os.path.join(folders['results'], 'forecast_results.json')
        save_json(output_json, json_path)
        
        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        logger.info("\n" + "="*70)
        logger.info("🎉 FORECASTING COMPLETE!")
        logger.info("="*70)
        logger.info(f"\n📈 {stock_name} ({ticker})")
        logger.info(f"\n📊 PREDICTIONS:")
        
        for target_name, pred_info in final_predictions.items():
            logger.info(f"\n  {target_name}:")
            logger.info(f"    Current Value:  {pred_info['current_value']:.2f}")
            logger.info(f"    Predicted:      {pred_info['prediction']:.2f}")
            logger.info(f"    Change:         {pred_info['change']:+.2f} ({pred_info['change_pct']:+.2f}%)")
            logger.info(f"    R² Score:       {pred_info['metrics']['r2']:.4f}")
            logger.info(f"    Dir. Accuracy:  {pred_info['metrics']['directional_accuracy']:.1f}%")
        
        logger.info(f"\n📁 OUTPUT LOCATIONS:")
        logger.info(f"    Results (JSON):  {json_path}")
        logger.info(f"    Models:          {folders['models']}")
        logger.info(f"    Plots:           {folders['plots']}")
        logger.info(f"    Data:            {folders['data']}")
        
        logger.info("\n" + "="*70)
        
        # Ask if user wants to predict another stock
        print("\n")
        another = input("🔄 Would you like to predict another stock? (y/n): ").strip().lower()
        if another == 'y':
            main()  # Recursive call
        else:
            logger.info("\n👋 Thank you for using the forecasting system!")
        
    except Exception as e:
        logger.error(f"\n❌ Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        # Ask if user wants to try again
        retry = input("\n🔄 Would you like to try again? (y/n): ").strip().lower()
        if retry == 'y':
            main()

if __name__ == "__main__":
    main()
