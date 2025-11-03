"""
Model prediction module
"""
import numpy as np
import pandas as pd
from utils.logger import setup_logger
from utils.file_ops import load_model

logger = setup_logger(__name__)

class ModelPredictor:
    """Handle model inference and forecasting"""
    
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        
    def load_model(self, model_path, scaler_path=None):
        """
        Load saved model and scaler
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler (optional)
            
        Returns:
            bool indicating success
        """
        self.model = load_model(model_path)
        
        if scaler_path is not None:
            self.scaler = load_model(scaler_path)
        
        return self.model is not None
    
    def predict(self, X, return_confidence=False):
        """
        Make predictions on new data
        
        Args:
            X: Features for prediction
            return_confidence: Whether to return confidence intervals (if supported)
            
        Returns:
            Predictions (and confidence intervals if requested)
        """
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        if return_confidence:
            # For ensemble models, we can estimate confidence from individual predictions
            if hasattr(self.model, 'estimators_'):
                # Get predictions from all estimators
                all_preds = np.array([est.predict(X_scaled) for est in self.model.estimators_])
                confidence_lower = np.percentile(all_preds, 5, axis=0)
                confidence_upper = np.percentile(all_preds, 95, axis=0)
                
                return predictions, (confidence_lower, confidence_upper)
        
        return predictions
    
    def predict_next(self, latest_features):
        """
        Predict next period value
        
        Args:
            latest_features: Most recent feature values
            
        Returns:
            Single prediction value
        """
        if isinstance(latest_features, pd.Series):
            latest_features = latest_features.values.reshape(1, -1)
        elif isinstance(latest_features, pd.DataFrame):
            latest_features = latest_features.values
        
        prediction = self.predict(latest_features)
        
        return float(prediction[0])
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """
        Calculate directional accuracy (up/down prediction correctness)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy percentage
        """
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        accuracy = np.mean(actual_direction == pred_direction) * 100
        
        return accuracy
    
    def generate_signals(self, predictions, current_prices, threshold=0.05):
        """
        Generate trading signals based on predictions
        
        Args:
            predictions: Predicted future prices
            current_prices: Current prices
            threshold: Minimum percentage change to trigger signal
            
        Returns:
            DataFrame with signals (BUY/SELL/HOLD)
        """
        price_change_pct = (predictions - current_prices) / current_prices
        
        signals = []
        for pct_change in price_change_pct:
            if pct_change > threshold:
                signals.append('BUY')
            elif pct_change < -threshold:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        signal_df = pd.DataFrame({
            'current_price': current_prices,
            'predicted_price': predictions,
            'change_pct': price_change_pct * 100,
            'signal': signals
        })
        
        return signal_df
    
    def backtest_predictions(self, y_true, y_pred, dates=None):
        """
        Backtest prediction performance
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Index dates (optional)
            
        Returns:
            DataFrame with backtest results
        """
        errors = y_true - y_pred
        pct_errors = (errors / y_true) * 100
        
        results = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'error': errors,
            'pct_error': pct_errors
        })
        
        if dates is not None:
            results.index = dates
        
        # Add cumulative metrics
        results['cumulative_error'] = results['error'].cumsum()
        
        return results
