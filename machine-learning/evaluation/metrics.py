"""
Evaluation metrics calculation
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MetricsCalculator:
    """Calculate various performance metrics"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred):
        """
        Calculate comprehensive set of metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            dict with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (handle division by zero)
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            mask = y_true != 0
            if mask.sum() > 0:
                metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                metrics['mape'] = np.nan
        
        # Directional accuracy
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction) * 100
        else:
            metrics['directional_accuracy'] = np.nan
        
        # Correlation
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Error statistics
        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['max_error'] = np.max(np.abs(errors))
        
        # Percentage errors
        pct_errors = np.abs((y_true - y_pred) / y_true) * 100
        metrics['within_5pct'] = np.sum(pct_errors < 5) / len(pct_errors) * 100
        metrics['within_10pct'] = np.sum(pct_errors < 10) / len(pct_errors) * 100
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, title="Model Performance"):
        """Pretty print metrics"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 {title}")
        logger.info(f"{'='*60}")
        logger.info(f"R² Score:               {metrics['r2']:.4f}")
        logger.info(f"Mean Absolute Error:    {metrics['mae']:.4f}")
        logger.info(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
        logger.info(f"MAPE:                   {metrics['mape']:.2f}%")
        logger.info(f"Directional Accuracy:   {metrics['directional_accuracy']:.2f}%")
        logger.info(f"Correlation:            {metrics['correlation']:.4f}")
        logger.info(f"Predictions within ±5%: {metrics['within_5pct']:.1f}%")
        logger.info(f"Predictions within ±10%: {metrics['within_10pct']:.1f}%")
        logger.info(f"{'='*60}\n")
    
    @staticmethod
    def compare_models(results_dict):
        """
        Compare multiple models
        
        Args:
            results_dict: Dict of {model_name: metrics_dict}
            
        Returns:
            DataFrame with comparison
        """
        import pandas as pd
        
        comparison = pd.DataFrame(results_dict).T
        comparison = comparison.sort_values('r2', ascending=False)
        
        logger.info("\n📊 Model Comparison:")
        logger.info(comparison.to_string())
        
        return comparison
