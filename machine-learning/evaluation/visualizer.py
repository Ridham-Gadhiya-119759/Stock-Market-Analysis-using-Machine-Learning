"""
Visualization module for plotting results
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import setup_logger
from config.settings import PLOTS_DIR, FIGURE_DPI

logger = setup_logger(__name__)

class Visualizer:
    """Create visualizations for model results"""
    
    def __init__(self, ticker_name='Stock'):
        self.ticker_name = ticker_name
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('viridis')
        
    def plot_predictions(self, y_true, y_pred, dates, target_name, save_dir=None):
        """
        Plot actual vs predicted values
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Date index
            target_name: Name of target variable
            save_dir: Directory to save plot
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series plot
        ax1.plot(dates, y_true, label='Actual', color='#2E86AB', linewidth=2, marker='o', markersize=4)
        ax1.plot(dates, y_pred, label='Predicted', color='#F77F00', linewidth=2, marker='s', markersize=4)
        ax1.fill_between(dates, y_true, y_pred, alpha=0.2, color='red')
        ax1.set_title(f'{target_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Value', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='black')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_title('Correlation Plot', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Actual', fontsize=11)
        ax2.set_ylabel('Predicted', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        if save_dir is None:
            save_dir = PLOTS_DIR
        filename = f'{target_name}_predictions.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved plot: {filepath}")
        
        return filepath
    
    def plot_error_analysis(self, y_true, y_pred, target_name, save_dir=None):
        """
        Plot error distribution and analysis
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            target_name: Name of target variable
            save_dir: Directory to save plot
            
        Returns:
            Path to saved plot
        """
        errors = y_true - y_pred
        pct_errors = (errors / y_true) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error distribution
        axes[0, 0].hist(errors, bins=30, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Error')
        axes[0, 0].set_ylabel('Frequency')
        
        # Percentage error distribution
        axes[0, 1].hist(pct_errors, bins=30, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Percentage Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Percentage Error (%)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Error over time
        axes[1, 0].plot(errors, color='purple', linewidth=1.5)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Error Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Error')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'{target_name} - Error Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if save_dir is None:
            save_dir = PLOTS_DIR
        filename = f'{target_name}_error_analysis.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved plot: {filepath}")
        
        return filepath
    
    def plot_feature_importance(self, importances, feature_names, target_name, top_n=20, save_dir=None):
        """
        Plot feature importance
        
        Args:
            importances: Feature importance values
            feature_names: Feature names
            target_name: Name of target variable
            top_n: Number of top features to show
            save_dir: Directory to save plot
            
        Returns:
            Path to saved plot
        """
        # Sort and get top features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_importances, color=colors, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'{target_name} - Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save
        if save_dir is None:
            save_dir = PLOTS_DIR
        filename = f'{target_name}_feature_importance.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved plot: {filepath}")
        
        return filepath
    
    def create_summary_dashboard(self, all_results, save_dir=None):
        """
        Create comprehensive dashboard with all targets
        
        Args:
            all_results: Dict of results for all targets
            save_dir: Directory to save plot
            
        Returns:
            Path to saved plot
        """
        n_targets = len(all_results)
        fig, axes = plt.subplots(n_targets, 2, figsize=(16, 5*n_targets))
        
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (target_name, result) in enumerate(all_results.items()):
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            # Time series
            ax1 = axes[idx, 0]
            ax1.plot(y_true, label='Actual', linewidth=2)
            ax1.plot(y_pred, label='Predicted', linewidth=2)
            ax1.set_title(f'{target_name} - Time Series', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Scatter
            ax2 = axes[idx, 1]
            ax2.scatter(y_true, y_pred, alpha=0.6)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            ax2.set_title(f'{target_name} - Correlation', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Predicted')
            ax2.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics = result.get('metrics', {})
            metrics_text = f"R²: {metrics.get('r2', 0):.3f}\nMAE: {metrics.get('mae', 0):.2f}\nDir Acc: {metrics.get('directional_accuracy', 0):.1f}%"
            ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.suptitle(f'{self.ticker_name} - Complete Forecast Dashboard', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if save_dir is None:
            save_dir = PLOTS_DIR
        filename = 'complete_dashboard.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved dashboard: {filepath}")
        
        return filepath
