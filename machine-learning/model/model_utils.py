"""
Model utilities - Save/load and hyperparameter tuning
"""
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils.logger import setup_logger
from utils.file_ops import save_model, load_model
from config.settings import MODEL_DIR

logger = setup_logger(__name__)

class ModelUtils:
    """Utilities for model management and optimization"""
    
    @staticmethod
    def save_model_and_scaler(model, scaler, target_name, model_type='ensemble'):
        """
        Save both model and scaler
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            target_name: Name of the target variable
            model_type: Type of model
            
        Returns:
            dict with paths to saved files
        """
        model_path = os.path.join(MODEL_DIR, f'{target_name}_{model_type}_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'{target_name}_{model_type}_scaler.pkl')
        
        save_model(model, model_path)
        save_model(scaler, scaler_path)
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path
        }
    
    @staticmethod
    def load_model_and_scaler(target_name, model_type='ensemble'):
        """
        Load both model and scaler
        
        Args:
            target_name: Name of the target variable
            model_type: Type of model
            
        Returns:
            tuple of (model, scaler)
        """
        model_path = os.path.join(MODEL_DIR, f'{target_name}_{model_type}_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'{target_name}_{model_type}_scaler.pkl')
        
        model = load_model(model_path)
        scaler = load_model(scaler_path)
        
        return model, scaler
    
    @staticmethod
    def hyperparameter_tuning(X_train, y_train, model_type='RandomForest', 
                            search_type='random', n_iter=20, cv=5):
        """
        Perform hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to tune
            search_type: 'random' or 'grid'
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            
        Returns:
            Best model and parameters
        """
        logger.info(f"🔍 Performing {search_type} search for {model_type}...")
        
        # Define parameter grids
        if model_type == 'RandomForest':
            model = RandomForestRegressor(random_state=42)
            param_dist = {
                'n_estimators': [100, 150, 200, 250],
                'max_depth': [10, 15, 20, 25, 30],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'max_features': ['sqrt', 'log2', None]
            }
        elif model_type == 'GradientBoosting':
            model = GradientBoostingRegressor(random_state=42)
            param_dist = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.7, 0.8, 0.9, 1.0]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Perform search
        if search_type == 'random':
            search = RandomizedSearchCV(
                model, param_dist,
                n_iter=n_iter,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            search = GridSearchCV(
                model, param_dist,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        
        logger.info(f"✅ Best parameters: {search.best_params_}")
        logger.info(f"✅ Best score: {-search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_
    
    @staticmethod
    def cross_validate_model(model, X, y, cv=5):
        """
        Perform time-series cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            dict with validation scores
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        logger.info(f"🔄 Performing {cv}-fold time-series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=cv)
        scores = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_cv = X[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X[val_idx]
            y_val_cv = y.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            
            scores['mse'].append(mean_squared_error(y_val_cv, y_pred))
            scores['mae'].append(mean_absolute_error(y_val_cv, y_pred))
            scores['r2'].append(r2_score(y_val_cv, y_pred))
            
            logger.info(f"  Fold {fold}: MSE={scores['mse'][-1]:.4f}, MAE={scores['mae'][-1]:.4f}, R²={scores['r2'][-1]:.4f}")
        
        # Calculate mean scores
        mean_scores = {k: np.mean(v) for k, v in scores.items()}
        std_scores = {k: np.std(v) for k, v in scores.items()}
        
        logger.info(f"✅ Mean scores: MSE={mean_scores['mse']:.4f}±{std_scores['mse']:.4f}, "
                   f"MAE={mean_scores['mae']:.4f}±{std_scores['mae']:.4f}, "
                   f"R²={mean_scores['r2']:.4f}±{std_scores['r2']:.4f}")
        
        return {'mean': mean_scores, 'std': std_scores, 'all': scores}
