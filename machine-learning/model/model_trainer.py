"""
Model training module with ensemble methods
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from utils.logger import setup_logger
from utils.file_ops import save_model
from config.settings import OPTIMIZED_PARAMS, MODEL_DIR
import os

logger = setup_logger(__name__)

class ModelTrainer:
    """Train and manage ML models for stock prediction"""
    
    def __init__(self, model_type='ensemble', params=None):
        self.model_type = model_type
        self.params = params or OPTIMIZED_PARAMS
        self.model = None
        
    def create_model(self):
        """Create model based on type"""
        logger.info(f"🤖 Creating {self.model_type} model...")
        
        if self.model_type == 'RandomForest':
            self.model = RandomForestRegressor(**self.params['RandomForest'])
            
        elif self.model_type == 'GradientBoosting':
            self.model = GradientBoostingRegressor(**self.params['GradientBoosting'])
            
        elif self.model_type == 'ExtraTrees':
            self.model = ExtraTreesRegressor(**self.params['ExtraTrees'])
            
        elif self.model_type == 'voting':
            # Voting ensemble
            estimators = [
                ('rf', RandomForestRegressor(**self.params['RandomForest'])),
                ('gb', GradientBoostingRegressor(**self.params['GradientBoosting'])),
                ('et', ExtraTreesRegressor(**self.params['ExtraTrees']))
            ]
            self.model = VotingRegressor(estimators=estimators)
            
        elif self.model_type == 'stacking':
            # Stacking ensemble
            estimators = [
                ('rf', RandomForestRegressor(**self.params['RandomForest'])),
                ('gb', GradientBoostingRegressor(**self.params['GradientBoosting'])),
                ('et', ExtraTreesRegressor(**self.params['ExtraTrees']))
            ]
            self.model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(),
                cv=5
            )
            
        else:  # default ensemble
            # Best performing: Stacking
            estimators = [
                ('rf', RandomForestRegressor(**self.params['RandomForest'])),
                ('gb', GradientBoostingRegressor(**self.params['GradientBoosting'])),
                ('et', ExtraTreesRegressor(**self.params['ExtraTrees']))
            ]
            self.model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(),
                cv=5
            )
        
        logger.info(f"✅ Model created: {type(self.model).__name__}")
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        logger.info(f"🎯 Training {self.model_type} model...")
        logger.info(f"   Training samples: {len(X_train)}")
        
        if self.model is None:
            self.create_model()
        
        self.model.fit(X_train, y_train)
        
        logger.info("✅ Model training complete!")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X)
        
        return predictions
    
    def save(self, filepath=None, target_name='model'):
        """
        Save trained model
        
        Args:
            filepath: Path to save model (optional)
            target_name: Name for the model file
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, f'{target_name}_{self.model_type}.pkl')
        
        save_model(self.model, filepath)
        
        return filepath
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (if supported by model)
        
        Args:
            feature_names: List of feature names
            
        Returns:
            dict of feature importances or None
        """
        if self.model is None:
            return None
        
        # For ensemble models, try to get importance from final estimator
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'final_estimator_') and hasattr(self.model.final_estimator_, 'coef_'):
            importances = np.abs(self.model.final_estimator_.coef_)
        else:
            logger.warning("Model doesn't support feature importance")
            return None
        
        if feature_names is not None:
            importance_dict = dict(zip(feature_names, importances))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importance_dict
        
        return importances
