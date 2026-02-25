import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import optuna
from typing import Dict, Any

from models.tree_models import TreeModels

class TrainingPipeline:
    def __init__(self, n_splits=5):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.tree_factory = TreeModels()
        
    def optimize_tree_model(self, X: pd.DataFrame, y: pd.Series, model_name: str, n_trials=20) -> Dict[str, Any]:
        """
        Use Optuna to discover best parameters for Tree models using TimeSeriesSplit.
        """
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'random_state': 42
                }
                model = self.tree_factory.get_model('random_forest')
                model.set_params(**params)
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    'random_state': 42
                }
                model = self.tree_factory.get_model('xgboost')
                model.set_params(**params)
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'random_state': 42
                }
                model = self.tree_factory.get_model('lightgbm')
                model.set_params(**params)
            else:
                raise ValueError("Unsupported model for optimization")
                
            scores = []
            for train_idx, val_idx in self.tscv.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                loss = log_loss(y_val, preds)
                scores.append(loss)
                
            return np.mean(scores)
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def train_final_tree_model(self, X: pd.DataFrame, y: pd.Series, model_name: str, best_params: Dict[str, Any]):
        model = self.tree_factory.get_model(model_name)
        model.set_params(**best_params)
        model.fit(X, y)
        
        # Feature importance extract
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(X.columns, model.feature_importances_))
            
        return model, importance
