import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class TreeModels:
    def __init__(self):
        self.models = {
            "random_forest": RandomForestClassifier(random_state=42),
            "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "lightgbm": LGBMClassifier(random_state=42)
        }
        
    def get_model(self, name: str):
        if name not in self.models:
            raise ValueError(f"Model {name} not supported.")
        return self.models[name]
        
    def save_model(self, model, path: str):
        joblib.dump(model, path)
        
    def load_model(self, path: str):
        return joblib.load(path)
