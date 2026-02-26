"""
model.py — Ensemble model: LightGBM + XGBoost + Logistic Regression + LSTM.

Design:
  - Binary classification: +1 (bull) vs -1 (bear), trained on no-trade-filtered rows
  - Probability threshold: only trade if max(P) >= CONF_THRESHOLD
  - SHAP-based feature importance ranking (saved to disk)
  - Walk-forward safe: no future leakage (StandardScaler fitted on train only)
  - Per-regime models: separate ensemble per regime (trending/ranging)
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    warnings.warn("xgboost not installed; XGBoost model will be skipped in ensemble.")

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    warnings.warn("torch not installed; LSTM model will be skipped in ensemble.")

# ─── Config ───────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.60          # minimum predicted probability to trade
LSTM_SEQ_LEN   = 20            # look-back window for LSTM
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─── LSTM ─────────────────────────────────────────────────────────────────────
if _HAS_TORCH:
    class _LSTMClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :]))


def _build_lstm_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Convert 2D feature matrix to 3D sequences [N, seq_len, features]."""
    out = []
    for i in range(seq_len, len(X) + 1):
        out.append(X[i - seq_len:i])
    return np.array(out)


def _train_lstm(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, seq_len: int = LSTM_SEQ_LEN,
                epochs: int = 30, lr: float = 1e-3) -> Optional[Any]:
    if not _HAS_TORCH or len(X_train) < seq_len * 2:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_seq = _build_lstm_sequences(X_train, seq_len)
    y_seq = y_train[seq_len - 1:]      # align labels
    if len(X_seq) == 0 or len(np.unique(y_seq)) < 2:
        return None

    X_t = torch.FloatTensor(X_seq).to(device)
    y_t = torch.FloatTensor((y_seq == 1).astype(float)).unsqueeze(1).to(device)

    model = _LSTMClassifier(X_train.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCELoss()

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = crit(pred, y_t)
        loss.backward()
        opt.step()

    model.eval()
    return model


def _lstm_predict_proba(model: Any, X: np.ndarray, seq_len: int = LSTM_SEQ_LEN) -> np.ndarray:
    """Return (N, 2) probability array [P(bear), P(bull)]."""
    if model is None or not _HAS_TORCH:
        return None
    device = next(model.parameters()).device
    X_seq = _build_lstm_sequences(X, seq_len)
    if len(X_seq) == 0:
        return None
    with torch.no_grad():
        p_bull = model(torch.FloatTensor(X_seq).to(device)).cpu().numpy().flatten()
    # Pad head with 0.5 (no prediction for first seq_len-1 rows)
    pad = np.full((seq_len - 1, 2), 0.5)
    proba = np.stack([1 - p_bull, p_bull], axis=1)
    return np.vstack([pad, proba])


# ─── Per-class Ensemble ────────────────────────────────────────────────────────
class RegimeEnsemble:
    """
    Ensemble of LightGBM + XGBoost + LogReg (+ optional LSTM) for one regime.
    Predict by averaging softmax probabilities.
    """

    def __init__(self, regime_label: int = 0):
        self.regime_label = regime_label
        self.scaler = StandardScaler()
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.xgb_model: Optional[Any] = None
        self.lr_model:  Optional[LogisticRegression] = None
        self.lstm_model: Optional[Any] = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_names = list(X_train.columns)
        X = self.scaler.fit_transform(X_train.values)

        # Map labels -1,+1 → 0,1 for sklearn compatibility
        y = (y_train.values == 1).astype(int)
        if len(np.unique(y)) < 2:
            print(f"[model/regime={self.regime_label}] Skipping: only one class present.")
            return

        print(f"[model/regime={self.regime_label}] Training on {len(y)} samples "
              f"({y.sum()} bull, {(1-y).sum()} bear)")

        # LightGBM
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        self.lgb_model.fit(X, y,
            eval_set=[(X, y)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)])

        # XGBoost
        if _HAS_XGB:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.7,
                scale_pos_weight=(1-y).sum() / (y.sum() + 1e-9),
                eval_metric="logloss", use_label_encoder=False,
                random_state=42, verbosity=0,
            )
            self.xgb_model.fit(X, y)

        # Logistic Regression (baseline)
        self.lr_model = LogisticRegression(
            max_iter=500, C=0.1, class_weight="balanced", solver="lbfgs"
        )
        self.lr_model.fit(X, y)

        # LSTM
        self.lstm_model = _train_lstm(X, y_train.values, X)

        self.is_fitted = True
        print(f"[model/regime={self.regime_label}] Fitted ✓")

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """Return (N, 2) array: [P(bear), P(bull)] averaged across ensemble."""
        if not self.is_fitted:
            return np.full((len(X_df), 2), 0.5)

        X = self.scaler.transform(X_df[self.feature_names].values)
        probas = []

        if self.lgb_model:
            probas.append(self.lgb_model.predict_proba(X))
        if self.xgb_model and _HAS_XGB:
            probas.append(self.xgb_model.predict_proba(X))
        if self.lr_model:
            probas.append(self.lr_model.predict_proba(X))
        if self.lstm_model:
            lp = _lstm_predict_proba(self.lstm_model, X)
            if lp is not None:
                # Align length (LSTM pads head rows)
                probas.append(lp[-len(X):])

        if not probas:
            return np.full((len(X), 2), 0.5)
        avg = np.mean(probas, axis=0)
        return avg

    def predict_with_confidence(self, X_df: pd.DataFrame,
                                 threshold: float = CONF_THRESHOLD) -> pd.DataFrame:
        """
        Return DataFrame with columns: pred (-1, 0, +1), confidence.
        pred=0 means 'no trade' (confidence below threshold).
        """
        proba = self.predict_proba(X_df)
        p_bull = proba[:, 1]
        p_bear = proba[:, 0]
        pred = np.where(p_bull >= threshold, 1,
               np.where(p_bear >= threshold, -1, 0))
        conf = np.maximum(p_bull, p_bear)
        return pd.DataFrame({"pred": pred, "confidence": conf,
                              "p_bull": p_bull, "p_bear": p_bear},
                             index=X_df.index)

    def save(self, prefix: str) -> None:
        if not self.is_fitted:
            return
        path = MODELS_DIR / f"{prefix}_regime{self.regime_label}"
        joblib.dump(self.lgb_model,  str(path) + "_lgb.pkl")
        if self.xgb_model and _HAS_XGB:
            joblib.dump(self.xgb_model, str(path) + "_xgb.pkl")
        joblib.dump(self.lr_model,   str(path) + "_lr.pkl")
        joblib.dump(self.scaler,     str(path) + "_scaler.pkl")
        with open(str(path) + "_meta.json", "w") as f:
            json.dump({"features": self.feature_names,
                       "regime": self.regime_label}, f)
        print(f"[model] Saved regime {self.regime_label} → {path}*")

    @classmethod
    def load(cls, prefix: str, regime_label: int) -> "RegimeEnsemble":
        path = MODELS_DIR / f"{prefix}_regime{regime_label}"
        ens = cls(regime_label)
        ens.lgb_model = joblib.load(str(path) + "_lgb.pkl")
        if _HAS_XGB and (path.parent / (path.name + "_xgb.pkl")).exists():
            ens.xgb_model = joblib.load(str(path) + "_xgb.pkl")
        ens.lr_model  = joblib.load(str(path) + "_lr.pkl")
        ens.scaler    = joblib.load(str(path) + "_scaler.pkl")
        with open(str(path) + "_meta.json") as f:
            meta = json.load(f)
        ens.feature_names = meta["features"]
        ens.is_fitted = True
        return ens


# ─── SHAP Feature Importance ──────────────────────────────────────────────────
def compute_shap_importance(model: lgb.LGBMClassifier,
                             X: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X.values)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class 1 (bull)
        imp = pd.DataFrame({
            "feature": X.columns,
            "shap_mean_abs": np.abs(shap_vals).mean(axis=0)
        }).sort_values("shap_mean_abs", ascending=False)
        return imp.head(top_n)
    except ImportError:
        # Fall back to LightGBM built-in importance
        imp = pd.DataFrame({
            "feature": X.columns,
            "shap_mean_abs": model.feature_importances_
        }).sort_values("shap_mean_abs", ascending=False)
        return imp.head(top_n)
