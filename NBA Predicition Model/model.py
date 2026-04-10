"""
Prediction Model Module — Moneyline Only
──────────────────────────────────────────
Ensemble of XGBoost, LightGBM, and Logistic Regression.

Trains a single classification target: home_win (0/1).
The ensemble averages calibrated probabilities from all classifiers,
producing a home win probability used to find moneyline edge.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import CV_FOLDS, MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, TEST_SIZE
from features import build_features, get_feature_columns

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ═══════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def build_classifiers() -> dict:
    """Build the ensemble of classifiers for win probability."""
    models = {}

    # Logistic Regression (strong baseline, well-calibrated)
    models["logistic"] = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs"
    )

    # XGBoost (captures non-linear interactions)
    if HAS_XGB:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )

    # LightGBM (fast, handles categorical features well)
    if HAS_LGB:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            verbose=-1,
        )

    return models


def build_regressors() -> dict:
    """Build regressors for spread and total predictions."""
    models = {}

    # Ridge regression baseline
    models["ridge"] = Ridge(alpha=1.0, random_state=RANDOM_SEED)

    if HAS_XGB:
        models["xgboost"] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbosity=0,
        )

    return models


# ═══════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

class NBAPredictor:
    """
    Full prediction pipeline: trains models, generates calibrated
    probabilities, predicts spreads & totals.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifiers: dict = {}
        self.calibrated_classifiers: dict = {}
        self.spread_models: dict = {}
        self.total_models: dict = {}
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def train(self, games: pd.DataFrame) -> dict:
        """
        Train all models using time-series aware splitting.
        Returns evaluation metrics.
        """
        print("\n[Model Training]")
        self.feature_cols = get_feature_columns(games)
        print(f"  Using {len(self.feature_cols)} features")

        X = games[self.feature_cols].values
        y_win = games["home_win"].values
        y_spread = games["spread"].values
        y_total = games["total_pts"].values

        # Time-series split: train on earlier games, test on later
        n = len(X)
        split_idx = int(n * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_win_train, y_win_test = y_win[:split_idx], y_win[split_idx:]
        y_spread_train, y_spread_test = y_spread[:split_idx], y_spread[split_idx:]
        y_total_train, y_total_test = y_total[:split_idx], y_total[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        metrics = {}

        # ── Train win probability classifiers ──
        print("\n  Training win probability models...")
        self.classifiers = build_classifiers()
        for name, clf in self.classifiers.items():
            print(f"    {name}...")
            if name == "logistic":
                clf.fit(X_train_scaled, y_win_train)
                # Calibrate
                cal_clf = CalibratedClassifierCV(clf, cv=3, method="isotonic")
                cal_clf.fit(X_train_scaled, y_win_train)
            else:
                clf.fit(X_train, y_win_train)
                cal_clf = CalibratedClassifierCV(clf, cv=3, method="isotonic")
                cal_clf.fit(X_train, y_win_train)

            self.calibrated_classifiers[name] = cal_clf

            # Evaluate
            if name == "logistic":
                probs = cal_clf.predict_proba(X_test_scaled)[:, 1]
                preds = (probs > 0.5).astype(int)
            else:
                probs = cal_clf.predict_proba(X_test)[:, 1]
                preds = (probs > 0.5).astype(int)

            metrics[f"{name}_accuracy"] = accuracy_score(y_win_test, preds)
            metrics[f"{name}_auc"] = roc_auc_score(y_win_test, probs)
            metrics[f"{name}_brier"] = brier_score_loss(y_win_test, probs)
            metrics[f"{name}_logloss"] = log_loss(y_win_test, probs)

            print(f"      Accuracy: {metrics[f'{name}_accuracy']:.4f}")
            print(f"      AUC:      {metrics[f'{name}_auc']:.4f}")
            print(f"      Brier:    {metrics[f'{name}_brier']:.4f}")

        # ── Ensemble win probability ──
        print("\n    Ensemble...")
        ensemble_probs = self._ensemble_predict_proba(X_test, X_test_scaled)
        ensemble_preds = (ensemble_probs > 0.5).astype(int)

        metrics["ensemble_accuracy"] = accuracy_score(y_win_test, ensemble_preds)
        metrics["ensemble_auc"] = roc_auc_score(y_win_test, ensemble_probs)
        metrics["ensemble_brier"] = brier_score_loss(y_win_test, ensemble_probs)
        metrics["ensemble_logloss"] = log_loss(y_win_test, ensemble_probs)

        print(f"      Accuracy: {metrics['ensemble_accuracy']:.4f}")
        print(f"      AUC:      {metrics['ensemble_auc']:.4f}")
        print(f"      Brier:    {metrics['ensemble_brier']:.4f}")

        # ── ELO baseline comparison ──
        if "elo_home_win_prob" in games.columns:
            elo_probs = games["elo_home_win_prob"].values[split_idx:]
            elo_preds = (elo_probs > 0.5).astype(int)
            metrics["elo_accuracy"] = accuracy_score(y_win_test, elo_preds)
            metrics["elo_auc"] = roc_auc_score(y_win_test, elo_probs)
            metrics["elo_brier"] = brier_score_loss(y_win_test, elo_probs)
            print(f"\n    ELO baseline:")
            print(f"      Accuracy: {metrics['elo_accuracy']:.4f}")
            print(f"      AUC:      {metrics['elo_auc']:.4f}")
            print(f"      Brier:    {metrics['elo_brier']:.4f}")

        # ── Train spread regressors ──
        print("\n  Training spread prediction models...")
        self.spread_models = build_regressors()
        for name, reg in self.spread_models.items():
            if name == "ridge":
                reg.fit(X_train_scaled, y_spread_train)
                spread_pred = reg.predict(X_test_scaled)
            else:
                reg.fit(X_train, y_spread_train)
                spread_pred = reg.predict(X_test)

            mae = mean_absolute_error(y_spread_test, spread_pred)
            metrics[f"spread_{name}_mae"] = mae
            print(f"    {name} MAE: {mae:.2f} points")

        # ── Train total regressors ──
        print("\n  Training total points prediction models...")
        self.total_models = build_regressors()
        for name, reg in self.total_models.items():
            if name == "ridge":
                reg.fit(X_train_scaled, y_total_train)
                total_pred = reg.predict(X_test_scaled)
            else:
                reg.fit(X_train, y_total_train)
                total_pred = reg.predict(X_test)

            mae = mean_absolute_error(y_total_test, total_pred)
            metrics[f"total_{name}_mae"] = mae
            print(f"    {name} MAE: {mae:.2f} points")

        self.is_fitted = True

        # Save model
        self.save()
        print(f"\n  Models saved to {MODELS_DIR}")

        return metrics

    def _ensemble_predict_proba(self, X_raw: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
        """Average calibrated probabilities from all classifiers."""
        all_probs = []
        for name, cal_clf in self.calibrated_classifiers.items():
            if name == "logistic":
                probs = cal_clf.predict_proba(X_scaled)[:, 1]
            else:
                probs = cal_clf.predict_proba(X_raw)[:, 1]
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for new games.
        Returns DataFrame with win_prob, predicted_spread, predicted_total.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call .train() first.")

        X = features[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Win probability (ensemble)
        win_prob = self._ensemble_predict_proba(X, X_scaled)

        # Spread prediction (average across regressors)
        spread_preds = []
        for name, reg in self.spread_models.items():
            if name == "ridge":
                spread_preds.append(reg.predict(X_scaled))
            else:
                spread_preds.append(reg.predict(X))
        predicted_spread = np.mean(spread_preds, axis=0)

        # Total prediction
        total_preds = []
        for name, reg in self.total_models.items():
            if name == "ridge":
                total_preds.append(reg.predict(X_scaled))
            else:
                total_preds.append(reg.predict(X))
        predicted_total = np.mean(total_preds, axis=0)

        results = features[["game_id", "date", "home_team", "away_team"]].copy()
        results["home_win_prob"] = win_prob
        results["away_win_prob"] = 1 - win_prob
        results["predicted_spread"] = np.round(predicted_spread, 1)
        results["predicted_total"] = np.round(predicted_total, 1)

        return results

    def save(self, path: Path | None = None):
        """Serialize the full model to disk."""
        path = path or MODELS_DIR / "nba_predictor.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "classifiers": self.classifiers,
                "calibrated_classifiers": self.calibrated_classifiers,
                "spread_models": self.spread_models,
                "total_models": self.total_models,
                "feature_cols": self.feature_cols,
            }, f)

    def load(self, path: Path | None = None):
        """Load a trained model from disk."""
        path = path or MODELS_DIR / "nba_predictor.pkl"
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.scaler = state["scaler"]
        self.classifiers = state["classifiers"]
        self.calibrated_classifiers = state["calibrated_classifiers"]
        self.spread_models = state["spread_models"]
        self.total_models = state["total_models"]
        self.feature_cols = state["feature_cols"]
        self.is_fitted = True


# ═══════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def cross_validate_model(games: pd.DataFrame) -> pd.DataFrame:
    """
    Run time-series cross-validation and return per-fold metrics.
    Uses expanding window: train on all data before fold, test on fold.
    """
    feature_cols = get_feature_columns(games)
    X = games[feature_cols].values
    y = games["home_win"].values

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n  Fold {fold + 1}/{CV_FOLDS}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train simple LogReg for CV speed
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(X_train_s, y_train)

        probs = clf.predict_proba(X_test_s)[:, 1]
        preds = (probs > 0.5).astype(int)

        fold_metrics.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": accuracy_score(y_test, preds),
            "auc": roc_auc_score(y_test, probs),
            "brier": brier_score_loss(y_test, probs),
            "logloss": log_loss(y_test, probs),
        })

        print(f"    Accuracy: {fold_metrics[-1]['accuracy']:.4f}")
        print(f"    AUC:      {fold_metrics[-1]['auc']:.4f}")

    return pd.DataFrame(fold_metrics)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading feature data...")
    games = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])

    print("\nRunning cross-validation...")
    cv_results = cross_validate_model(games)
    print(f"\nCV Summary:\n{cv_results.describe()}")

    print("\nTraining full model...")
    predictor = NBAPredictor()
    metrics = predictor.train(games)

    print("\n" + "=" * 60)
    print("FINAL METRICS SUMMARY")
    print("=" * 60)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
