"""
Live in-tournament XGBoost model for Augusta National.

Architecture:
  - Target A: top10 (binary classifier) — primary metric AUC
  - Target B: finish_pct (regression) — secondary metric Spearman

Blend: as holes_completed increases, live features get progressively more weight.
  - 0 holes: 100% pre-tournament baseline
  - 9 holes: ~70% live + 30% baseline
  - 18 holes: ~90% live + 10% baseline

Train: 2015-2023, Val: 2024, Test: 2025
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost not installed — run: pip install xgboost")

from augusta_model.features.live_features import get_live_feature_columns

MODEL_DIR = Path(__file__).parents[2] / "models"
DATA_DIR = Path(__file__).parents[2] / "data" / "processed"

MODEL_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = list(range(2015, 2024))   # 2015-2023
VAL_YEAR = 2024
TEST_YEAR = 2025

XGB_TOP10_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.02,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

XGB_REGRESSION_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


def _prepare_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Select and fill feature columns."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    # Fill missing with median / 0
    for col in X.columns:
        if X[col].dtype in (np.float64, np.float32, float):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)
    return X


def train_live_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    snapshot_hole: Optional[int] = None,
    verbose: bool = True,
) -> tuple:
    """
    Train top10 classifier and finish_pct regression model.

    If snapshot_hole is specified, only use data from that snapshot.
    Otherwise, train on all snapshot points (the model gets holes_completed_pct
    as a feature so it can self-adjust).

    Returns: (clf_top10, reg_finish, feature_cols, metadata)
    """
    feature_cols = get_live_feature_columns()

    if snapshot_hole is not None:
        train_data = train_data[train_data["snapshot_hole"] == snapshot_hole]
        val_data = val_data[val_data["snapshot_hole"] == snapshot_hole]

    X_train = _prepare_features(train_data, feature_cols)
    y_top10_train = train_data["top10"].fillna(0).astype(int)
    y_finish_train = train_data["finish_pct"].fillna(0.5)

    X_val = _prepare_features(val_data, feature_cols)
    y_top10_val = val_data["top10"].fillna(0).astype(int)
    y_finish_val = val_data["finish_pct"].fillna(0.5)

    used_cols = list(X_train.columns)

    # Top-10 classifier
    neg = (y_top10_train == 0).sum()
    pos = (y_top10_train == 1).sum()
    scale_pos = float(neg / pos) if pos > 0 else 5.0

    clf = xgb.XGBClassifier(
        **XGB_TOP10_PARAMS,
        scale_pos_weight=scale_pos,
    )
    clf.fit(
        X_train,
        y_top10_train,
        eval_set=[(X_val, y_top10_val)],
        verbose=False,
    )

    # Finish regression
    reg = xgb.XGBRegressor(**XGB_REGRESSION_PARAMS)
    reg.fit(X_train, y_finish_train, verbose=False)

    # Validation metrics
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds = reg.predict(X_val)

    auc = roc_auc_score(y_top10_val, val_probs) if y_top10_val.sum() > 0 else float("nan")
    spear = spearmanr(y_finish_val, val_preds)[0]

    # Top-10 precision @ top-13 selections
    n_top = min(13, len(val_data))
    val_eval = val_data.copy()
    val_eval = val_eval.drop_duplicates(subset=["player_name", "year"])
    if "snapshot_hole" not in val_eval.columns or snapshot_hole is not None:
        X_val_nodup = _prepare_features(val_eval, used_cols)
        probs_nodup = clf.predict_proba(X_val_nodup)[:, 1]
        val_eval = val_eval.assign(_prob=probs_nodup)
    else:
        val_eval = val_eval[val_eval["snapshot_hole"] == 18].drop_duplicates("player_name")
        X_val_nodup = _prepare_features(val_eval, used_cols)
        probs_nodup = clf.predict_proba(X_val_nodup)[:, 1]
        val_eval = val_eval.assign(_prob=probs_nodup)

    top_picks = val_eval.nlargest(n_top, "_prob")
    t10_prec = float(top_picks["top10"].mean()) if len(top_picks) > 0 else 0.0

    metadata = {
        "train_years": TRAIN_YEARS,
        "val_year": VAL_YEAR,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "feature_cols": used_cols,
        "val_auc_top10": round(auc, 4),
        "val_spearman": round(float(spear), 4),
        "val_top10_precision": round(t10_prec, 3),
        "snapshot_hole": snapshot_hole,
    }

    if verbose:
        print(f"  Val AUC(top10): {auc:.3f}  |  Spearman: {spear:.3f}  |  T10 Prec@13: {t10_prec:.1%}")

    return clf, reg, used_cols, metadata


def save_live_model(clf, reg, feature_cols: list[str], metadata: dict) -> None:
    """Persist model artifacts."""
    MODEL_DIR.mkdir(exist_ok=True)
    clf.save_model(str(MODEL_DIR / "live_clf_top10.json"))
    reg.save_model(str(MODEL_DIR / "live_reg_finish.json"))
    with open(MODEL_DIR / "live_feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODEL_DIR / "live_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Models saved to {MODEL_DIR}/")


def load_live_model() -> tuple:
    """Load persisted model artifacts."""
    clf = xgb.XGBClassifier()
    clf.load_model(str(MODEL_DIR / "live_clf_top10.json"))
    reg = xgb.XGBRegressor()
    reg.load_model(str(MODEL_DIR / "live_reg_finish.json"))
    with open(MODEL_DIR / "live_feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(MODEL_DIR / "live_metadata.json") as f:
        metadata = json.load(f)
    return clf, reg, feature_cols, metadata


def predict_live(
    snapshot_df: pd.DataFrame,
    clf,
    reg,
    feature_cols: list[str],
    predictions_2026: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate live probability updates for a snapshot DataFrame.

    snapshot_df should have one row per player (latest snapshot).
    If predictions_2026 is provided, blends live model with pre-tournament odds.

    Returns DataFrame with updated win/top10/top20 probability estimates.
    """
    from augusta_model.features.live_features import add_pretournament_baseline

    if predictions_2026 is not None:
        snapshot_df = add_pretournament_baseline(snapshot_df, predictions_2026)

    X = _prepare_features(snapshot_df, feature_cols)

    # Fill any columns that the model was trained on but snapshot doesn't have
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_cols]

    top10_probs = clf.predict_proba(X)[:, 1]
    finish_preds = reg.predict(X)

    result = snapshot_df[["player_name"]].copy()
    result["live_top10_prob"] = top10_probs
    result["live_finish_pred"] = finish_preds
    result["holes_completed"] = snapshot_df.get("holes_completed", 0)
    result["cumulative_score_to_par"] = snapshot_df.get("cumulative_score_to_par", 0)

    # Blend with pre-tournament if available
    confidence = snapshot_df.get("confidence_weight", pd.Series(0.5, index=snapshot_df.index))
    if isinstance(confidence, (int, float)):
        confidence = pd.Series(confidence, index=snapshot_df.index)

    if "top10_prob" in snapshot_df.columns:
        pre_top10 = snapshot_df["top10_prob"].fillna(0.114)
        result["blended_top10_prob"] = (
            confidence * top10_probs + (1 - confidence) * pre_top10
        )
    else:
        result["blended_top10_prob"] = top10_probs

    # Normalise top10 probs so they sum to 10.0
    total = result["blended_top10_prob"].sum()
    if total > 0:
        result["blended_top10_prob"] = result["blended_top10_prob"] * (10.0 / total)

    # Win probability: approximate from top10 × pre-tournament win/top10 ratio
    if "win_prob" in snapshot_df.columns and "top10_prob" in snapshot_df.columns:
        ratio = (snapshot_df["win_prob"] / snapshot_df["top10_prob"].clip(lower=0.01)).fillna(0.1)
        result["blended_win_prob"] = (result["blended_top10_prob"] * ratio).clip(upper=1.0)
    else:
        result["blended_win_prob"] = (result["blended_top10_prob"] * 0.1).clip(upper=1.0)

    # Normalise win probs to sum=1
    win_total = result["blended_win_prob"].sum()
    if win_total > 0:
        result["blended_win_prob"] = result["blended_win_prob"] / win_total

    # Compute movement vs pre-tournament
    if "top10_prob" in snapshot_df.columns:
        result["top10_prob_delta"] = result["blended_top10_prob"] - snapshot_df["top10_prob"].values * 10.0
    else:
        result["top10_prob_delta"] = 0.0

    return result.sort_values("blended_top10_prob", ascending=False).reset_index(drop=True)


def run_full_training(verbose: bool = True) -> dict:
    """
    Load training data, train model, save artifacts.

    Returns evaluation metrics.
    """
    data_path = DATA_DIR / "live_training_data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Run: python3 scripts/build_live_training_data.py"
        )

    print("Loading training data...")
    data = pd.read_parquet(data_path)
    print(f"  Total rows: {len(data):,}")
    print(f"  Years: {sorted(data['year'].unique())}")

    train = data[data["year"].isin(TRAIN_YEARS)]
    val = data[data["year"] == VAL_YEAR]
    test = data[data["year"] == TEST_YEAR]

    print(f"\nSplit: train={len(train):,} | val={len(val):,} | test={len(test):,}")

    # Train on ALL snapshot points simultaneously
    print("\nTraining global live model (all snapshot points)...")
    clf, reg, feature_cols, meta = train_live_model(train, val, verbose=verbose)
    save_live_model(clf, reg, feature_cols, meta)

    # Evaluate on test year
    if len(test) > 0:
        print("\nTest year evaluation (2025):")
        X_test = _prepare_features(test, feature_cols)
        # Ensure all feature cols present
        for col in feature_cols:
            if col not in X_test.columns:
                X_test[col] = 0.0
        X_test = X_test[feature_cols]

        y_test = test["top10"].fillna(0).astype(int)
        probs = clf.predict_proba(X_test)[:, 1]
        finish = reg.predict(X_test)

        auc = roc_auc_score(y_test, probs) if y_test.sum() > 0 else float("nan")
        spear = spearmanr(test["finish_pct"].fillna(0.5), finish)[0]
        print(f"  Test AUC(top10): {auc:.3f}  |  Spearman: {spear:.3f}")
        meta["test_auc_top10"] = round(float(auc), 4)
        meta["test_spearman"] = round(float(spear), 4)

    # Snapshot-by-snapshot breakdown
    print("\nBreakdown by snapshot hole (validation year 2024):")
    snap_metrics = []
    for snap in [3, 6, 9, 12, 15, 18]:
        val_snap = val[val["snapshot_hole"] == snap]
        if len(val_snap) == 0:
            continue
        X_s = _prepare_features(val_snap, feature_cols)
        for col in feature_cols:
            if col not in X_s.columns:
                X_s[col] = 0.0
        X_s = X_s[feature_cols]
        y_s = val_snap["top10"].fillna(0).astype(int)
        probs_s = clf.predict_proba(X_s)[:, 1]
        finish_s = reg.predict(X_s)
        auc_s = roc_auc_score(y_s, probs_s) if y_s.sum() > 0 else float("nan")
        spear_s = spearmanr(val_snap["finish_pct"].fillna(0.5), finish_s)[0]
        snap_metrics.append({
            "snapshot": snap,
            "n": len(val_snap),
            "auc": round(float(auc_s), 3),
            "spearman": round(float(spear_s), 3),
        })
        print(f"  Hole {snap:2d}: AUC={auc_s:.3f}  Spearman={spear_s:.3f}  n={len(val_snap)}")

    meta["snapshot_metrics_val"] = snap_metrics

    # Resave with test metrics
    with open(MODEL_DIR / "live_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
