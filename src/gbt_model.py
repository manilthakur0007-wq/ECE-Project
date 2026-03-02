"""
Gradient Boosted Trees (GBT) Classifier for HRV Feature Classification
========================================================================
Uses scikit-learn's HistGradientBoostingClassifier for speed and built-in
handling of class imbalance via sample_weight / class_weight.

The GBT operates on the 9-dimensional HRV feature vector produced by
features.extract_hrv_features(), complementing the CNN-LSTM waveform model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    GBT_LEARNING_RATE,
    GBT_MAX_DEPTH,
    GBT_N_ESTIMATORS,
    MODELS_DIR,
    N_CLASSES,
)

log = logging.getLogger(__name__)

_MODEL_PATH = MODELS_DIR / "gbt_model.joblib"


# ── Model builder ─────────────────────────────────────────────────────────────

def build_gbt(
    n_estimators: int = GBT_N_ESTIMATORS,
    max_depth: int = GBT_MAX_DEPTH,
    learning_rate: float = GBT_LEARNING_RATE,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline: StandardScaler → HistGradientBoostingClassifier.

    HistGradientBoostingClassifier natively supports missing values and is
    significantly faster than GradientBoostingClassifier on large datasets.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    gbt = HistGradientBoostingClassifier(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        scoring="loss",
        random_state=42,
        verbose=0,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gbt",    gbt),
    ])
    return pipeline


# ── Training ──────────────────────────────────────────────────────────────────

def train_gbt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_path: Path = _MODEL_PATH,
) -> Pipeline:
    """
    Fit the GBT pipeline on HRV features with balanced class weights.

    Parameters
    ----------
    X_train   : np.ndarray, shape (N, N_HRV_FEATURES)
    y_train   : np.ndarray, shape (N,)  AAMI class indices
    model_path : Path  where to persist the trained pipeline

    Returns
    -------
    fitted Pipeline
    """
    pipeline = build_gbt()

    # Balanced sample weights to handle class imbalance
    sample_weights = compute_sample_weight("balanced", y=y_train)

    log.info("Training GBT on %d samples …", len(X_train))
    pipeline.fit(X_train, y_train, gbt__sample_weight=sample_weights)

    joblib.dump(pipeline, model_path)
    log.info("GBT saved to %s", model_path)
    return pipeline


# ── Inference ─────────────────────────────────────────────────────────────────

def load_gbt(model_path: Path = _MODEL_PATH) -> Pipeline:
    """Load a previously saved GBT pipeline."""
    return joblib.load(model_path)


def predict_proba_gbt(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Return class probability matrix for HRV feature rows.

    Parameters
    ----------
    X : np.ndarray, shape (N, N_HRV_FEATURES)

    Returns
    -------
    np.ndarray, shape (N, N_CLASSES)
        Rows that correspond to classes not seen during training are filled with 0.
    """
    raw_proba = pipeline.predict_proba(X)           # shape (N, seen_classes)
    n_samples  = X.shape[0]

    # Map trained classes to full N_CLASSES columns
    trained_classes = pipeline.named_steps["gbt"].classes_
    proba = np.zeros((n_samples, N_CLASSES), dtype=np.float32)
    for col_idx, cls in enumerate(trained_classes):
        if cls < N_CLASSES:
            proba[:, cls] = raw_proba[:, col_idx]
    return proba
