"""
Soft-Vote Ensemble Classifier
==============================
Combines the CNN-LSTM waveform model and the GBT HRV-feature model via a
weighted soft-vote strategy:

    P_ensemble = w_cnn * P_cnn + w_gbt * P_gbt

Default weights (config.ENSEMBLE_WEIGHTS): [0.6, 0.4].

The ensemble exposes a unified interface for training, saving, loading,
and inference that the real-time demo and the main training script both use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import CLASS_NAMES, ENSEMBLE_WEIGHTS, MODELS_DIR, N_CLASSES
from src.cnn_lstm_model import (
    load_cnn_lstm,
    predict_proba_cnn_lstm,
    train_cnn_lstm,
)
from src.gbt_model import load_gbt, predict_proba_gbt, train_gbt

log = logging.getLogger(__name__)

_ENSEMBLE_META_PATH = MODELS_DIR / "ensemble_meta.joblib"


@dataclass
class EnsembleClassifier:
    """
    Soft-vote ensemble of CNN-LSTM and GBT models.

    Attributes
    ----------
    cnn_lstm   : Keras Model   waveform classifier
    gbt        : sklearn Pipeline   HRV feature classifier
    weights    : list[float]   [w_cnn, w_gbt], must sum to 1
    """
    cnn_lstm: Any = field(default=None)
    gbt:      Any = field(default=None)
    weights:  list[float] = field(default_factory=lambda: ENSEMBLE_WEIGHTS)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_beats_train: np.ndarray,
        X_hrv_train:   np.ndarray,
        y_train:       np.ndarray,
        X_beats_val:   np.ndarray,
        X_hrv_val:     np.ndarray,
        y_val:         np.ndarray,
    ) -> "EnsembleClassifier":
        """
        Train both sub-models independently and store them.

        Parameters
        ----------
        X_beats_train : (N, BEAT_LEN)    normalised beat waveforms
        X_hrv_train   : (N, N_HRV_FEATURES)  HRV feature vectors
        y_train       : (N,)             AAMI labels
        *_val         : validation counterparts
        """
        log.info("=== Fitting CNN-LSTM ===")
        self.cnn_lstm = train_cnn_lstm(
            X_beats_train, y_train,
            X_beats_val,   y_val,
        )

        log.info("=== Fitting GBT ===")
        self.gbt = train_gbt(X_hrv_train, y_train)

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        beats: np.ndarray,
        X_hrv: np.ndarray,
    ) -> np.ndarray:
        """
        Weighted soft-vote probability matrix.

        Parameters
        ----------
        beats : np.ndarray, shape (N, BEAT_LEN)
        X_hrv : np.ndarray, shape (N, N_HRV_FEATURES)

        Returns
        -------
        np.ndarray, shape (N, N_CLASSES)
        """
        p_cnn = predict_proba_cnn_lstm(self.cnn_lstm, beats)
        p_gbt = predict_proba_gbt(self.gbt, X_hrv)

        w_cnn, w_gbt = self.weights
        return w_cnn * p_cnn + w_gbt * p_gbt

    def predict(
        self,
        beats: np.ndarray,
        X_hrv: np.ndarray,
    ) -> np.ndarray:
        """Return argmax class indices, shape (N,)."""
        proba = self.predict_proba(beats, X_hrv)
        return np.argmax(proba, axis=1).astype(np.int32)

    def predict_single(
        self,
        beat: np.ndarray,
        hrv_feat: np.ndarray,
    ) -> tuple[int, str, np.ndarray]:
        """
        Classify a single beat.

        Returns
        -------
        class_idx : int
        class_name : str
        proba : np.ndarray, shape (N_CLASSES,)
        """
        proba = self.predict_proba(
            beat[np.newaxis, :],
            hrv_feat[np.newaxis, :],
        )[0]
        cls_idx = int(np.argmax(proba))
        return cls_idx, CLASS_NAMES[cls_idx], proba

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, meta_path: Path = _ENSEMBLE_META_PATH) -> None:
        """Save ensemble weights and GBT (CNN-LSTM is saved separately via Keras)."""
        meta = {"weights": self.weights}
        joblib.dump(meta, meta_path)
        log.info("Ensemble metadata saved to %s", meta_path)

    @classmethod
    def load(
        cls,
        meta_path: Path = _ENSEMBLE_META_PATH,
        cnn_path:  Path | None = None,
        gbt_path:  Path | None = None,
    ) -> "EnsembleClassifier":
        """Load a saved ensemble from disk."""
        meta = joblib.load(meta_path)
        cnn_lstm = load_cnn_lstm(cnn_path) if cnn_path else load_cnn_lstm()
        gbt      = load_gbt(gbt_path)      if gbt_path  else load_gbt()
        return cls(cnn_lstm=cnn_lstm, gbt=gbt, weights=meta["weights"])


# ── Evaluation helpers ────────────────────────────────────────────────────────

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    header = " " * 20 + "  ".join(f"{n[:4]:>4}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:<20}" + "  ".join(f"{v:>4}" for v in row))
    print()
