"""
CNN-LSTM Model for Beat-Level Arrhythmia Classification
=========================================================
Architecture
------------
Input  : (batch, BEAT_LEN, 1)  — raw beat waveform
Block 1: Conv1D(32, 5) → BN → ReLU → MaxPool(2)
Block 2: Conv1D(64, 5) → BN → ReLU → MaxPool(2)
Block 3: Conv1D(128, 5) → BN → ReLU → MaxPool(2)
LSTM   : LSTM(64, return_sequences=False) + Dropout(0.3)
Output : Dense(5, softmax)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    BATCH_SIZE,
    BEAT_LEN,
    CNN_FILTERS,
    CNN_KERNEL,
    DROPOUT_RATE,
    EPOCHS,
    LEARNING_RATE,
    LSTM_UNITS,
    MODELS_DIR,
    N_CLASSES,
)

log = logging.getLogger(__name__)

_MODEL_PATH = MODELS_DIR / "cnn_lstm.keras"


# ── Model builder ─────────────────────────────────────────────────────────────

def build_cnn_lstm(
    beat_len: int = BEAT_LEN,
    n_classes: int = N_CLASSES,
    cnn_filters: list[int] = CNN_FILTERS,
    kernel_size: int = CNN_KERNEL,
    lstm_units: int = LSTM_UNITS,
    dropout: float = DROPOUT_RATE,
) -> keras.Model:
    """
    Construct and compile the CNN-LSTM model.

    Returns
    -------
    keras.Model  (untrainedweights)
    """
    inp = keras.Input(shape=(beat_len, 1), name="beat_input")
    x = inp

    # Convolutional feature extraction
    for n_filters in cnn_filters:
        x = layers.Conv1D(
            n_filters, kernel_size, padding="same", use_bias=False, name=f"conv_{n_filters}"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    # Temporal feature learning
    x = layers.LSTM(lstm_units, name="lstm")(x)
    x = layers.Dropout(dropout)(x)

    # Classification head
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="cnn_lstm_ecg")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_cnn_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: Path = _MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    class_weight: dict | None = None,
) -> keras.Model:
    """
    Train the CNN-LSTM model with early stopping and class-imbalance weighting.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, BEAT_LEN)   normalised beat waveforms
    y_train : np.ndarray, shape (N,)             AAMI class indices
    X_val   : np.ndarray, shape (M, BEAT_LEN)
    y_val   : np.ndarray, shape (M,)
    model_path : Path   where to save the best model
    class_weight : dict or None   {class_index: weight}

    Returns
    -------
    keras.Model  trained model
    """
    # Add channel dimension required by Conv1D
    X_train_3d = X_train[..., np.newaxis]
    X_val_3d   = X_val[..., np.newaxis]

    model = build_cnn_lstm()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    if class_weight is None:
        class_weight = _compute_class_weights(y_train)

    log.info("Training CNN-LSTM for up to %d epochs …", epochs)
    model.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return model


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


# ── Inference ─────────────────────────────────────────────────────────────────

def load_cnn_lstm(model_path: Path = _MODEL_PATH) -> keras.Model:
    """Load a saved CNN-LSTM model from disk."""
    return keras.models.load_model(str(model_path))


def predict_proba_cnn_lstm(
    model: keras.Model,
    beats: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Return class probabilities for an array of beat waveforms.

    Parameters
    ----------
    beats : np.ndarray, shape (N, BEAT_LEN)

    Returns
    -------
    np.ndarray, shape (N, N_CLASSES)
    """
    X = beats[..., np.newaxis]
    return model.predict(X, batch_size=batch_size, verbose=0)
