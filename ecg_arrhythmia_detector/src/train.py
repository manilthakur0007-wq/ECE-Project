"""
Main Training Script
====================
Full end-to-end pipeline:

  1. Download MIT-BIH records (skips cached files)
  2. Preprocess signals (bandpass + notch)
  3. Segment beats around R-peaks
  4. Extract HRV feature vectors
  5. Train / validate split (80/20 stratified)
  6. Train CNN-LSTM on normalised beat waveforms
  7. Train GBT on HRV features
  8. Evaluate ensemble on the held-out test set
  9. Save both models to models/

Usage
-----
    python -m src.train                          # full 48-record run
    python -m src.train --records 100 108 200    # quick subset
    python -m src.train --skip-download          # use cached data only
    python -m src.train --epochs 5 --records 100 # fast smoke-test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ECG arrhythmia ensemble")
    p.add_argument(
        "--records", nargs="+", default=None,
        metavar="REC",
        help="MIT-BIH record IDs to use (default: all 48)",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading; use only locally cached records",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override EPOCHS in config",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Override DATA_DIR",
    )
    p.add_argument(
        "--test-size", type=float, default=0.20,
        help="Fraction of data to hold out for evaluation (default 0.20)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args(argv)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_download(records: list[str], data_dir: Path, skip: bool) -> None:
    if skip:
        log.info("--skip-download: skipping PhysioNet download.")
        return
    from src.data_loader import download_all
    log.info("Downloading %d MIT-BIH records …", len(records))
    download_all(records=records, dest=data_dir)


def step_build_dataset(
    records: list[str],
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    from src.data_loader import build_dataset
    from src.preprocessing import preprocess

    log.info("Loading and preprocessing records …")
    beats, labels, raw_signals, r_samples_list = build_dataset(
        records=records,
        data_dir=data_dir,
        preprocessor=preprocess,
    )
    log.info(
        "Dataset: %d beats | class counts: %s",
        len(beats),
        {c: int((labels == c).sum()) for c in np.unique(labels)},
    )
    return beats, labels, raw_signals, r_samples_list


def step_extract_hrv(
    beats: np.ndarray,
    r_samples_list: list,
) -> np.ndarray:
    from src.features import extract_features_for_dataset

    log.info("Extracting HRV features …")
    X_hrv = extract_features_for_dataset(beats, r_samples_list)
    log.info("HRV feature matrix: %s", X_hrv.shape)
    return X_hrv


def step_split(
    beats: np.ndarray,
    X_hrv: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple:
    from src.preprocessing import normalize_beats

    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed
    )

    X_beats_raw = normalize_beats(beats)

    X_beats_train, X_beats_test = X_beats_raw[idx_train], X_beats_raw[idx_test]
    X_hrv_train,   X_hrv_test   = X_hrv[idx_train],       X_hrv[idx_test]
    y_train,       y_test        = labels[idx_train],       labels[idx_test]

    log.info("Train: %d  |  Test: %d", len(y_train), len(y_test))
    return X_beats_train, X_beats_test, X_hrv_train, X_hrv_test, y_train, y_test


def step_train_ensemble(
    X_beats_train, X_beats_test,
    X_hrv_train,   X_hrv_test,
    y_train,       y_test,
    epochs: int | None,
) -> "EnsembleClassifier":  # noqa: F821
    from src import config as cfg
    from src.ensemble import EnsembleClassifier, print_classification_report

    if epochs is not None:
        cfg.EPOCHS = epochs

    # Val split from training set
    idx = np.arange(len(y_train))
    idx_t, idx_v = train_test_split(idx, test_size=0.15, stratify=y_train, random_state=0)

    ens = EnsembleClassifier()
    ens.fit(
        X_beats_train[idx_t], X_hrv_train[idx_t], y_train[idx_t],
        X_beats_train[idx_v], X_hrv_train[idx_v], y_train[idx_v],
    )
    ens.save()

    log.info("Evaluating on held-out test set …")
    y_pred = ens.predict(X_beats_test, X_hrv_test)
    print_classification_report(y_test, y_pred)

    return ens


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)

    from src import config as cfg

    records  = args.records  or cfg.MITBIH_RECORDS
    data_dir = args.data_dir or cfg.DATA_DIR

    np.random.seed(args.seed)

    step_download(records, data_dir, args.skip_download)
    beats, labels, raw_signals, r_samples_list = step_build_dataset(records, data_dir)

    if len(beats) == 0:
        log.error("No beats loaded. Ensure records are downloaded to %s", data_dir)
        sys.exit(1)

    X_hrv = step_extract_hrv(beats, r_samples_list)

    (
        X_beats_train, X_beats_test,
        X_hrv_train,   X_hrv_test,
        y_train,       y_test,
    ) = step_split(beats, X_hrv, labels, args.test_size, args.seed)

    step_train_ensemble(
        X_beats_train, X_beats_test,
        X_hrv_train,   X_hrv_test,
        y_train,       y_test,
        epochs=args.epochs,
    )

    log.info("Training complete. Models saved to %s", cfg.MODELS_DIR)


if __name__ == "__main__":
    main()
