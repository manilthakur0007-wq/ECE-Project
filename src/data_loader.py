"""
Download and load MIT-BIH Arrhythmia Database records via wfdb.

Each record is a 30-minute, 360 Hz, two-channel Holter recording.
Annotations use symbol codes that we map to 5 AAMI arrhythmia classes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import wfdb

from src.config import (
    AAMI_MAP,
    BEAT_LEN,
    BEAT_PRE_MS,
    DATA_DIR,
    FS,
    MITBIH_DB,
    MITBIH_RECORDS,
    N_CLASSES,
)

log = logging.getLogger(__name__)


# ── Download ──────────────────────────────────────────────────────────────────

def download_record(record_id: str, dest: Path = DATA_DIR) -> None:
    """Download a single MIT-BIH record (signal + annotation) if not cached."""
    dest.mkdir(parents=True, exist_ok=True)
    sig_path = dest / f"{record_id}.hea"
    if sig_path.exists():
        log.debug("Record %s already cached – skipping download.", record_id)
        return
    log.info("Downloading record %s …", record_id)
    wfdb.dl_database(
        db_dir=MITBIH_DB,
        dl_dir=str(dest),
        records=[record_id],
        annotators=["atr"],
    )


def download_all(records: list[str] = MITBIH_RECORDS, dest: Path = DATA_DIR) -> None:
    """Download all MIT-BIH records listed in *records*."""
    for rec in records:
        try:
            download_record(rec, dest)
        except Exception as exc:          # noqa: BLE001
            log.warning("Failed to download record %s: %s", rec, exc)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_record(
    record_id: str,
    channel: int = 0,
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single record and return (signal, r_sample_indices, aami_labels).

    Parameters
    ----------
    record_id : str
        MIT-BIH record number, e.g. ``"100"``.
    channel : int
        Channel index (0 = MLII for most records).
    data_dir : Path
        Directory that contains downloaded .hea / .dat / .atr files.

    Returns
    -------
    signal : np.ndarray, shape (N,)
        Raw ECG signal (physical units, mV).
    r_samples : np.ndarray, shape (B,)
        Sample indices of annotated beat locations.
    labels : np.ndarray, shape (B,)
        AAMI class index for each beat (0–4).
    """
    rec_path = str(data_dir / record_id)
    record = wfdb.rdrecord(rec_path, channels=[channel])
    annotation = wfdb.rdann(rec_path, "atr")

    signal: np.ndarray = record.p_signal[:, 0].astype(np.float32)

    beat_mask = np.isin(annotation.symbol, list(AAMI_MAP.keys()))
    r_samples = annotation.sample[beat_mask]
    raw_symbols = np.array(annotation.symbol)[beat_mask]
    labels = np.array([AAMI_MAP.get(s, 4) for s in raw_symbols], dtype=np.int32)

    return signal, r_samples, labels


# ── Beat segmentation ─────────────────────────────────────────────────────────

def _pre_samples() -> int:
    return int(FS * BEAT_PRE_MS / 1000)


def segment_beats(
    signal: np.ndarray,
    r_samples: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut fixed-length windows centred (with *pre*-sample offset) at each R-peak.

    Returns
    -------
    beats : np.ndarray, shape (B, BEAT_LEN)
    valid_labels : np.ndarray, shape (B,)
    """
    pre = _pre_samples()
    beats, valid_labels = [], []

    for idx, (r, lbl) in enumerate(zip(r_samples, labels)):
        start = r - pre
        end   = start + BEAT_LEN
        if start < 0 or end > len(signal):
            continue
        beats.append(signal[start:end])
        valid_labels.append(lbl)

    return np.array(beats, dtype=np.float32), np.array(valid_labels, dtype=np.int32)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    records: list[str] = MITBIH_RECORDS,
    data_dir: Path = DATA_DIR,
    channel: int = 0,
    preprocessor=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build raw beat arrays and annotation arrays from a list of records.

    Parameters
    ----------
    preprocessor : callable or None
        If provided, called as ``preprocessor(signal, fs)`` before segmentation.

    Returns
    -------
    beats      : np.ndarray, shape (total_beats, BEAT_LEN)
    labels     : np.ndarray, shape (total_beats,)
    raw_signals : list[np.ndarray]   – one full-record signal per record
    r_samples_list : list[np.ndarray]
    """
    all_beats, all_labels = [], []
    raw_signals: list[np.ndarray] = []
    r_samples_list: list[np.ndarray] = []

    for rec in records:
        try:
            signal, r_samples, labels = load_record(rec, channel, data_dir)
        except FileNotFoundError:
            log.warning("Record %s not found – skipping.", rec)
            continue

        if preprocessor is not None:
            signal = preprocessor(signal, FS)

        beats, valid_labels = segment_beats(signal, r_samples, labels)
        if len(beats) == 0:
            continue

        all_beats.append(beats)
        all_labels.append(valid_labels)
        raw_signals.append(signal)
        r_samples_list.append(r_samples)

    beats  = np.concatenate(all_beats,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return beats, labels, raw_signals, r_samples_list


# ── Record streaming iterator ─────────────────────────────────────────────────

def stream_record(
    record_id: str,
    channel: int = 0,
    data_dir: Path = DATA_DIR,
    preprocessor=None,
) -> Iterator[tuple[np.ndarray, int]]:
    """
    Yield (beat_waveform, aami_label) one beat at a time from a record.
    Useful for the real-time demo.
    """
    signal, r_samples, labels = load_record(record_id, channel, data_dir)
    if preprocessor is not None:
        signal = preprocessor(signal, FS)
    beats, valid_labels = segment_beats(signal, r_samples, labels)
    for beat, lbl in zip(beats, valid_labels):
        yield beat, int(lbl)
