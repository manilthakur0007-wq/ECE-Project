"""
Heart Rate Variability (HRV) Feature Extraction
================================================
Extracts the following features from a sequence of RR intervals:

Time-domain
-----------
* rr_mean     – mean RR interval (s)
* rr_std      – standard deviation of RR intervals (SDNN)
* rmssd       – root mean square of successive differences
* pnn50       – proportion of successive differences > 50 ms

Frequency-domain (Lomb-Scargle periodogram)
-------------------------------------------
* lf_power    – power in LF band (0.04 – 0.15 Hz)
* hf_power    – power in HF band (0.15 – 0.40 Hz)
* lf_hf_ratio – LF / HF ratio

Beat morphology (from the raw beat waveform)
--------------------------------------------
* qrs_duration – width of QRS complex (samples / fs  → seconds)
* qrs_amplitude – peak-to-trough amplitude within QRS window
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle

from src.config import (
    BEAT_LEN,
    FS,
    HRV_HF_BAND,
    HRV_LF_BAND,
    PNN50_THRESH,
)

# Feature vector length (used by GBT)
N_HRV_FEATURES = 9


# ── Time-domain HRV ───────────────────────────────────────────────────────────

def compute_rr_intervals(r_samples: np.ndarray, fs: float = FS) -> np.ndarray:
    """Convert R-peak sample indices to RR intervals in seconds."""
    if len(r_samples) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(r_samples).astype(np.float64) / fs


def rr_mean(rr: np.ndarray) -> float:
    return float(np.mean(rr)) if len(rr) > 0 else 0.0


def rr_std(rr: np.ndarray) -> float:
    return float(np.std(rr)) if len(rr) > 1 else 0.0


def rmssd(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return 0.0
    successive_diff = np.diff(rr)
    return float(np.sqrt(np.mean(successive_diff ** 2)))


def pnn50(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return 0.0
    successive_diff = np.abs(np.diff(rr))
    return float(np.mean(successive_diff > PNN50_THRESH))


# ── Frequency-domain HRV (Lomb-Scargle) ──────────────────────────────────────

def _band_power(
    freqs: np.ndarray,
    power: np.ndarray,
    band: tuple[float, float],
) -> float:
    """Integrate power spectral density over a frequency band."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    return float(np.trapz(power[mask], freqs[mask]))


def lomb_scargle_psd(
    rr: np.ndarray,
    fs_resample: float = 4.0,
    n_freqs: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Lomb-Scargle periodogram of unevenly-sampled RR intervals.

    Parameters
    ----------
    rr : np.ndarray, shape (K,)  RR intervals in seconds
    fs_resample : float          frequency resolution reference
    n_freqs : int                number of frequency points

    Returns
    -------
    freqs : np.ndarray, shape (n_freqs,)
    power : np.ndarray, shape (n_freqs,)
    """
    if len(rr) < 4:
        freqs = np.linspace(0.001, 0.5, n_freqs)
        return freqs, np.zeros(n_freqs)

    # Cumulative time of each beat
    t = np.cumsum(rr)
    t -= t[0]   # start at zero

    # Angular frequencies
    freqs = np.linspace(0.001, 0.5, n_freqs)
    omegas = 2.0 * np.pi * freqs

    # Detrend RR intervals before Lomb-Scargle
    rr_detrended = rr - np.mean(rr)

    power = lombscargle(t, rr_detrended, omegas, normalize=True)
    return freqs, power


def frequency_features(rr: np.ndarray) -> dict[str, float]:
    freqs, power = lomb_scargle_psd(rr)
    lf = _band_power(freqs, power, HRV_LF_BAND)
    hf = _band_power(freqs, power, HRV_HF_BAND)
    ratio = lf / (hf + 1e-10)
    return {"lf_power": lf, "hf_power": hf, "lf_hf_ratio": ratio}


# ── Beat morphology features ──────────────────────────────────────────────────

def qrs_features(beat: np.ndarray, fs: float = FS) -> dict[str, float]:
    """
    Estimate QRS duration and amplitude from a single beat waveform.

    Uses a simple energy-based approach: QRS is assumed to lie within
    ±75 ms of the R-peak, which is positioned at BEAT_PRE_MS into the window.
    """
    from src.config import BEAT_PRE_MS  # local import to avoid circular
    r_idx = int(fs * BEAT_PRE_MS / 1000)
    half  = int(fs * 0.075)                        # ±75 ms
    lo = max(0, r_idx - half)
    hi = min(len(beat), r_idx + half)
    qrs_segment = beat[lo:hi]

    duration  = (hi - lo) / fs                    # seconds
    amplitude = float(np.max(qrs_segment) - np.min(qrs_segment))
    return {"qrs_duration": duration, "qrs_amplitude": amplitude}


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_hrv_features(
    rr: np.ndarray,
    beat: np.ndarray | None = None,
    fs: float = FS,
) -> np.ndarray:
    """
    Build a feature vector of length N_HRV_FEATURES (9).

    Parameters
    ----------
    rr   : np.ndarray  RR intervals in seconds (at least 2 elements)
    beat : np.ndarray or None  single beat waveform for morphology features.
           If None, QRS features are set to 0.
    fs   : float  sampling rate

    Returns
    -------
    np.ndarray, shape (N_HRV_FEATURES,)
    Order: [rr_mean, rr_std, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio,
            qrs_duration, qrs_amplitude]
    """
    td = {
        "rr_mean": rr_mean(rr),
        "rr_std":  rr_std(rr),
        "rmssd":   rmssd(rr),
        "pnn50":   pnn50(rr),
    }
    fd = frequency_features(rr)

    if beat is not None:
        md = qrs_features(beat, fs)
    else:
        md = {"qrs_duration": 0.0, "qrs_amplitude": 0.0}

    vec = [
        td["rr_mean"],
        td["rr_std"],
        td["rmssd"],
        td["pnn50"],
        fd["lf_power"],
        fd["hf_power"],
        fd["lf_hf_ratio"],
        md["qrs_duration"],
        md["qrs_amplitude"],
    ]
    return np.array(vec, dtype=np.float32)


def extract_features_for_dataset(
    beats: np.ndarray,
    r_samples_list: list[np.ndarray],
    fs: float = FS,
) -> np.ndarray:
    """
    Extract one HRV feature vector per beat across all records.

    Parameters
    ----------
    beats         : np.ndarray, shape (total_beats, BEAT_LEN)
    r_samples_list : list of per-record R-peak arrays

    Returns
    -------
    X_hrv : np.ndarray, shape (total_beats, N_HRV_FEATURES)
    """
    feature_rows: list[np.ndarray] = []

    beat_cursor = 0
    for r_samples in r_samples_list:
        rr = compute_rr_intervals(r_samples, fs)
        n_beats = len(r_samples) - 1  # one fewer RR than R-peaks

        # Pad RR sequence so it aligns beat-by-beat
        rr_padded = np.concatenate([[rr[0] if len(rr) > 0 else 0.75], rr])

        for local_i in range(n_beats + 1):
            global_i = beat_cursor + local_i
            if global_i >= len(beats):
                break
            beat  = beats[global_i]
            # Use a sliding window of RR intervals centred on current beat
            rr_window = rr_padded[max(0, local_i - 10): local_i + 1]
            fv = extract_hrv_features(rr_window, beat, fs)
            feature_rows.append(fv)

        beat_cursor += n_beats + 1

    if not feature_rows:
        return np.zeros((len(beats), N_HRV_FEATURES), dtype=np.float32)

    X = np.array(feature_rows, dtype=np.float32)
    # Trim or pad to match beats length exactly
    if len(X) > len(beats):
        X = X[: len(beats)]
    elif len(X) < len(beats):
        pad = np.zeros((len(beats) - len(X), N_HRV_FEATURES), dtype=np.float32)
        X = np.vstack([X, pad])
    return X
