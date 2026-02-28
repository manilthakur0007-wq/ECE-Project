"""
ECG Signal Preprocessing Pipeline
===================================
1. Butterworth bandpass filter  (0.5 – 40 Hz)  – removes baseline wander & HF noise
2. IIR notch filter             (60 Hz)         – removes power-line interference
3. Pan-Tompkins QRS detector                    – R-peak localisation

References
----------
Pan, J. & Tompkins, W. J. (1985). "A real-time QRS detection algorithm."
IEEE Transactions on Biomedical Engineering, 32(3), 230–236.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp

from src.config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BANDPASS_ORDER,
    FS,
    NOTCH_FREQ,
    NOTCH_Q,
)


# ── 1. Bandpass filter ────────────────────────────────────────────────────────

def bandpass_filter(ecg: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth bandpass filter (0.5 – 40 Hz).

    Parameters
    ----------
    ecg : np.ndarray, shape (N,)
    fs  : float  sampling rate in Hz

    Returns
    -------
    np.ndarray, shape (N,)  filtered signal
    """
    nyq = fs / 2.0
    low  = BANDPASS_LOW  / nyq
    high = BANDPASS_HIGH / nyq
    b, a = sp.butter(BANDPASS_ORDER, [low, high], btype="bandpass")
    return sp.filtfilt(b, a, ecg).astype(np.float32)


# ── 2. Notch filter ───────────────────────────────────────────────────────────

def notch_filter(ecg: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Zero-phase IIR notch filter to remove 60 Hz power-line noise.

    Parameters
    ----------
    ecg : np.ndarray, shape (N,)
    fs  : float  sampling rate in Hz

    Returns
    -------
    np.ndarray, shape (N,)  filtered signal
    """
    b, a = sp.iirnotch(NOTCH_FREQ, NOTCH_Q, fs)
    return sp.filtfilt(b, a, ecg).astype(np.float32)


# ── Combined preprocessor ─────────────────────────────────────────────────────

def preprocess(ecg: np.ndarray, fs: float = FS) -> np.ndarray:
    """Apply bandpass then notch filter."""
    ecg = bandpass_filter(ecg, fs)
    ecg = notch_filter(ecg, fs)
    return ecg


# ── 3. Pan-Tompkins QRS detector ─────────────────────────────────────────────

class PanTompkinsDetector:
    """
    Pure-NumPy / SciPy implementation of the Pan-Tompkins algorithm.

    Steps
    -----
    1. Derivative filter         – enhances QRS slope
    2. Squaring                  – all positive, non-linear amplification
    3. Moving-window integration – smoothing (window ≈ 150 ms)
    4. Adaptive thresholding     – dual thresholds on signal & noise peaks
    5. Refractory period         – ≥ 200 ms between R-peaks

    Usage
    -----
    >>> detector = PanTompkinsDetector(fs=360)
    >>> r_peaks = detector.detect(filtered_ecg)
    """

    def __init__(self, fs: float = FS) -> None:
        self.fs = fs
        self._integration_window = int(0.150 * fs)   # 150 ms
        self._refractory        = int(0.200 * fs)    # 200 ms

    # ── internal filters ──────────────────────────────────────────────────────

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """Five-point derivative filter from Pan & Tompkins (1985)."""
        fs = self.fs
        # H(z) = (1/8T)(-z^{-2} - 2z^{-1} + 2z + z^{2})
        h = np.array([-1, -2, 0, 2, 1], dtype=np.float32) * (1.0 / (8.0 / fs))
        return np.convolve(x, h, mode="same")

    def _integrate(self, x: np.ndarray) -> np.ndarray:
        kernel = np.ones(self._integration_window) / self._integration_window
        return np.convolve(x, kernel, mode="same")

    # ── peak detection ────────────────────────────────────────────────────────

    @staticmethod
    def _find_peaks(x: np.ndarray, min_distance: int = 1) -> np.ndarray:
        """Return indices where x[i] > x[i-1] and x[i] > x[i+1]."""
        peaks = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
        idx = np.where(peaks)[0] + 1

        # enforce minimum distance
        if len(idx) < 2:
            return idx
        selected = [idx[0]]
        for i in idx[1:]:
            if i - selected[-1] >= min_distance:
                selected.append(i)
        return np.array(selected)

    # ── main detection ────────────────────────────────────────────────────────

    def detect(self, ecg: np.ndarray) -> np.ndarray:
        """
        Detect R-peak sample indices in a pre-filtered ECG signal.

        Parameters
        ----------
        ecg : np.ndarray, shape (N,)
            Must already be bandpass-filtered.

        Returns
        -------
        r_peaks : np.ndarray, shape (K,)
            Sample indices of detected R-peaks in ascending order.
        """
        # Step 1-3: derivative → square → integrate
        diff   = self._derivative(ecg)
        squared = diff ** 2
        mwi    = self._integrate(squared)

        # Step 4: adaptive thresholding
        # Initialise from first 2 s of signal
        init_len = min(int(2.0 * self.fs), len(mwi))
        spki = float(np.max(mwi[:init_len]))   # signal peak estimate
        npki = 0.0                              # noise peak estimate
        threshold1 = npki + 0.25 * (spki - npki)

        peaks = self._find_peaks(mwi, min_distance=self._refractory)
        r_peaks: list[int] = []

        for pk in peaks:
            if mwi[pk] >= threshold1:
                # QRS candidate – update signal peak
                spki = 0.125 * mwi[pk] + 0.875 * spki
                r_peaks.append(int(pk))
            else:
                # Noise peak – update noise peak
                npki = 0.125 * mwi[pk] + 0.875 * npki

            threshold1 = npki + 0.25 * (spki - npki)

        # Step 5: back-search in original signal within ±correction window
        r_peaks_arr = self._correct_peaks(ecg, np.array(r_peaks))
        return r_peaks_arr

    def _correct_peaks(
        self, ecg: np.ndarray, r_peaks: np.ndarray, search_ms: int = 50
    ) -> np.ndarray:
        """Shift each R-peak to the nearest local maximum in the raw ECG."""
        half = int(search_ms / 1000 * self.fs)
        corrected: list[int] = []
        for pk in r_peaks:
            lo = max(0, pk - half)
            hi = min(len(ecg), pk + half)
            local_max = lo + int(np.argmax(ecg[lo:hi]))
            corrected.append(local_max)
        return np.array(corrected, dtype=np.int64)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def detect_r_peaks(ecg: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Full pipeline: bandpass → notch → Pan-Tompkins detector.

    Returns sample indices of detected R-peaks.
    """
    filtered = preprocess(ecg, fs)
    detector = PanTompkinsDetector(fs=fs)
    return detector.detect(filtered)


# ── Beat normalisation ────────────────────────────────────────────────────────

def normalize_beats(beats: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each beat independently.

    Parameters
    ----------
    beats : np.ndarray, shape (B, L)

    Returns
    -------
    np.ndarray, shape (B, L)
    """
    mu  = beats.mean(axis=1, keepdims=True)
    std = beats.std(axis=1, keepdims=True) + 1e-8
    return ((beats - mu) / std).astype(np.float32)
