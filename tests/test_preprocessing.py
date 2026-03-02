"""
Unit tests for src/preprocessing.py
=====================================
Tests cover:
* bandpass_filter  – shape preservation, frequency attenuation
* notch_filter     – 60 Hz suppression
* PanTompkinsDetector – R-peak detection on a synthetic signal
* preprocess       – end-to-end pipeline
* normalize_beats  – zero-mean, unit-variance per beat
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from src.preprocessing import (
    PanTompkinsDetector,
    bandpass_filter,
    normalize_beats,
    notch_filter,
    preprocess,
)

FS   = 360
DURATION = 10       # seconds
N    = FS * DURATION


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ecg(fs: int = FS, duration: int = DURATION) -> np.ndarray:
    """Return a realistic-ish synthetic ECG: sinusoid + baseline + noise."""
    t = np.arange(fs * duration) / fs
    signal = (
        0.8 * np.sin(2 * np.pi * 1.2 * t)          # ~72 bpm fundamental
        + 0.3 * np.sin(2 * np.pi * 0.1 * t)         # slow baseline wander
        + 0.1 * np.random.default_rng(0).normal(size=len(t))
    )
    return signal.astype(np.float32)


def _make_sinusoid(freq: float, fs: int = FS, duration: int = DURATION) -> np.ndarray:
    t = np.arange(fs * duration) / fs
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_qrs_train(fs: int = FS, duration: int = DURATION, hr_bpm: int = 72) -> np.ndarray:
    """Synthetic ECG with sharp Gaussian QRS complexes at regular intervals."""
    n  = fs * duration
    rr = int(60.0 / hr_bpm * fs)
    sig = np.zeros(n, dtype=np.float32)
    for r in range(rr, n - rr, rr):
        # QRS spike
        sig[r] = 1.5
        if r + 1 < n: sig[r + 1] = 0.5
        if r - 1 >= 0: sig[r - 1] = 0.3
    # Add small noise
    sig += np.random.default_rng(1).normal(0, 0.02, n).astype(np.float32)
    return sig


# ── Bandpass filter tests ─────────────────────────────────────────────────────

class TestBandpassFilter:
    def test_output_shape(self):
        ecg = _make_ecg()
        out = bandpass_filter(ecg)
        assert out.shape == ecg.shape

    def test_output_dtype(self):
        ecg = _make_ecg()
        out = bandpass_filter(ecg)
        assert out.dtype == np.float32

    def test_attenuates_dc(self):
        """DC component (0 Hz) must be attenuated below passband."""
        dc = np.ones(N, dtype=np.float32) * 2.0
        out = bandpass_filter(dc)
        assert np.abs(out).mean() < 0.1

    def test_attenuates_high_frequency(self):
        """80 Hz sinusoid is above the 40 Hz cutoff – must be strongly attenuated."""
        hf_sig = _make_sinusoid(80.0)
        out = bandpass_filter(hf_sig)
        assert float(np.std(out)) < 0.05

    def test_passes_midband(self):
        """10 Hz sinusoid is within passband – amplitude should be preserved."""
        mid_sig = _make_sinusoid(10.0)
        out = bandpass_filter(mid_sig)
        # After filter, std should be close to original (≥ 0.5)
        assert float(np.std(out)) > 0.5

    def test_no_nan(self):
        ecg = _make_ecg()
        out = bandpass_filter(ecg)
        assert not np.any(np.isnan(out))


# ── Notch filter tests ────────────────────────────────────────────────────────

class TestNotchFilter:
    def test_output_shape(self):
        ecg = _make_ecg()
        out = notch_filter(ecg)
        assert out.shape == ecg.shape

    def test_suppresses_60hz(self):
        """60 Hz sinusoid should be heavily attenuated after the notch filter."""
        sig60 = _make_sinusoid(60.0)
        out   = notch_filter(sig60)
        # Power at 60 Hz should drop significantly
        power_before = float(np.var(sig60))
        power_after  = float(np.var(out))
        assert power_after < 0.01 * power_before

    def test_preserves_10hz(self):
        """10 Hz content (well outside the notch) should be mostly preserved."""
        sig10 = _make_sinusoid(10.0)
        out   = notch_filter(sig10)
        assert float(np.std(out)) > 0.5

    def test_no_nan(self):
        ecg = _make_ecg()
        out = notch_filter(ecg)
        assert not np.any(np.isnan(out))


# ── Pan-Tompkins detector tests ───────────────────────────────────────────────

class TestPanTompkinsDetector:
    def test_detects_peaks_in_synthetic_ecg(self):
        """Detector must find at least 80 % of ground-truth R-peaks."""
        hr_bpm  = 72
        rr_samp = int(60.0 / hr_bpm * FS)
        qrs     = _make_qrs_train(FS, DURATION, hr_bpm)
        filtered = bandpass_filter(qrs)

        detector = PanTompkinsDetector(fs=FS)
        detected = detector.detect(filtered)

        # Expected peaks (starting after first RR)
        expected = np.arange(rr_samp, N - rr_samp, rr_samp)

        # Match detected peaks to expected with ±25-sample tolerance
        tolerance = 25
        matched = 0
        for exp_pk in expected:
            if len(detected) > 0 and np.min(np.abs(detected - exp_pk)) <= tolerance:
                matched += 1

        recall = matched / len(expected)
        assert recall >= 0.80, f"Recall {recall:.2f} < 0.80"

    def test_no_false_detections_on_silence(self):
        """Near-zero signal should produce zero or very few detections."""
        silence = np.random.default_rng(2).normal(0, 0.001, N).astype(np.float32)
        detector = PanTompkinsDetector(fs=FS)
        peaks    = detector.detect(silence)
        # Max 1 false positive per second on silence
        assert len(peaks) <= DURATION

    def test_returns_sorted_peaks(self):
        qrs      = _make_qrs_train()
        detector = PanTompkinsDetector(fs=FS)
        peaks    = detector.detect(qrs)
        assert list(peaks) == sorted(peaks)

    def test_refractory_period(self):
        """No two consecutive detections should be closer than ~200 ms."""
        qrs      = _make_qrs_train()
        detector = PanTompkinsDetector(fs=FS)
        peaks    = detector.detect(qrs)
        if len(peaks) > 1:
            gaps = np.diff(peaks)
            min_gap = int(0.200 * FS) - 5   # small tolerance
            assert np.all(gaps >= min_gap), f"Min gap {gaps.min()} < {min_gap}"


# ── preprocess pipeline ───────────────────────────────────────────────────────

class TestPreprocess:
    def test_shape_preserved(self):
        ecg = _make_ecg()
        out = preprocess(ecg)
        assert out.shape == ecg.shape

    def test_attenuates_60hz_and_dc(self):
        dc_plus_60 = np.ones(N, dtype=np.float32) + _make_sinusoid(60.0)
        out = preprocess(dc_plus_60)
        # Both DC and 60 Hz should be small
        assert float(np.std(out)) < 0.2


# ── normalize_beats ───────────────────────────────────────────────────────────

class TestNormalizeBeats:
    def test_zero_mean(self):
        beats = np.random.default_rng(3).normal(5.0, 2.0, (100, 216)).astype(np.float32)
        normed = normalize_beats(beats)
        means = normed.mean(axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_std(self):
        beats = np.random.default_rng(4).normal(0.0, 3.0, (100, 216)).astype(np.float32)
        normed = normalize_beats(beats)
        stds = normed.std(axis=1)
        np.testing.assert_allclose(stds, 1.0, atol=1e-4)

    def test_shape_unchanged(self):
        beats = np.ones((50, 216), dtype=np.float32)
        normed = normalize_beats(beats)
        assert normed.shape == beats.shape

    def test_constant_beat_no_nan(self):
        """Constant beat (std = 0) must not produce NaN after normalisation."""
        beats = np.ones((5, 216), dtype=np.float32) * 3.0
        normed = normalize_beats(beats)
        assert not np.any(np.isnan(normed))
