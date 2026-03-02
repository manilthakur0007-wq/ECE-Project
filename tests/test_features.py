"""
Unit tests for src/features.py
================================
Tests cover:
* compute_rr_intervals    – conversion correctness
* rmssd / pnn50           – known-value checks
* lomb_scargle_psd        – output shape and non-negativity
* extract_hrv_features    – vector length and no-NaN
* qrs_features            – amplitude and duration bounds
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import BEAT_LEN, FS
from src.features import (
    N_HRV_FEATURES,
    compute_rr_intervals,
    extract_hrv_features,
    lomb_scargle_psd,
    pnn50,
    qrs_features,
    rmssd,
    rr_mean,
    rr_std,
)


# ── compute_rr_intervals ──────────────────────────────────────────────────────

class TestComputeRRIntervals:
    def test_basic(self):
        # R-peaks at 360, 720, 1080 → RR = 1.0 s each
        r_samples = np.array([0, 360, 720, 1080])
        rr = compute_rr_intervals(r_samples, fs=360)
        np.testing.assert_allclose(rr, [1.0, 1.0, 1.0])

    def test_single_peak_returns_empty(self):
        rr = compute_rr_intervals(np.array([100]), fs=360)
        assert len(rr) == 0

    def test_empty_returns_empty(self):
        rr = compute_rr_intervals(np.array([]), fs=360)
        assert len(rr) == 0

    def test_dtype(self):
        r_samples = np.array([0, 360, 720])
        rr = compute_rr_intervals(r_samples, fs=360)
        assert rr.dtype == np.float64


# ── Time-domain statistics ────────────────────────────────────────────────────

class TestTimeDomainHRV:
    @pytest.fixture
    def typical_rr(self):
        rng = np.random.default_rng(42)
        # 60 RR intervals around 1.0 s with ~30 ms variability
        return 1.0 + rng.normal(0, 0.03, 60)

    def test_rr_mean_close_to_one(self, typical_rr):
        assert abs(rr_mean(typical_rr) - 1.0) < 0.05

    def test_rr_std_positive(self, typical_rr):
        assert rr_std(typical_rr) > 0

    def test_rmssd_non_negative(self, typical_rr):
        assert rmssd(typical_rr) >= 0.0

    def test_pnn50_range(self, typical_rr):
        val = pnn50(typical_rr)
        assert 0.0 <= val <= 1.0

    def test_rmssd_known_value(self):
        # Alternating 0.9 and 1.1 → successive diffs all 0.2 → RMSSD = 0.2
        rr = np.tile([0.9, 1.1], 20)
        assert abs(rmssd(rr) - 0.2) < 1e-6

    def test_pnn50_all_large_diffs(self):
        # Alternating 0.7 and 1.3 → |diff| = 0.6 > 50 ms → pNN50 = 1.0
        rr = np.tile([0.7, 1.3], 20)
        assert pnn50(rr) == pytest.approx(1.0)

    def test_pnn50_no_large_diffs(self):
        # Constant RR → no diffs > 50 ms → pNN50 = 0.0
        rr = np.ones(40) * 0.8
        assert pnn50(rr) == pytest.approx(0.0)

    def test_rmssd_empty(self):
        assert rmssd(np.array([])) == 0.0

    def test_pnn50_empty(self):
        assert pnn50(np.array([])) == 0.0


# ── Lomb-Scargle PSD ─────────────────────────────────────────────────────────

class TestLombScargePSD:
    @pytest.fixture
    def typical_rr(self):
        rng = np.random.default_rng(7)
        return 1.0 + rng.normal(0, 0.05, 100)

    def test_output_shapes(self, typical_rr):
        freqs, power = lomb_scargle_psd(typical_rr, n_freqs=500)
        assert freqs.shape == (500,)
        assert power.shape == (500,)

    def test_non_negative_power(self, typical_rr):
        _, power = lomb_scargle_psd(typical_rr)
        assert np.all(power >= 0)

    def test_short_rr_returns_zeros(self):
        rr = np.array([1.0, 1.0])
        freqs, power = lomb_scargle_psd(rr)
        assert np.all(power == 0)

    def test_freq_range(self, typical_rr):
        freqs, _ = lomb_scargle_psd(typical_rr)
        assert freqs[0] > 0
        assert freqs[-1] <= 0.5


# ── extract_hrv_features ──────────────────────────────────────────────────────

class TestExtractHRVFeatures:
    @pytest.fixture
    def rr(self):
        return 1.0 + np.random.default_rng(0).normal(0, 0.03, 50)

    @pytest.fixture
    def beat(self):
        rng = np.random.default_rng(1)
        t   = np.linspace(0, 2 * np.pi, BEAT_LEN)
        return (np.sin(t) + rng.normal(0, 0.02, BEAT_LEN)).astype(np.float32)

    def test_vector_length(self, rr, beat):
        fv = extract_hrv_features(rr, beat=beat)
        assert len(fv) == N_HRV_FEATURES

    def test_no_nan(self, rr, beat):
        fv = extract_hrv_features(rr, beat=beat)
        assert not np.any(np.isnan(fv))

    def test_no_nan_without_beat(self, rr):
        fv = extract_hrv_features(rr, beat=None)
        assert not np.any(np.isnan(fv))

    def test_dtype_float32(self, rr, beat):
        fv = extract_hrv_features(rr, beat=beat)
        assert fv.dtype == np.float32

    def test_pnn50_index_in_range(self, rr, beat):
        fv = extract_hrv_features(rr, beat=beat)
        # pNN50 is at index 3
        assert 0.0 <= float(fv[3]) <= 1.0

    def test_qrs_duration_positive(self, rr, beat):
        fv = extract_hrv_features(rr, beat=beat)
        assert float(fv[7]) > 0.0   # qrs_duration


# ── qrs_features ─────────────────────────────────────────────────────────────

class TestQRSFeatures:
    def test_amplitude_non_negative(self):
        beat = np.random.default_rng(9).normal(0, 1, BEAT_LEN).astype(np.float32)
        feats = qrs_features(beat, fs=FS)
        assert feats["qrs_amplitude"] >= 0.0

    def test_duration_reasonable(self):
        beat = np.zeros(BEAT_LEN, dtype=np.float32)
        feats = qrs_features(beat, fs=FS)
        # ±75 ms window → 150 ms duration
        assert abs(feats["qrs_duration"] - 0.150) < 0.01

    def test_known_amplitude(self):
        beat = np.zeros(BEAT_LEN, dtype=np.float32)
        from src.config import BEAT_PRE_MS
        r_idx = int(FS * BEAT_PRE_MS / 1000)
        beat[r_idx]     =  1.5
        beat[r_idx + 5] = -0.5
        feats = qrs_features(beat, fs=FS)
        # max=1.5, min=-0.5 → amplitude=2.0
        assert abs(feats["qrs_amplitude"] - 2.0) < 0.01
