"""
Unit tests for model output shapes and basic sanity
=====================================================
Tests cover:
* build_cnn_lstm  – correct input/output shapes
* predict_proba_cnn_lstm – probability sums to 1
* build_gbt  – fits and predicts on tiny data
* predict_proba_gbt – shape and valid probabilities
* EnsembleClassifier.predict_proba – weighted combination
* EnsembleClassifier.predict_single – correct return types
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import BEAT_LEN, N_CLASSES
from src.features import N_HRV_FEATURES


# ── CNN-LSTM ──────────────────────────────────────────────────────────────────

class TestCNNLSTM:
    @pytest.fixture(scope="class")
    def model(self):
        from src.cnn_lstm_model import build_cnn_lstm
        return build_cnn_lstm()

    def test_model_summary_not_empty(self, model):
        """Model should have trainable parameters."""
        params = model.count_params()
        assert params > 0

    def test_input_shape(self, model):
        assert model.input_shape == (None, BEAT_LEN, 1)

    def test_output_shape(self, model):
        assert model.output_shape == (None, N_CLASSES)

    def test_predict_proba_shape(self, model):
        from src.cnn_lstm_model import predict_proba_cnn_lstm
        beats = np.random.default_rng(0).normal(0, 1, (16, BEAT_LEN)).astype(np.float32)
        proba = predict_proba_cnn_lstm(model, beats)
        assert proba.shape == (16, N_CLASSES)

    def test_predict_proba_sums_to_one(self, model):
        from src.cnn_lstm_model import predict_proba_cnn_lstm
        beats = np.random.default_rng(1).normal(0, 1, (8, BEAT_LEN)).astype(np.float32)
        proba = predict_proba_cnn_lstm(model, beats)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_non_negative(self, model):
        from src.cnn_lstm_model import predict_proba_cnn_lstm
        beats = np.random.default_rng(2).normal(0, 1, (8, BEAT_LEN)).astype(np.float32)
        proba = predict_proba_cnn_lstm(model, beats)
        assert np.all(proba >= 0)

    def test_single_beat_inference(self, model):
        from src.cnn_lstm_model import predict_proba_cnn_lstm
        beat  = np.random.default_rng(3).normal(0, 1, (1, BEAT_LEN)).astype(np.float32)
        proba = predict_proba_cnn_lstm(model, beat)
        assert proba.shape == (1, N_CLASSES)


# ── GBT model ─────────────────────────────────────────────────────────────────

class TestGBT:
    @pytest.fixture(scope="class")
    def fitted_gbt(self):
        """Fit a tiny GBT on random data (all 5 classes present)."""
        from src.gbt_model import build_gbt

        rng = np.random.default_rng(42)
        X   = rng.normal(0, 1, (200, N_HRV_FEATURES)).astype(np.float32)
        y   = rng.integers(0, N_CLASSES, 200)
        # Ensure all classes present
        for c in range(N_CLASSES):
            y[c] = c

        pipeline = build_gbt(n_estimators=50)
        pipeline.fit(X, y)
        return pipeline

    def test_predict_proba_shape(self, fitted_gbt):
        from src.gbt_model import predict_proba_gbt
        X = np.random.default_rng(5).normal(0, 1, (32, N_HRV_FEATURES)).astype(np.float32)
        proba = predict_proba_gbt(fitted_gbt, X)
        assert proba.shape == (32, N_CLASSES)

    def test_predict_proba_non_negative(self, fitted_gbt):
        from src.gbt_model import predict_proba_gbt
        X = np.random.default_rng(6).normal(0, 1, (16, N_HRV_FEATURES)).astype(np.float32)
        proba = predict_proba_gbt(fitted_gbt, X)
        assert np.all(proba >= 0)

    def test_predict_proba_rows_sum_to_one(self, fitted_gbt):
        from src.gbt_model import predict_proba_gbt
        X = np.random.default_rng(7).normal(0, 1, (16, N_HRV_FEATURES)).astype(np.float32)
        proba = predict_proba_gbt(fitted_gbt, X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_labels_in_range(self, fitted_gbt):
        X   = np.random.default_rng(8).normal(0, 1, (32, N_HRV_FEATURES)).astype(np.float32)
        preds = fitted_gbt.predict(X)
        assert np.all((preds >= 0) & (preds < N_CLASSES))


# ── Ensemble ──────────────────────────────────────────────────────────────────

class TestEnsemble:
    @pytest.fixture(scope="class")
    def ensemble(self):
        """Build a minimal ensemble from un-trained/lightly-trained sub-models."""
        from src.cnn_lstm_model import build_cnn_lstm
        from src.gbt_model import build_gbt
        from src.ensemble import EnsembleClassifier

        rng = np.random.default_rng(99)
        X_hrv = rng.normal(0, 1, (100, N_HRV_FEATURES)).astype(np.float32)
        y     = rng.integers(0, N_CLASSES, 100)
        for c in range(N_CLASSES):
            y[c] = c

        gbt = build_gbt(n_estimators=30)
        gbt.fit(X_hrv, y)

        cnn = build_cnn_lstm()   # untrained – still valid for shape tests

        return EnsembleClassifier(cnn_lstm=cnn, gbt=gbt, weights=[0.6, 0.4])

    def test_predict_proba_shape(self, ensemble):
        beats = np.random.default_rng(10).normal(0, 1, (8, BEAT_LEN)).astype(np.float32)
        X_hrv = np.random.default_rng(11).normal(0, 1, (8, N_HRV_FEATURES)).astype(np.float32)
        proba = ensemble.predict_proba(beats, X_hrv)
        assert proba.shape == (8, N_CLASSES)

    def test_predict_proba_non_negative(self, ensemble):
        beats = np.random.default_rng(12).normal(0, 1, (8, BEAT_LEN)).astype(np.float32)
        X_hrv = np.random.default_rng(13).normal(0, 1, (8, N_HRV_FEATURES)).astype(np.float32)
        proba = ensemble.predict_proba(beats, X_hrv)
        assert np.all(proba >= 0)

    def test_predict_returns_valid_classes(self, ensemble):
        beats = np.random.default_rng(14).normal(0, 1, (16, BEAT_LEN)).astype(np.float32)
        X_hrv = np.random.default_rng(15).normal(0, 1, (16, N_HRV_FEATURES)).astype(np.float32)
        preds = ensemble.predict(beats, X_hrv)
        assert preds.shape == (16,)
        assert np.all((preds >= 0) & (preds < N_CLASSES))

    def test_predict_single_return_types(self, ensemble):
        beat  = np.random.default_rng(16).normal(0, 1, BEAT_LEN).astype(np.float32)
        hrv   = np.random.default_rng(17).normal(0, 1, N_HRV_FEATURES).astype(np.float32)
        cls_idx, cls_name, proba = ensemble.predict_single(beat, hrv)
        assert isinstance(cls_idx, int)
        assert isinstance(cls_name, str)
        assert proba.shape == (N_CLASSES,)
        assert 0 <= cls_idx < N_CLASSES

    def test_weights_respected(self, ensemble):
        """Manually verify soft-vote: with weight [1, 0] result = CNN-LSTM output."""
        from src.cnn_lstm_model import predict_proba_cnn_lstm
        from src.ensemble import EnsembleClassifier

        rng   = np.random.default_rng(20)
        beats = rng.normal(0, 1, (4, BEAT_LEN)).astype(np.float32)
        X_hrv = rng.normal(0, 1, (4, N_HRV_FEATURES)).astype(np.float32)

        cnn_only_ens = EnsembleClassifier(
            cnn_lstm=ensemble.cnn_lstm,
            gbt=ensemble.gbt,
            weights=[1.0, 0.0],
        )
        p_ens = cnn_only_ens.predict_proba(beats, X_hrv)
        p_cnn = predict_proba_cnn_lstm(ensemble.cnn_lstm, beats)
        np.testing.assert_allclose(p_ens, p_cnn, atol=1e-5)
