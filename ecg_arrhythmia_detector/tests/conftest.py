"""
Shared pytest fixtures and configuration.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import BEAT_LEN, FS


@pytest.fixture(scope="session")
def sample_ecg() -> np.ndarray:
    """10-second synthetic ECG at 360 Hz."""
    t = np.arange(FS * 10) / FS
    rng = np.random.default_rng(0)
    return (
        0.8 * np.sin(2 * np.pi * 1.2 * t)
        + 0.05 * rng.normal(size=len(t))
    ).astype(np.float32)


@pytest.fixture(scope="session")
def sample_beats() -> np.ndarray:
    """50 synthetic normalised beat waveforms."""
    rng = np.random.default_rng(1)
    t   = np.linspace(0, 2 * np.pi, BEAT_LEN)
    beats = np.array(
        [np.sin(t) + rng.normal(0, 0.05, BEAT_LEN) for _ in range(50)],
        dtype=np.float32,
    )
    return beats


@pytest.fixture(scope="session")
def sample_rr() -> np.ndarray:
    """60 typical RR intervals around 1.0 s."""
    rng = np.random.default_rng(2)
    return (1.0 + rng.normal(0, 0.03, 60)).astype(np.float64)
