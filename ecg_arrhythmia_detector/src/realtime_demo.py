"""
Real-Time ECG Streaming Demo
=============================
Animates an ECG waveform beat-by-beat from a MIT-BIH record and flashes a
colour-coded alert whenever a dangerous arrhythmia (Ventricular / Fusion) is
detected by the ensemble classifier.

Usage
-----
    python -m src.realtime_demo                  # uses DEMO_RECORD from config
    python -m src.realtime_demo --record 208     # specify a record
    python -m src.realtime_demo --no-model       # sine-wave mock (no models needed)

Controls
--------
    q  /  close window   → quit
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config import (
    BEAT_LEN,
    CLASS_NAMES,
    DANGER_CLASSES,
    DATA_DIR,
    DEMO_CHANNEL,
    DEMO_FPS,
    DEMO_RECORD,
    FS,
    N_CLASSES,
)

log = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
_CLASS_COLORS = {
    0: "#2ecc71",   # Normal        – green
    1: "#f39c12",   # Supra         – orange
    2: "#e74c3c",   # Ventricular   – red
    3: "#c0392b",   # Fusion        – dark red
    4: "#95a5a6",   # Unknown       – grey
}
_ALERT_COLOR   = "#e74c3c"
_NORMAL_COLOR  = "#2c3e50"
_BG_COLOR      = "#0d1117"
_ECG_COLOR     = "#00ff7f"


# ── Mock model (used when --no-model flag is set) ─────────────────────────────

class _MockEnsemble:
    """Returns uniform class probabilities – useful for UI testing."""

    def predict_single(
        self, beat: np.ndarray, hrv_feat: np.ndarray
    ) -> tuple[int, str, np.ndarray]:
        import random
        cls = random.choices(range(N_CLASSES), weights=[0.7, 0.1, 0.1, 0.05, 0.05])[0]
        proba = np.zeros(N_CLASSES, dtype=np.float32)
        proba[cls] = 1.0
        return cls, CLASS_NAMES[cls], proba


# ── Beat generator ────────────────────────────────────────────────────────────

def _generate_beats(record_id: str):
    """Yield (beat_waveform, aami_label) from a real MIT-BIH record."""
    from src.data_loader import stream_record
    from src.preprocessing import preprocess

    yield from stream_record(
        record_id,
        channel=DEMO_CHANNEL,
        data_dir=DATA_DIR,
        preprocessor=preprocess,
    )


def _generate_mock_beats():
    """Yield synthetic sinusoidal beats (no data download needed)."""
    t = np.linspace(0, 2 * np.pi, BEAT_LEN)
    rng = np.random.default_rng(0)
    while True:
        noise = rng.normal(0, 0.05, BEAT_LEN)
        beat  = np.sin(t) * 0.8 + noise
        yield beat.astype(np.float32), 0


# ── HRV feature stub for demo ─────────────────────────────────────────────────

def _hrv_stub(recent_rr: deque) -> np.ndarray:
    """Compute a minimal HRV feature vector from the last few RR intervals."""
    from src.features import extract_hrv_features
    rr = np.array(list(recent_rr), dtype=np.float64)
    return extract_hrv_features(rr, beat=None, fs=FS)


# ── Main animation ────────────────────────────────────────────────────────────

def run_demo(
    record_id: str = DEMO_RECORD,
    use_model:  bool = True,
    fps:        int  = DEMO_FPS,
) -> None:
    """Launch the animated ECG streaming window."""
    matplotlib.use("TkAgg")          # change to "Qt5Agg" / "MacOSX" if needed

    # Load model
    ensemble = None
    if use_model:
        try:
            from src.ensemble import EnsembleClassifier
            ensemble = EnsembleClassifier.load()
            log.info("Ensemble loaded successfully.")
        except Exception as exc:
            log.warning("Could not load ensemble: %s – using mock.", exc)
            ensemble = _MockEnsemble()
    else:
        ensemble = _MockEnsemble()

    # Beat source
    if use_model:
        try:
            beat_gen = _generate_beats(record_id)
        except Exception:
            beat_gen = _generate_mock_beats()
    else:
        beat_gen = _generate_mock_beats()

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor=_BG_COLOR)
    fig.canvas.manager.set_window_title("ECG Arrhythmia Detector – Real-Time Demo")

    # Top panel: scrolling ECG
    ax_ecg = fig.add_axes([0.05, 0.45, 0.90, 0.48], facecolor=_BG_COLOR)
    # Bottom panel: beat classification bar
    ax_bar = fig.add_axes([0.05, 0.08, 0.60, 0.28], facecolor=_BG_COLOR)
    # Bottom right: probability bars
    ax_prob = fig.add_axes([0.70, 0.08, 0.25, 0.28], facecolor=_BG_COLOR)

    # Axes styling
    for ax in [ax_ecg, ax_bar, ax_prob]:
        ax.set_facecolor(_BG_COLOR)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ECG scroll buffer: 5 s
    scroll_len = int(5 * FS)
    ecg_buffer = deque([0.0] * scroll_len, maxlen=scroll_len)
    time_axis  = np.arange(scroll_len) / FS - 5.0

    (line_ecg,) = ax_ecg.plot(time_axis, list(ecg_buffer), color=_ECG_COLOR, lw=1.0)
    ax_ecg.set_xlim(-5, 0)
    ax_ecg.set_ylim(-2.5, 2.5)
    ax_ecg.set_xlabel("Time (s)", color="white")
    ax_ecg.set_ylabel("Amplitude (mV)", color="white")
    ax_ecg.set_title("ECG – real-time stream", color="white", fontsize=11)
    ax_ecg.axvline(0, color="#555555", lw=0.5)

    # History strip (last 30 beats coloured by class)
    history_len = 30
    beat_history: deque[int] = deque([0] * history_len, maxlen=history_len)
    bar_colors_ref = [_CLASS_COLORS.get(c, "grey") for c in beat_history]
    bar_rects = ax_bar.bar(
        range(history_len), [1] * history_len,
        color=bar_colors_ref, edgecolor="none"
    )
    ax_bar.set_xlim(-0.5, history_len - 0.5)
    ax_bar.set_ylim(0, 1.2)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_title("Beat classification history (newest → right)", color="white", fontsize=9)

    # Legend for the history strip
    handles = [
        mpatches.Patch(color=_CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(N_CLASSES)
    ]
    ax_bar.legend(
        handles=handles, loc="upper left", fontsize=7,
        facecolor=_BG_COLOR, labelcolor="white", framealpha=0.7
    )

    # Probability bars
    prob_bars = ax_prob.bar(
        range(N_CLASSES),
        [1 / N_CLASSES] * N_CLASSES,
        color=[_CLASS_COLORS[i] for i in range(N_CLASSES)],
        edgecolor="none",
    )
    ax_prob.set_xlim(-0.5, N_CLASSES - 0.5)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_xticks(range(N_CLASSES))
    ax_prob.set_xticklabels(
        [n[:4] for n in CLASS_NAMES], color="white", fontsize=7
    )
    ax_prob.set_yticks([0, 0.5, 1.0])
    ax_prob.set_yticklabels(["0", "0.5", "1"], color="white", fontsize=7)
    ax_prob.set_title("Class probabilities", color="white", fontsize=9)

    # Alert text
    alert_text = ax_ecg.text(
        -4.9, 2.1, "", color=_ALERT_COLOR,
        fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a0000", alpha=0.8),
    )

    # Status text (top-right of ECG panel)
    status_text = ax_ecg.text(
        -0.1, 2.1, "", color="white", fontsize=10, ha="right"
    )

    # ── RR interval tracker ────────────────────────────────────────────────────
    recent_rr: deque[float] = deque(maxlen=20)
    recent_rr.append(0.75)   # seed with typical RR
    last_beat_time = time.monotonic()

    interval = 1.0 / fps
    running  = [True]

    def on_close(_evt):
        running[0] = False

    fig.canvas.mpl_connect("close_event", on_close)

    # ── Animation loop ─────────────────────────────────────────────────────────
    plt.ion()
    plt.show()

    for beat, true_label in beat_gen:
        if not running[0]:
            break

        t_now  = time.monotonic()
        rr_val = t_now - last_beat_time
        recent_rr.append(float(np.clip(rr_val, 0.3, 2.0)))
        last_beat_time = t_now

        # Append beat samples to scroll buffer
        ecg_buffer.extend(beat.tolist())

        # Classify
        hrv_feat = _hrv_stub(recent_rr)
        cls_idx, cls_name, proba = ensemble.predict_single(beat, hrv_feat)

        # Update ECG line
        line_ecg.set_ydata(list(ecg_buffer))

        # Update history strip
        beat_history.append(cls_idx)
        for rect, c in zip(bar_rects, beat_history):
            rect.set_color(_CLASS_COLORS.get(c, "grey"))

        # Update probability bars
        for rect, p in zip(prob_bars, proba):
            rect.set_height(float(p))

        # Alert flash
        if cls_idx in DANGER_CLASSES:
            alert_msg = f"⚠  {cls_name.upper()} ARRHYTHMIA DETECTED"
            alert_text.set_text(alert_msg)
            ax_ecg.set_facecolor("#200000")
        else:
            alert_text.set_text("")
            ax_ecg.set_facecolor(_BG_COLOR)

        # Status
        status_text.set_text(f"Class: {cls_name}  |  True: {CLASS_NAMES[true_label]}")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(interval)

    plt.ioff()
    plt.close(fig)
    log.info("Demo finished.")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time ECG arrhythmia demo")
    p.add_argument("--record",   default=DEMO_RECORD, help="MIT-BIH record ID")
    p.add_argument("--no-model", action="store_true",  help="Use mock model (no inference)")
    p.add_argument("--fps",      type=int, default=DEMO_FPS)
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    run_demo(
        record_id=args.record,
        use_model=not args.no_model,
        fps=args.fps,
    )
