# ECG Arrhythmia Detector

A production-quality Python pipeline that downloads real ECG recordings from
PhysioNet, filters and segments them, extracts HRV features, trains a
CNN-LSTM + Gradient-Boosted-Trees ensemble, and streams a live animated demo.

---

## Project Structure

```
ecg_arrhythmia_detector/
├── src/
│   ├── config.py           # All hyper-parameters and paths
│   ├── data_loader.py      # MIT-BIH download, record loading, beat segmentation
│   ├── preprocessing.py    # Butterworth bandpass, IIR notch, Pan-Tompkins QRS
│   ├── features.py         # HRV feature extraction (time + frequency domain)
│   ├── cnn_lstm_model.py   # CNN-LSTM Keras model (builder, train, infer)
│   ├── gbt_model.py        # HistGradientBoosting sklearn pipeline
│   ├── ensemble.py         # Soft-vote ensemble + evaluation helpers
│   ├── train.py            # Full training script (CLI)
│   └── realtime_demo.py    # Animated ECG streaming demo (CLI)
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
├── data/                   # Downloaded MIT-BIH records (auto-created)
├── models/                 # Saved model weights (auto-created)
├── requirements.txt
├── setup.py
└── pytest.ini
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU support**: install `tensorflow[and-cuda]` instead of plain `tensorflow`
> for NVIDIA GPU acceleration.

### 2. Train the ensemble

```bash
# Full 48-record training (downloads ~100 MB from PhysioNet first time)
python -m src.train

# Fast smoke-test on 3 records, 5 epochs
python -m src.train --records 100 108 200 --epochs 5
```

**What happens:**

| Step | Details |
|------|---------|
| Download | `wfdb.dl_database` pulls `.hea/.dat/.atr` files from PhysioNet |
| Preprocessing | Butterworth 4th-order bandpass (0.5–40 Hz) → IIR notch (60 Hz) |
| QRS detection | Pan-Tompkins algorithm locates R-peaks |
| Segmentation | ±600 ms windows centred 200 ms before each R-peak |
| HRV features | RR mean/std, RMSSD, pNN50, LF/HF (Lomb-Scargle), QRS dur/amp |
| Split | 80/20 stratified train/test |
| CNN-LSTM | Conv1D×3 → BatchNorm → LSTM(64) → Dense(5·softmax) |
| GBT | `HistGradientBoostingClassifier` on 9 HRV features |
| Ensemble | Soft-vote: 0.6 × CNN-LSTM + 0.4 × GBT |
| Evaluation | Per-class precision/recall/F1 + confusion matrix |

### 3. Run the real-time demo

```bash
# Uses trained models + MIT-BIH record 208 (many arrhythmias)
python -m src.realtime_demo

# Different record
python -m src.realtime_demo --record 119

# Mock mode (no models or data required)
python -m src.realtime_demo --no-model
```

The demo shows:
- A scrolling green ECG trace (5 s window)
- A beat-history colour strip (last 30 beats)
- A real-time class-probability bar chart
- A flashing red alert when a **Ventricular** or **Fusion** arrhythmia is detected

### 4. Run the test suite

```bash
pytest
```

---

## AAMI Arrhythmia Classes

| Index | Class | MIT-BIH symbols |
|-------|-------|----------------|
| 0 | Normal | N, ., n |
| 1 | Supraventricular | A, a, J, S, e, j |
| 2 | Ventricular | V, E |
| 3 | Fusion | F |
| 4 | Unknown | Q (and unmapped) |

---

## Signal Processing Pipeline

```
Raw ECG (360 Hz)
     │
     ▼
Butterworth bandpass (0.5 – 40 Hz, order 4, zero-phase)
     │
     ▼
IIR notch filter (60 Hz, Q=30, zero-phase)
     │
     ▼
Pan-Tompkins QRS detector
  ├─ Derivative filter (5-point)
  ├─ Squaring
  ├─ Moving-window integration (150 ms)
  ├─ Adaptive dual threshold
  └─ Refractory period (200 ms) + peak correction
     │
     ▼
Beat segmentation  (-200 ms … +400 ms around each R-peak)
     │
     ├──► Z-score normalisation → CNN-LSTM
     └──► HRV feature extraction → GBT
```

---

## Model Architecture

### CNN-LSTM

```
Input (216, 1)
│
├─ Conv1D(32, k=5, same) → BN → ReLU → MaxPool(2)   → (108, 32)
├─ Conv1D(64, k=5, same) → BN → ReLU → MaxPool(2)   → ( 54, 64)
├─ Conv1D(128, k=5, same) → BN → ReLU → MaxPool(2)  → ( 27, 128)
│
├─ LSTM(64)
├─ Dropout(0.3)
│
└─ Dense(5, softmax)
```

Training: Adam(lr=1e-3), sparse cross-entropy, early stopping (patience 5),
ReduceLROnPlateau, balanced class weights.

### GBT

`StandardScaler` → `HistGradientBoostingClassifier(max_iter=300, max_depth=5, lr=0.05)`

Fitted with per-sample balanced class weights.

### Ensemble

```
P_ensemble = 0.6 × P_CNN-LSTM + 0.4 × P_GBT
ŷ = argmax(P_ensemble)
```

---

## HRV Features (9-dimensional vector)

| # | Feature | Description |
|---|---------|-------------|
| 0 | rr_mean | Mean RR interval (s) |
| 1 | rr_std | SDNN – std of RR intervals |
| 2 | rmssd | Root mean square of successive differences |
| 3 | pnn50 | Proportion of diffs > 50 ms |
| 4 | lf_power | LF band power 0.04–0.15 Hz (Lomb-Scargle) |
| 5 | hf_power | HF band power 0.15–0.40 Hz |
| 6 | lf_hf_ratio | LF/HF autonomic balance ratio |
| 7 | qrs_duration | Width of QRS complex (s) |
| 8 | qrs_amplitude | Peak-to-trough QRS amplitude (mV) |

---

## Configuration

All parameters live in [src/config.py](src/config.py). Key knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MITBIH_RECORDS` | all 48 | Records used for training |
| `FS` | 360 Hz | Sampling rate |
| `BEAT_WINDOW_MS` | 600 ms | Beat window length |
| `EPOCHS` | 30 | CNN-LSTM training epochs |
| `ENSEMBLE_WEIGHTS` | [0.6, 0.4] | CNN-LSTM / GBT vote weights |
| `DEMO_RECORD` | "208" | Record streamed in the demo |
| `DANGER_CLASSES` | {2, 3} | Classes that trigger the alert |

---

## Requirements

- Python ≥ 3.9
- TensorFlow ≥ 2.13
- scikit-learn ≥ 1.3
- scipy ≥ 1.11
- wfdb ≥ 4.1
- matplotlib ≥ 3.7
- numpy, pandas, tqdm, joblib

---

## Data Source

MIT-BIH Arrhythmia Database — Moody GB, Mark RG (2001).
*The impact of the MIT-BIH Arrhythmia Database.*
IEEE Eng Med Biol 20(3):45-50.
[PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
