"""
Central configuration for the ECG Arrhythmia Detector.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── MIT-BIH records to download ───────────────────────────────────────────────
# Full set of 48 two-channel recordings; subset used for fast dev runs.
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
MITBIH_DB = "mitdb"

# ── Signal processing ─────────────────────────────────────────────────────────
FS               = 360          # MIT-BIH native sampling rate (Hz)
BANDPASS_LOW     = 0.5          # Hz
BANDPASS_HIGH    = 40.0         # Hz
BANDPASS_ORDER   = 4
NOTCH_FREQ       = 60.0         # Hz  (power-line interference)
NOTCH_Q          = 30.0         # quality factor

# ── Beat segmentation ─────────────────────────────────────────────────────────
BEAT_WINDOW_MS   = 600          # total window around each R-peak (ms)
BEAT_PRE_MS      = 200          # samples before R-peak
BEAT_LEN         = int(FS * BEAT_WINDOW_MS / 1000)   # samples per beat

# ── AAMI class mapping ────────────────────────────────────────────────────────
# Maps MIT-BIH annotation symbols → 5 AAMI classes
AAMI_MAP = {
    # Normal
    "N": 0, ".": 0, "n": 0, "Q": 4,
    # Supraventricular
    "A": 1, "a": 1, "J": 1, "S": 1, "e": 1, "j": 1,
    # Ventricular
    "V": 2, "E": 2,
    # Fusion
    "F": 3,
}
CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
N_CLASSES   = len(CLASS_NAMES)

# ── HRV feature extraction ────────────────────────────────────────────────────
HRV_LF_BAND = (0.04, 0.15)   # Hz
HRV_HF_BAND = (0.15, 0.40)   # Hz
PNN50_THRESH = 0.050          # 50 ms in seconds

# ── CNN-LSTM model ────────────────────────────────────────────────────────────
CNN_FILTERS      = [32, 64, 128]
CNN_KERNEL        = 5
LSTM_UNITS        = 64
DROPOUT_RATE      = 0.3
BATCH_SIZE        = 64
EPOCHS            = 30
LEARNING_RATE     = 1e-3

# ── GBT model ─────────────────────────────────────────────────────────────────
GBT_N_ESTIMATORS  = 300
GBT_MAX_DEPTH     = 5
GBT_LEARNING_RATE = 0.05

# ── Ensemble ──────────────────────────────────────────────────────────────────
# Soft-vote weights: [cnn_lstm, gbt]
ENSEMBLE_WEIGHTS = [0.6, 0.4]

# ── Real-time demo ────────────────────────────────────────────────────────────
DEMO_RECORD      = "208"       # record known to contain many arrhythmias
DEMO_CHANNEL     = 0
DEMO_FPS         = 60          # animation frame rate
DANGER_CLASSES   = {2, 3}      # Ventricular + Fusion → alert
