from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parents[1]
DATA_DIR     = BASE_DIR / "data"
RUNTIME_DIR  = DATA_DIR / "runtime"
SESSIONS_DIR = RUNTIME_DIR / "sessions"
DECISIONS_DIR= RUNTIME_DIR / "decisions"

MODELS_DIR   = BASE_DIR / "training" / "models"
ACTIVE_DIR   = MODELS_DIR / "active"

MODEL_PATH   = ACTIVE_DIR / "model.pkl"
SCALER_PATH  = ACTIVE_DIR / "scaler.pkl"
ENC_PATH     = ACTIVE_DIR / "label_encoder.pkl"
THR_PATH     = ACTIVE_DIR / "thresholds.json"

for p in [SESSIONS_DIR, DECISIONS_DIR, ACTIVE_DIR]:
    p.mkdir(parents=True, exist_ok=True)
