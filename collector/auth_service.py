# collector/auth_service.py
import json
from pathlib import Path

import numpy as np
import joblib

# مشروعك
PROJ = Path(__file__).resolve().parent.parent

ACTIVE_WORD_DIR = PROJ / "training" / "models" / "active"

MODEL_PATH  = ACTIVE_WORD_DIR / "model.pkl"
SCALER_PATH = ACTIVE_WORD_DIR / "scaler.pkl"
LE_PATH     = ACTIVE_WORD_DIR / "label_encoder.pkl"
THR_PATH    = ACTIVE_WORD_DIR / "thresholds.json"

_CACHE = {}


def _load_thresholds_file(path: Path):
    """
    يدعم شكلين:
    1) Wrapped:
       {"thresholds": {...}, "eer": {...}, "score_type": "...", "default_threshold": 0.0}
    2) Simple:
       {"0":0.12, "1":0.08, "model_path":"...", "score_type":"decision_function"}
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    # wrapped
    if isinstance(raw, dict) and isinstance(raw.get("thresholds"), dict):
        thr_map = {str(k): float(v) for k, v in raw["thresholds"].items()}
        eer_raw = raw.get("eer", {})
        eer_map = {str(k): float(v) for k, v in eer_raw.items()} if isinstance(eer_raw, dict) else {}
        score_type = str(raw.get("score_type", "decision_function"))
        default_thr = float(raw.get("default_threshold", 0.0))
        return thr_map, eer_map, score_type, default_thr

    # simple
    thr_map = {}
    for k, v in raw.items():
        if k in ("model_path", "score_type", "version", "policy", "eer", "default_threshold"):
            continue
        try:
            thr_map[str(k)] = float(v)
        except:
            pass

    eer_map = {}
    score_type = str(raw.get("score_type", "decision_function"))
    default_thr = float(raw.get("default_threshold", 0.0))
    return thr_map, eer_map, score_type, default_thr


def _load_active_word_bundle():
    """
    Lazy-load: لا ينهار السيرفر إذا الملفات ناقصة.
    """
    if "word_bundle" in _CACHE:
        return _CACHE["word_bundle"]

    # لو الموديل غير موجود، نخزن None ونرجع
    if not MODEL_PATH.exists():
        _CACHE["word_bundle"] = (None, None, None, {}, {}, "decision_function", 0.0)
        return _CACHE["word_bundle"]

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    le = joblib.load(LE_PATH) if LE_PATH.exists() else None

    thr_map, eer_map, score_type, default_thr = ({}, {}, "decision_function", 0.0)
    if THR_PATH.exists():
        try:
            thr_map, eer_map, score_type, default_thr = _load_thresholds_file(THR_PATH)
        except:
            thr_map, eer_map, score_type, default_thr = ({}, {}, "decision_function", 0.0)

    _CACHE["word_bundle"] = (model, scaler, le, thr_map, eer_map, score_type, default_thr)
    return _CACHE["word_bundle"]


def _resolve_name(label, le=None):
    """
    يرجّع اسم المستخدم إذا label_encoder موجود، وإلا يرجّع label نفسه كنص.
    """
    if le is not None:
        try:
            return str(le.inverse_transform([label])[0])
        except:
            pass
    return str(label)


def score_and_decide(X, claimed_user):
    """
    X: shape (1, n_features) numpy array
    claimed_user: عادة نص (مثلاً "5") لأن مفاتيح thresholds تكون نصوص
    """
    model, scaler, le, thr_map, eer_map, score_type, default_thr = _load_active_word_bundle()

    # ✅ أهم نقطة: لا نكسر السيرفر إذا الموديل ناقص
    if model is None:
        return {
            "mode": "verification",
            "score_type": score_type,
            "test_type": "word",
            "claimed_id": claimed_user,
            "claimed_name": str(claimed_user),
            "pred_id": None,
            "pred_name": None,
            "score": 0.0,
            "threshold": float(default_thr),
            "accepted": False,
            "eer": None,
            "error": f"ACTIVE word model missing: {MODEL_PATH}"
        }

    # تطبيق scaler إن وُجد
    x = X
    if scaler is not None:
        try:
            x = scaler.transform(X)
        except:
            x = X

    classes = list(getattr(model, "classes_", []))

    # حساب scores حسب score_type
    if score_type == "predict_proba":
        if not hasattr(model, "predict_proba"):
            return {
                "mode": "verification",
                "score_type": score_type,
                "test_type": "word",
                "claimed_id": claimed_user,
                "claimed_name": str(claimed_user),
                "pred_id": None,
                "pred_name": None,
                "score": 0.0,
                "threshold": float(default_thr),
                "accepted": False,
                "eer": None,
                "error": "model_has_no_predict_proba"
            }
        vec = model.predict_proba(x)[0]

    else:
        if not hasattr(model, "decision_function"):
            return {
                "mode": "verification",
                "score_type": score_type,
                "test_type": "word",
                "claimed_id": claimed_user,
                "claimed_name": str(claimed_user),
                "pred_id": None,
                "pred_name": None,
                "score": 0.0,
                "threshold": float(default_thr),
                "accepted": False,
                "eer": None,
                "error": "model_has_no_decision_function"
            }

        dfun = model.decision_function(x)

        # ممكن يكون (n_classes,) أو (1,n_classes)
        if isinstance(dfun, np.ndarray) and dfun.ndim == 2:
            vec = dfun[0]
        else:
            vec = np.array(dfun).reshape(-1)

    vec = np.array(vec).reshape(-1)

    # predicted class
    pred_idx = int(np.argmax(vec))
    pred_label = classes[pred_idx] if classes else pred_idx

    # score للـ claimed_user (لو موجود ضمن classes)
    claimed_key = str(claimed_user).strip()

    score = float(vec[pred_idx])  # fallback
    if classes:
        # حاول تطابق claimed_user مع classes (int أو str)
        claimed_candidates = []
        claimed_candidates.append(claimed_user)
        try:
            claimed_candidates.append(int(float(claimed_user)))
        except:
            pass
        claimed_candidates.append(claimed_key)

        col = None
        for c in claimed_candidates:
            if c in classes:
                col = classes.index(c)
                break

        if col is not None:
            score = float(vec[col])

    # threshold per user
    thr = float(thr_map.get(claimed_key, default_thr))
    accepted = bool(score >= thr)

    result = {
        "mode": "verification",
        "score_type": score_type,
        "test_type": "word",
        "claimed_id": claimed_user,
        "claimed_name": str(claimed_user),
        "pred_id": pred_label,
        "pred_name": _resolve_name(pred_label, le=le),
        "score": score,
        "threshold": thr,
        "accepted": accepted,
        "eer": float(eer_map.get(claimed_key, 0.0)) if isinstance(eer_map, dict) else None,
    }
    return result
