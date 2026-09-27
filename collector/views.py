# collector/views.py
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from .models import Participant, Session, Attempt, KeystrokeEvent


# =========================
# Paragraph Text
# =========================
PARAGRAPH_TEXT = (
    "Günlük hayatımızda hepimiz benzer ve basit şeyler yaşarız. "
    "Sabah uyanırız, kahvaltı yaparız, okula ya da işe gideriz. "
    "Akşam eve dönüp dinleniriz. "
    "Bu sade ve tanıdık anlar, aslında hayatımızın en büyük kısmını oluşturur."
)


def auth_paragraph_page(request):
    # صفحة Identification القديمة
    return render(request, "auth_paragraph.html")


def auth_paragraph_verify_page(request):
    # صفحة Verification الجديدة
    return render(request, "auth_paragraph_verify.html")


# =========================
# Project root + Active dirs
# =========================
PROJ = Path(__file__).resolve().parent.parent

ACTIVE_WORD_DIR = PROJ / "training" / "models" / "active"
ACTIVE_PARA_DIR = PROJ / "training" / "models" / "active_paragraph"

WORD_CSV = PROJ / "data" / "processed" / "processed_keystrokes.csv"
PARA_CSV = PROJ / "data" / "processed" / "processed_paragraph.csv"

RUNTIME_DIR = PROJ / "data" / "runtime" / "decisions"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

LOG_WORD = RUNTIME_DIR / "decisions_log_SVM_word.csv"
LOG_PARA = RUNTIME_DIR / "decisions_log_RF_paragraph.csv"

_CACHE = {}


# =========================
# Small helpers
# =========================
def _load_scaler(active_dir: Path):
    """
    Loads StandardScaler saved during training (scaler.pkl).
    If not found, returns None (no scaling will be applied).
    """
    p = active_dir / "scaler.pkl"
    if p.exists():
        return joblib.load(p)
    return None


def _expected_n_features(model):
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "n_features_in_"):
                return int(step.n_features_in_)
    return None


def _safe_int(v):
    s = str(v).strip()
    if s == "":
        raise ValueError("claimed_user is empty")
    return int(float(s))


def _append_log(path: Path, row: dict):
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def _load_thresholds_file(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))

    # A) wrapped format
    if isinstance(raw, dict) and isinstance(raw.get("thresholds"), dict):
        thr_map = raw.get("thresholds", {})
        eer_map = raw.get("eer", {}) if isinstance(raw.get("eer"), dict) else {}
        score_type = str(raw.get("score_type", "decision_function"))
        default_thr = float(raw.get("default_threshold", 0.0))
        return thr_map, eer_map, score_type, default_thr

    # B) simple format
    thr_map = {}
    for k, v in raw.items():
        if k in ("model_path", "score_type", "version", "policy", "eer", "default_threshold"):
            continue
        try:
            thr_map[str(k)] = float(v)
        except:
            pass

    eer_map = raw.get("eer", {}) if isinstance(raw.get("eer"), dict) else {}
    score_type = str(raw.get("score_type", "decision_function"))
    default_thr = float(raw.get("default_threshold", 0.0)) if "default_threshold" in raw else 0.0
    return thr_map, eer_map, score_type, default_thr


def _load_id_to_name_from_csv(csv_path: Path):
    mp = {}
    try:
        if not csv_path.exists():
            return mp
        tmp = pd.read_csv(csv_path)
        id_col = "subject_id" if "subject_id" in tmp.columns else None
        name_col = None
        for cand in ("subject", "participant"):
            if cand in tmp.columns:
                name_col = cand
                break
        if not id_col or not name_col:
            return mp
        tmp = tmp[[id_col, name_col]].dropna().drop_duplicates()
        for sid, name in tmp.values:
            try:
                mp[int(sid)] = str(name)
            except:
                pass
    except:
        pass
    return mp


def _load_label_encoder(active_dir: Path):
    candidates = [
        active_dir / "label_encoder.pkl",
        active_dir / "label_encoder_paragraph.pkl",
        active_dir / "label_encoder_word.pkl",
    ]
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p)
            except:
                pass

    for p in sorted(active_dir.glob("label_encoder*.pkl")):
        try:
            return joblib.load(p)
        except:
            pass
    return None


def _resolve_name(user_id: int, le=None, id_to_name=None):
    if le is not None:
        try:
            return str(le.inverse_transform([int(user_id)])[0])
        except:
            pass
    if id_to_name and int(user_id) in id_to_name:
        return str(id_to_name[int(user_id)])
    return str(user_id)


def _resolve_claimed_id(claimed_user_raw, le=None):
    # try numeric
    try:
        return _safe_int(claimed_user_raw)
    except:
        pass

    # try label encoder from name
    if le is not None:
        try:
            return int(le.transform([str(claimed_user_raw)])[0])
        except:
            pass

    raise ValueError(f"claimed_user not resolvable: {claimed_user_raw}")


def _pick_score_vector(model, x, score_type: str):
    classes = list(getattr(model, "classes_", []))

    if score_type == "predict_proba":
        if not hasattr(model, "predict_proba"):
            raise AttributeError("model has no predict_proba")
        proba = model.predict_proba(x)
        scores = np.asarray(proba)[0]
        return scores, classes, "predict_proba"

    # decision_function
    if not hasattr(model, "decision_function"):
        raise AttributeError("model has no decision_function")
    dec = np.asarray(model.decision_function(x))

    if dec.ndim == 1:
        s = float(dec[0])
        if len(classes) == 2:
            scores = np.array([-s, s], dtype=float)
        else:
            scores = np.array([s], dtype=float)
    else:
        scores = dec[0].astype(float)

    return scores, classes, "decision_function"


def _load_active_bundle(test_type: str, force_reload: bool = False):
    """
    test_type: "word" or "paragraph"
    Returns cached bundle:
      (active_dir, model, scaler, thr_map, eer_map, score_type, default_thr, le, id_to_name)
    """
    key = f"bundle:{test_type}"
    if force_reload:
        _CACHE.pop(key, None)

    if key in _CACHE:
        return _CACHE[key]

    active_dir = ACTIVE_WORD_DIR if test_type == "word" else ACTIVE_PARA_DIR

    model_path = active_dir / "model.pkl"
    thr_path = active_dir / "thresholds.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing thresholds: {thr_path}")

    model = joblib.load(model_path)
    scaler = _load_scaler(active_dir)

    thr_map, eer_map, score_type, default_thr = _load_thresholds_file(thr_path)

    le = _load_label_encoder(active_dir)
    id_to_name = _load_id_to_name_from_csv(PARA_CSV if test_type == "paragraph" else WORD_CSV)

    _CACHE[key] = (active_dir, model, scaler, thr_map, eer_map, score_type, default_thr, le, id_to_name)
    return _CACHE[key]


# =========================
# ✅ Unified API
# =========================
@csrf_exempt
def api_authenticate(request):
    try:
        if request.method != "POST":
            return JsonResponse({"error": "POST only"}, status=405)

        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception as e:
            return JsonResponse({"error": "bad_json", "message": str(e)}, status=400)

        attempt_id = payload.get("attempt_id", payload.get("attemptId"))
        task = str(payload.get("task", "identify")).strip().lower()  # identify / verify
        test_type = str(payload.get("test_type", payload.get("mode", "word"))).strip().lower()

        force_reload = bool(payload.get("reload") or payload.get("force_reload"))

        # ---------------------------------------------------------
        # Attempt-based mode (recommended)
        # ---------------------------------------------------------
        if attempt_id is not None:
            try:
                a = Attempt.objects.select_related("session__participant").get(id=int(attempt_id))
            except Exception as e:
                return JsonResponse({"error": "bad_attempt_id", "message": str(e)}, status=400)

            # infer test_type from session.word
            if a.session.word == "TR_PARAGRAPH_1":
                test_type = "paragraph"
            elif a.session.word == ".tie5Roanl":
                test_type = "word"
            else:
                test_type = "paragraph" if "paragraph" in test_type else "word"

            if test_type not in ("word", "paragraph"):
                test_type = "word"

            active_dir, model, scaler, thr_map, eer_map, score_type, default_thr, le, id_to_name = _load_active_bundle(
                test_type, force_reload=force_reload
            )

            # ensure features_json exists
            recompute = bool(payload.get("recompute") or payload.get("force_recompute"))
            fdict = a.features_json or {}
            if recompute or not fdict:
                ev = list(a.events.values("type", "t", "key", "code"))
                if not ev:
                    return JsonResponse({"error": "no_events_for_attempt"}, status=400)

                if a.session.word == ".tie5Roanl":
                    fdict = compute_word_features(ev, a.session.index, a.attempt_no)
                else:
                    fdict = compute_paragraph_features(ev, a)
                    fdict["sessionIndex"] = a.session.index
                    fdict["rep"] = a.attempt_no

                a.features_json = fdict
                a.save()

            # build x vector (✅ apply scaler ONCE)
            try:
                cols = _load_feature_columns(active_dir, test_type)
                x = _features_dict_to_vector(fdict, cols)
                if scaler is not None:
                    x = scaler.transform(x)
            except Exception as e:
                return JsonResponse({"error": "feature_vector_build_failed", "message": str(e)}, status=500)

            expected = _expected_n_features(model)
            if expected is not None and int(x.shape[1]) != int(expected):
                return JsonResponse({
                    "error": "feature_length_mismatch",
                    "message": f"Model expects {expected} features but got {x.shape[1]}",
                    "expected": int(expected),
                    "got": int(x.shape[1]),
                    "test_type": test_type,
                    "mode": "attempt_id",
                }, status=400)

            # score vector
            scores, classes, used_score_type = _pick_score_vector(model, x, score_type)
            if not classes:
                return JsonResponse({"error": "model_has_no_classes_"}, status=500)

            # ---------------------------
            # (A) Identification
            # ---------------------------
            if task != "verify":
                pred_pos = int(np.argmax(scores))
                pred_id = int(classes[pred_pos])
                score = float(scores[pred_pos])

                thr = float(thr_map.get(str(pred_id), default_thr))
                accepted = bool(score >= thr)

                pred_name = _resolve_name(pred_id, le=le, id_to_name=id_to_name)

                order = np.argsort(scores)[::-1][:3]
                top_k = []
                for i in order:
                    cid = int(classes[int(i)])
                    top_k.append({
                        "id": cid,
                        "name": _resolve_name(cid, le=le, id_to_name=id_to_name),
                        "score": float(scores[int(i)])
                    })

                result = {
                    "mode": "identification",
                    "test_type": test_type,
                    "score_type": used_score_type,
                    "attempt_id": int(attempt_id),
                    "pred_id": pred_id,
                    "pred_name": pred_name,
                    "score": score,
                    "threshold": thr,
                    "accepted": accepted,
                    "top_k": top_k,
                }

                try:
                    log_row = {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": "identification",
                        "test_type": test_type,
                        "attempt_id": int(attempt_id),
                        "pred_id": pred_id,
                        "pred_name": pred_name,
                        "score_type": used_score_type,
                        "score": score,
                        "threshold": thr,
                        "accepted": int(accepted),
                    }
                    _append_log(LOG_PARA if test_type == "paragraph" else LOG_WORD, log_row)
                except:
                    pass

                return JsonResponse({"ok": True, "result": result})

            # ---------------------------
            # (B) Verification (Robust)
            # ---------------------------
            claimed_user_raw = payload.get("claimed_user", payload.get("claimed", payload.get("claimed_user_id")))
            if claimed_user_raw is None:
                return JsonResponse({"error": "missing claimed_user for verify"}, status=400)

            try:
                claimed_id = _resolve_claimed_id(claimed_user_raw, le=le)
            except Exception as e:
                return JsonResponse({"error": "bad_claimed_user", "message": str(e), "got": str(claimed_user_raw)},
                                    status=400)

            if claimed_id not in classes:
                return JsonResponse({"error": f"claimed_id {claimed_id} not in model.classes_", "classes": classes},
                                    status=400)

            col = classes.index(claimed_id)
            score = float(scores[col])
            thr = float(thr_map.get(str(claimed_id), default_thr))

            # ✅ Robust params (يمكن تعديلها من الواجهة)
            safety_above = float(payload.get("safety_above", 0.03))
            conflict_delta = float(payload.get("conflict_delta", 0.05))
            hard_margin = float(payload.get("margin", 0.0))

            # best_other
            if len(scores) > 1:
                best_other = float(np.max([scores[i] for i in range(len(scores)) if i != col]))
            else:
                best_other = -1e9

            accepted = False
            reason = ""
            warning = ""

            if score < thr:
                accepted = False
                reason = "below_threshold"
            else:
                if (score - thr) >= safety_above:
                    accepted = True
                    reason = "strong_accept"
                else:
                    if (best_other - score) >= conflict_delta:
                        accepted = False
                        reason = "strong_conflict"
                    else:
                        if (score - best_other) >= hard_margin:
                            accepted = True
                            reason = "accept"
                        else:
                            accepted = True
                            reason = "accept_close"
                            warning = "close_competitor"

            claimed_name = _resolve_name(claimed_id, le=le, id_to_name=id_to_name)

            result = {
                "mode": "verification",
                "test_type": test_type,
                "score_type": used_score_type,
                "attempt_id": int(attempt_id),
                "claimed_id": claimed_id,
                "claimed_name": claimed_name,
                "score": score,
                "threshold": thr,
                "best_other": best_other,
                "margin": hard_margin,
                "safety_above": safety_above,
                "conflict_delta": conflict_delta,
                "accepted": accepted,
                "reason": reason,
                "warning": warning,
            }

            try:
                log_row = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "verification",
                    "test_type": test_type,
                    "attempt_id": int(attempt_id),
                    "claimed_id": claimed_id,
                    "claimed_name": claimed_name,
                    "score_type": used_score_type,
                    "score": score,
                    "threshold": thr,
                    "best_other": best_other,
                    "margin": hard_margin,
                    "safety_above": safety_above,
                    "conflict_delta": conflict_delta,
                    "accepted": int(accepted),
                    "reason": reason,
                    "warning": warning,
                }
                _append_log(LOG_PARA if test_type == "paragraph" else LOG_WORD, log_row)
            except:
                pass

            return JsonResponse({"ok": True, "result": result})

        # ---------------------------------------------------------
        # OLD MODE: direct features + claimed_user verification
        # ---------------------------------------------------------
        features = payload.get("features", payload.get("Features"))
        claimed_user_raw = payload.get("claimed_user", payload.get("claimed", payload.get("claimed_user_id")))

        if features is None or claimed_user_raw is None:
            return JsonResponse({"error": "missing features/claimed_user"}, status=400)

        if test_type not in ("word", "paragraph"):
            test_type = "word"

        # ✅ load bundle FIRST (so scaler exists)
        active_dir, model, scaler, thr_map, eer_map, score_type, default_thr, le, id_to_name = _load_active_bundle(
            test_type, force_reload=force_reload
        )

        try:
            x = np.asarray(features, dtype=float).reshape(1, -1)
            if scaler is not None:
                x = scaler.transform(x)
        except Exception as e:
            return JsonResponse({"error": "bad_features", "message": str(e)}, status=400)

        expected = _expected_n_features(model)
        if expected is not None and int(x.shape[1]) != int(expected):
            return JsonResponse({
                "error": "feature_length_mismatch",
                "message": f"Model expects {expected} features but got {x.shape[1]}",
                "expected": int(expected),
                "got": int(x.shape[1]),
                "test_type": test_type,
                "mode": "direct_features",
            }, status=400)

        try:
            claimed_id = _resolve_claimed_id(claimed_user_raw, le=le)
        except Exception as e:
            return JsonResponse({"error": "bad_claimed_user", "message": str(e), "got": str(claimed_user_raw)}, status=400)

        scores, classes, used_score_type = _pick_score_vector(model, x, score_type)
        if not classes:
            return JsonResponse({"error": "model_has_no_classes_"}, status=500)

        pred_idx = int(np.argmax(scores))
        pred_id = int(classes[pred_idx])

        if claimed_id not in classes:
            return JsonResponse({"error": f"claimed_id {claimed_id} not in model.classes_", "classes": classes}, status=400)

        col = classes.index(claimed_id)
        score = float(scores[col])

        thr = float(thr_map.get(str(claimed_id), default_thr))
        accepted = bool(score >= thr)

        claimed_name = _resolve_name(claimed_id, le=le, id_to_name=id_to_name)
        pred_name = _resolve_name(pred_id, le=le, id_to_name=id_to_name)

        eer_val = None
        if isinstance(eer_map, dict) and str(claimed_id) in eer_map:
            try:
                eer_val = float(eer_map.get(str(claimed_id)))
            except:
                eer_val = None

        result = {
            "mode": "verification",
            "test_type": test_type,
            "score_type": used_score_type,
            "claimed_id": claimed_id,
            "claimed_name": claimed_name,
            "pred_id": pred_id,
            "pred_name": pred_name,
            "score": score,
            "threshold": thr,
            "accepted": accepted,
            "eer": eer_val,
        }

        try:
            log_row = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "verification",
                "test_type": test_type,
                "claimed_id": claimed_id,
                "claimed_name": claimed_name,
                "pred_id": pred_id,
                "pred_name": pred_name,
                "score_type": used_score_type,
                "score": score,
                "threshold": thr,
                "accepted": int(accepted),
                "eer": eer_val if eer_val is not None else "",
            }
            _append_log(LOG_PARA if test_type == "paragraph" else LOG_WORD, log_row)
        except:
            pass

        return JsonResponse({"ok": True, "result": result})

    except Exception as e:
        return JsonResponse({"error": type(e).__name__, "message": str(e)}, status=500)


# =========================
# Helpers (features)
# =========================
def char_label(c: str) -> str:
    mapping = {
        " ": "Space",
        ".": "Dot",
        ",": "Comma",
        ";": "Semicolon",
        ":": "Colon",
        "!": "Exclamation",
        "?": "Question",
    }
    return mapping.get(c, c)


def _load_feature_columns(active_dir: Path, test_type: str):
    """
    Reads feature columns from:
      1) active_dir/meta.json -> feature_columns / feature_cols / columns
      2) processed CSV header
    """
    DROP = ("subject", "participant", "subject_id", "sessionIndex", "rep")

    meta_path = active_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cols = meta.get("feature_columns") or meta.get("feature_cols") or meta.get("columns")
            if isinstance(cols, list) and cols:
                return [c for c in cols if c not in DROP]
        except:
            pass

    csv_path = PARA_CSV if test_type == "paragraph" else WORD_CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=1)
        cols = [c for c in df.columns if c not in DROP]
        return cols

    raise FileNotFoundError("Cannot determine feature columns (no meta.json and no processed CSV).")


def _features_dict_to_vector(fdict: dict, columns: list):
    return np.asarray([float(fdict.get(c, 0.0)) for c in columns], dtype=float).reshape(1, -1)


# =========================
# Pages
# =========================
def home(request):
    return redirect("typing")


def typing_page(request):
    p, _ = Participant.objects.get_or_create(alias="Kullanici_01")
    s, _ = Session.objects.get_or_create(participant=p, word=".tie5Roanl", index=1)
    return render(request, "typing.html", {"word": s.word})


def typing_paragraph_page(request):
    return render(request, "typing_paragraph.html")


SESSION_TARGET = 50


# =========================
# Attempts
# =========================
@csrf_exempt
def start_attempt(request):
    alias = (request.GET.get("alias") or "").strip()
    mode = (request.GET.get("mode") or "word").lower()

    if not alias:
        return JsonResponse({"error": "alias required"}, status=400)

    p, _ = Participant.objects.get_or_create(alias=alias)

    if mode == "paragraph":
        target_word = "TR_PARAGRAPH_1"
        last = Session.objects.filter(participant=p, word=target_word).order_by("index").last()
        next_index = 1 if not last else last.index + 1
        s = Session.objects.create(participant=p, word=target_word, index=next_index)
        a = Attempt.objects.create(session=s, attempt_no=1)
        return JsonResponse({"id": a.id, "sessionIndex": s.index})

    target_word = ".tie5Roanl"
    last = Session.objects.filter(participant=p, word=target_word).order_by("index").last()

    if not last:
        next_index = 1
    else:
        completed = (
            Attempt.objects.filter(session=last)
            .exclude(features_json=None)
            .exclude(features_json={})
            .count()
        )
        next_index = last.index + 1 if completed >= SESSION_TARGET else last.index

    s, _ = Session.objects.get_or_create(participant=p, word=target_word, index=next_index)

    done_here = (
        Attempt.objects.filter(session=s)
        .exclude(features_json=None)
        .exclude(features_json={})
        .count()
    )

    a = Attempt.objects.create(session=s, attempt_no=done_here + 1)
    return JsonResponse({"id": a.id, "sessionIndex": s.index})


@csrf_exempt
def post_batch(request):
    data = json.loads(request.body)
    a = Attempt.objects.get(id=data["attempt_id"])

    objs = [
        KeystrokeEvent(
            attempt=a,
            type=e["type"],
            key=e["key"],
            code=e["code"],
            t=e["t"],
        )
        for e in data["events"]
    ]
    if objs:
        KeystrokeEvent.objects.bulk_create(objs, batch_size=1000)
    return JsonResponse({"ok": True}, status=201)


@csrf_exempt
def finish_attempt(request):
    data = json.loads(request.body)
    a = Attempt.objects.get(id=data["attempt_id"])

    a.started_at = float(data["started_at"])
    a.finished_at = float(data["finished_at"])
    a.total_time_ms = a.finished_at - a.started_at

    prev_done = (
        Attempt.objects.filter(session=a.session)
        .exclude(id=a.id)
        .exclude(features_json=None)
        .exclude(features_json={})
        .count()
    )
    rep_index = prev_done + 1
    a.attempt_no = rep_index

    ev = list(a.events.values("type", "t", "key", "code"))
    if not ev:
        a.features_json = {"sessionIndex": a.session.index, "rep": rep_index}
        a.save()
        return JsonResponse({"ok": True})

    if a.session.word == ".tie5Roanl":
        features = compute_word_features(ev, a.session.index, rep_index)
    elif a.session.word == "TR_PARAGRAPH_1":
        features = compute_paragraph_features(ev, a)
        features["sessionIndex"] = a.session.index
        features["rep"] = rep_index
    else:
        features = {"sessionIndex": a.session.index, "rep": rep_index, "total_time_ms": a.total_time_ms}

    a.features_json = features
    a.save()
    return JsonResponse({"ok": True})


# =========================
# Feature Extractors
# =========================
def compute_word_features(events, session_index: int, rep_index: int):
    df = pd.DataFrame(events).sort_values("t").reset_index(drop=True)

    def canon_key(k: str) -> str:
        if k == "Enter":
            return "Return"
        if k == ".":
            return "period"
        if k == "5":
            return "five"
        return k

    shift_pressed = False
    r_label_stack = []
    norm_events = []

    for row in df.itertuples():
        k_raw = row.key or ""
        code = row.code or ""
        t = float(row.t)
        typ = row.type

        is_shift = (k_raw == "Shift") or (code in ("ShiftLeft", "ShiftRight"))
        if is_shift:
            if typ == "down":
                shift_pressed = True
            elif typ == "up":
                shift_pressed = False
            norm_events.append({"type": typ, "t": t, "key": "Shift", "code": code})
            continue

        k = canon_key(k_raw)
        is_r_key = (code == "KeyR") or (k_raw in ("r", "R"))
        if is_r_key:
            if typ == "down":
                label = "Shift.r" if (shift_pressed or k_raw == "R") else "r"
                r_label_stack.append(label)
                k = label
            elif typ == "up":
                label = r_label_stack.pop(0) if r_label_stack else (
                    "Shift.r" if (shift_pressed or k_raw == "R") else "r"
                )
                k = label

        norm_events.append({"type": typ, "t": t, "key": k, "code": code})

    nd = pd.DataFrame(norm_events)

    seq = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]
    holds_map = {k: [] for k in seq}
    last_down = {}

    for row in nd.itertuples():
        k = row.key
        if row.type == "down":
            last_down.setdefault(k, []).append(row.t)
        elif row.type == "up":
            if k in last_down and last_down[k]:
                t0 = last_down[k].pop(0)
                if k in holds_map:
                    holds_map[k].append(row.t - t0)

    H = {f"H.{k}": float(np.mean(holds_map[k])) if holds_map[k] else 0.0 for k in seq}

    first_down, first_up = {}, {}
    for row in nd.itertuples():
        if row.type == "down" and row.key in seq and row.key not in first_down:
            first_down[row.key] = row.t
        if row.type == "up" and row.key in seq and row.key not in first_up:
            first_up[row.key] = row.t

    DD, UD = {}, {}
    for i in range(len(seq) - 1):
        a_key, b_key = seq[i], seq[i + 1]
        a_d, b_d = first_down.get(a_key), first_down.get(b_key)
        a_u = first_up.get(a_key)
        DD[f"DD.{a_key}.{b_key}"] = float(b_d - a_d) if (a_d is not None and b_d is not None) else 0.0
        UD[f"UD.{a_key}.{b_key}"] = float(b_d - a_u) if (a_u is not None and b_d is not None) else 0.0

    features = {"sessionIndex": session_index, "rep": rep_index}
    features.update(H)
    features.update(DD)
    features.update(UD)
    return features


def compute_paragraph_features(events, attempt: Attempt):
    ev_sorted = sorted(events, key=lambda e: float(e["t"]))

    pending_down = {}
    char_events = []
    backspaces_total = 0
    error_to_bs = []
    bs_to_correct = []
    waiting_bs_up_time = None

    def is_printable_key(k: str):
        if not k:
            return False
        ignore = {"Shift", "Control", "Alt", "Meta", "AltGraph"}
        if k in ignore:
            return False
        if k in ["Backspace", "Enter"]:
            return False
        return True

    for e in ev_sorted:
        k = e["key"]
        t = float(e["t"])
        typ = e["type"]

        if k == "Backspace":
            if typ == "up":
                backspaces_total += 1
                idx = None
                for i in range(len(char_events) - 1, -1, -1):
                    if char_events[i]["alive"]:
                        idx = i
                        break
                if idx is not None:
                    ce = char_events[idx]
                    ce["alive"] = False
                    if ce["up"] is not None:
                        error_to_bs.append(t - ce["up"])
                waiting_bs_up_time = t
            continue

        if k == "Enter":
            continue

        if typ == "down" and is_printable_key(k):
            pending_down.setdefault(k, []).append(t)
            if waiting_bs_up_time is not None:
                bs_to_correct.append(t - waiting_bs_up_time)
                waiting_bs_up_time = None

        elif typ == "up" and is_printable_key(k):
            downs = pending_down.get(k) or []
            t0 = downs.pop(0) if downs else t
            ch = k
            char_events.append({"char": ch, "down": t0, "up": t, "alive": True})

    final_char_events = [ce for ce in char_events if ce["alive"]]
    if not final_char_events:
        return {"total_time_ms": attempt.total_time_ms, "backspaces_total": backspaces_total}

    final_text = "".join(ce["char"] for ce in final_char_events)

    H_map, DD_map, UD_map = {}, {}, {}
    for ce in final_char_events:
        lbl = char_label(ce["char"])
        dur = ce["up"] - ce["down"]
        H_map.setdefault(lbl, []).append(dur)

    for i in range(len(final_char_events) - 1):
        ce1 = final_char_events[i]
        ce2 = final_char_events[i + 1]
        lbl1 = char_label(ce1["char"])
        lbl2 = char_label(ce2["char"])
        pair = f"{lbl1}.{lbl2}"
        dd = ce2["down"] - ce1["down"]
        ud = ce2["down"] - ce1["up"]
        DD_map.setdefault(pair, []).append(dd)
        UD_map.setdefault(pair, []).append(ud)

    features = {}
    for lbl, arr in H_map.items():
        features[f"H.{lbl}"] = float(np.mean(arr)) if arr else 0.0
    for pair, arr in DD_map.items():
        features[f"DD.{pair}"] = float(np.mean(arr)) if arr else 0.0
    for pair, arr in UD_map.items():
        features[f"UD.{pair}"] = float(np.mean(arr)) if arr else 0.0

    n = len(final_char_events)
    seg_len = max(1, n // 3)

    def segment_speed(seg):
        if not seg:
            return 0.0
        chars = len(seg)
        dur_ms = seg[-1]["up"] - seg[0]["down"]
        if dur_ms <= 0:
            return 0.0
        return chars / (dur_ms / 1000.0)

    seg1 = final_char_events[0:seg_len]
    seg2 = final_char_events[seg_len:2 * seg_len]
    seg3 = final_char_events[2 * seg_len:]

    seg1_speed = segment_speed(seg1)
    seg2_speed = segment_speed(seg2)
    seg3_speed = segment_speed(seg3)

    features["seg1_chars_per_sec"] = seg1_speed
    features["seg2_chars_per_sec"] = seg2_speed
    features["seg3_chars_per_sec"] = seg3_speed
    features["delta_speed_1_3"] = seg1_speed - seg3_speed

    features["backspaces_total"] = backspaces_total
    features["avg_error_to_backspace_ms"] = float(np.mean(error_to_bs)) if error_to_bs else 0.0
    features["avg_backspace_to_correct_ms"] = float(np.mean(bs_to_correct)) if bs_to_correct else 0.0

    total_time_ms = attempt.total_time_ms
    duration_min = total_time_ms / 60000.0 if total_time_ms > 0 else 0.0
    target_text = PARAGRAPH_TEXT
    target_len = len(target_text)

    features["total_time_ms"] = total_time_ms
    features["duration_min"] = duration_min
    features["target_chars"] = target_len
    features["typed_chars"] = len(final_text)
    features["chars_per_sec"] = (len(final_text) / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    target_words = len(target_text.split())
    features["target_words"] = target_words
    features["wpm"] = (target_words / duration_min) if duration_min > 0 else 0.0

    min_len = min(len(final_text), target_len)
    mismatches = 0
    missing_caps = 0
    missing_punct = 0

    for i in range(min_len):
        tc = target_text[i]
        wc = final_text[i]
        if tc != wc:
            mismatches += 1
        if tc.isalpha() and tc.isupper():
            if wc != tc:
                missing_caps += 1
        if tc in ".,;:!?":
            if wc != tc:
                missing_punct += 1

    mismatches += abs(len(final_text) - target_len)

    features["char_mismatches"] = mismatches
    features["error_rate"] = (mismatches / target_len) if target_len > 0 else 0.0
    features["missing_caps"] = missing_caps
    features["missing_punct"] = missing_punct

    return features


# =========================
# Exporters (unchanged)
# =========================
@require_GET
def export_aggregated(request):
    seq = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]

    base_cols = ["subject", "sessionIndex", "rep"]
    columns = base_cols[:]
    for i, k in enumerate(seq):
        columns.append(f"H.{k}")
        if i < len(seq) - 1:
            nxt = seq[i + 1]
            columns.append(f"DD.{k}.{nxt}")
            columns.append(f"UD.{k}.{nxt}")

    raw_dir = settings.BASE_DIR / "data" / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    out_path = raw_dir / "aggregated_tie5Roanl.csv"

    atts = (
        Attempt.objects.select_related("session__participant")
        .filter(session__word=".tie5Roanl")
        .order_by("id")
    )

    rows = []
    for a in atts:
        alias = str(a.session.participant.alias)
        fjson = a.features_json or {}

        row = {c: 0.0 for c in columns}
        row["subject"] = alias
        row["sessionIndex"] = fjson.get("sessionIndex", a.session.index)
        row["rep"] = fjson.get("rep", a.attempt_no)
        for k, v in fjson.items():
            if k in row:
                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["subject"] = df["subject"].astype(str)
        df["sessionIndex"] = pd.to_numeric(df["sessionIndex"], errors="coerce").fillna(0).astype(int)
        df["rep"] = pd.to_numeric(df["rep"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(["subject", "sessionIndex", "rep"], kind="mergesort").reset_index(drop=True)
        df = df.reindex(columns=columns)

    df.to_csv(out_path, index=False)
    return JsonResponse({"saved": str(out_path), "rows": int(len(df)), "columns": list(df.columns)})


@require_GET
def export_aggregated_paragraph(request):
    raw_dir = settings.BASE_DIR / "data" / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    out_path = raw_dir / "aggregated_paragraph.csv"

    atts = (
        Attempt.objects
        .select_related("session__participant")
        .filter(session__word="TR_PARAGRAPH_1")
        .order_by("id")
    )

    rows = []
    all_cols = set(["subject", "sessionIndex", "rep"])

    for a in atts:
        alias = str(a.session.participant.alias)
        fjson = a.features_json or {}

        row = {
            "subject": alias,
            "sessionIndex": fjson.get("sessionIndex", a.session.index),
            "rep": fjson.get("rep", a.attempt_no),
        }

        for k, v in fjson.items():
            if k not in ("sessionIndex", "rep"):
                row[k] = v
                all_cols.add(k)

        rows.append(row)

    if not rows:
        df = pd.DataFrame(columns=sorted(all_cols))
        df.to_csv(out_path, index=False)
        return JsonResponse({"saved": str(out_path), "rows": 0, "columns": list(df.columns)})

    base_cols = ["subject", "sessionIndex", "rep"]

    seq_labels = []
    for ch in PARAGRAPH_TEXT:
        lbl = char_label(ch)
        if lbl not in seq_labels:
            seq_labels.append(lbl)

    ordered_cols = list(base_cols)
    feature_cols = set(all_cols) - set(base_cols)

    for lbl in seq_labels:
        h_col = f"H.{lbl}"
        if h_col in feature_cols:
            ordered_cols.append(h_col)
            feature_cols.discard(h_col)

        dd_prefix = f"DD.{lbl}."
        dd_cols = sorted([c for c in feature_cols if c.startswith(dd_prefix)])
        for c in dd_cols:
            ordered_cols.append(c)
            feature_cols.discard(c)

        ud_prefix = f"UD.{lbl}."
        ud_cols = sorted([c for c in feature_cols if c.startswith(ud_prefix)])
        for c in ud_cols:
            ordered_cols.append(c)
            feature_cols.discard(c)

    ordered_cols.extend(sorted(feature_cols))

    df = pd.DataFrame(rows).reindex(columns=ordered_cols)
    df["subject"] = df["subject"].astype(str)
    df["sessionIndex"] = pd.to_numeric(df["sessionIndex"], errors="coerce").fillna(0).astype(int)
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["subject", "sessionIndex", "rep"], kind="mergesort").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    return JsonResponse({"saved": str(out_path), "rows": int(len(df)), "columns": list(df.columns)})


def export_processed(request):
    rows = []
    for a in Attempt.objects.all():
        f = a.features_json or {}
        rows.append({
            "participant": a.session.participant.alias,
            "session_idx": a.session.index,
            "attempt_no": a.attempt_no,
            "hold_mean": f.get("hold_mean", 0.0),
            "dd_mean": f.get("dd_mean", 0.0),
            "total_time_ms": a.total_time_ms,
        })
    df = pd.DataFrame(rows)
    path = settings.BASE_DIR / "data" / "processed" / "processed_keystrokes.csv"
    os.makedirs(path.parent, exist_ok=True)
    df.to_csv(path, index=False)
    return JsonResponse({"saved": str(path), "rows": len(df)})
