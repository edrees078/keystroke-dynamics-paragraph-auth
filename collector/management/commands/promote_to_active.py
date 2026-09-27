# collector/management/commands/promote_to_active.py

import json
import shutil
import time
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from collector.constants import ACTIVE_DIR  # عادة يشير لـ training/models/active


# -------------------------
# Helpers
# -------------------------
def _find_first(exp: Path, patterns, recursive=True):
    """Return first matching file (latest mtime preferred)."""
    hits = []
    for pat in patterns:
        if recursive:
            hits.extend(list(exp.rglob(pat)))
        else:
            hits.extend(list(exp.glob(pat)))

    hits = [p for p in hits if p.is_file()]
    if not hits:
        return None

    # prefer newest
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def _copy_if_found(src: Path, dst: Path, stdout, label):
    if src and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        stdout.write(f"[OK] Copied {label}: {src.name} -> {dst}")
        return True
    return False


def _load_thresholds_any(path: Path):
    """
    Supports:
    - wrapped format: {"thresholds": {...}, "eer": {...}, "score_type": "..."}
    - simple format: {"0":0.12, "1":0.08, "model_path":"...", "score_type":"..."}
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    # wrapped
    if isinstance(raw, dict) and isinstance(raw.get("thresholds"), dict):
        thr = {str(k): float(v) for k, v in raw["thresholds"].items()}
        eer_raw = raw.get("eer", {})
        eer = {str(k): float(v) for k, v in eer_raw.items()} if isinstance(eer_raw, dict) else {}
        score_type = str(raw.get("score_type", "decision_function"))
        return thr, eer, score_type

    # simple
    thr = {}
    for k, v in raw.items():
        if k in ("model_path", "score_type", "version", "policy", "eer", "default_threshold"):
            continue
        try:
            thr[str(k)] = float(v)
        except:
            pass

    eer = {}
    score_type = str(raw.get("score_type", "decision_function"))
    return thr, eer, score_type


def _build_thresholds_from_eer_csv(eer_csv: Path):
    """
    Robustly parse eer_summary.csv with columns like:
    user, thr, EER
    OR user, threshold, eer
    """
    df = pd.read_csv(eer_csv)
    cols = {c.lower().strip(): c for c in df.columns}

    # locate columns
    user_col = cols.get("user") or cols.get("subject_id") or cols.get("id")
    thr_col = cols.get("thr") or cols.get("threshold") or cols.get("thresh")
    eer_col = cols.get("eer")

    if not user_col or not thr_col:
        # can't build thresholds
        return {}, {}

    thr_map = {}
    eer_map = {}

    for _, r in df.iterrows():
        try:
            u = str(int(float(r[user_col])))
        except:
            continue

        try:
            thr_map[u] = float(r[thr_col])
        except:
            pass

        if eer_col:
            try:
                eer_map[u] = float(r[eer_col])
            except:
                pass

    return thr_map, eer_map


# -------------------------
# Command
# -------------------------
class Command(BaseCommand):
    help = "Promote an experiment folder into training/models/<dest> (isolated per test type)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--exp",
            type=str,
            required=True,
            help="Path to experiment folder (contains model + reports + thresholds)",
        )
        parser.add_argument(
            "--score-type",
            type=str,
            default="decision_function",
            choices=["decision_function", "predict_proba"],
            dest="score_type",
        )
        parser.add_argument(
            "--dest",
            type=str,
            default="active",
            help="Destination folder name under training/models (active or active_paragraph)",
        )
        parser.add_argument("--name", type=str, default=None)
        parser.add_argument(
            "--force-clean",
            action="store_true",
            help="Delete known old files in dest before copying",
        )

    def handle(self, *args, **opts):
        exp = Path(opts["exp"]).resolve()
        if not exp.exists():
            return self.stderr.write(f"[ERR] Experiment path not found: {exp}")

        # base models dir is parent of ACTIVE_DIR (ACTIVE_DIR = .../training/models/active)
        models_dir = ACTIVE_DIR.parent
        dest_dir = (models_dir / opts["dest"]).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(f"[INFO] Promoting: {exp}")
        self.stdout.write(f"[INFO] Destination: {dest_dir}")

        if opts["force_clean"]:
            for f in [
                "model.pkl",
                "scaler.pkl",
                "scaler_paragraph.pkl",
                "label_encoder.pkl",
                "label_encoder_paragraph.pkl",
                "confusion_matrix.png",
                "confusion_matrix.csv",
                "classification_report.csv",
                "meta.json",
                "thresholds.json",
            ]:
                p = dest_dir / f
                if p.exists():
                    p.unlink()
            self.stdout.write("[OK] Cleaned old files in destination")

        is_paragraph = "paragraph" in opts["dest"].lower()

        # 1) Find & copy model
        model_src = _find_first(
            exp,
            patterns=[
                "model.pkl",
                "model.joblib",
                "*model*.pkl",
                "*best*.pkl",
                "*.joblib",
                "*.pkl",
            ],
            recursive=True,
        )
        if model_src:
            _copy_if_found(model_src, dest_dir / "model.pkl", self.stdout, "model")
        else:
            self.stdout.write("[SKIP] model not found (no suitable .pkl/.joblib)")

        # 2) Find & copy scaler / label_encoder
        # For paragraph you told me they may be named scaler_paragraph.pkl, label_encoder_paragraph.pkl
        scaler_patterns = (
            ["scaler_paragraph.pkl", "*scaler*paragraph*.pkl", "*scaler*.pkl"]
            if is_paragraph
            else ["scaler.pkl", "*scaler*.pkl"]
        )
        le_patterns = (
            ["label_encoder_paragraph.pkl", "label_encoder*.pkl"]
            if is_paragraph
            else ["label_encoder.pkl", "label_encoder*.pkl"]
        )

        scaler_src = _find_first(exp, scaler_patterns, recursive=True)
        le_src = _find_first(exp, le_patterns, recursive=True)

        if scaler_src:
            out_name = "scaler_paragraph.pkl" if (is_paragraph and "paragraph" in scaler_src.name.lower()) else "scaler.pkl"
            _copy_if_found(scaler_src, dest_dir / out_name, self.stdout, "scaler")
        else:
            self.stdout.write("[SKIP] scaler not found")

        if le_src:
            out_name = "label_encoder_paragraph.pkl" if (is_paragraph and "paragraph" in le_src.name.lower()) else "label_encoder.pkl"
            _copy_if_found(le_src, dest_dir / out_name, self.stdout, "label_encoder")
        else:
            self.stdout.write("[SKIP] label_encoder not found")

        # 3) Find & copy confusion matrix / report
        cm_src = _find_first(exp, ["confusion_matrix.png", "*confusion*matrix*.png"], recursive=True)
        if cm_src:
            _copy_if_found(cm_src, dest_dir / "confusion_matrix.png", self.stdout, "confusion_matrix.png")
        else:
            self.stdout.write("[SKIP] confusion_matrix not found")

        cm_csv_src = _find_first(exp, ["confusion_matrix.csv", "*confusion*matrix*.csv"], recursive=True)
        if cm_csv_src:
            _copy_if_found(cm_csv_src, dest_dir / "confusion_matrix.csv", self.stdout, "confusion_matrix.csv")

        rep_src = _find_first(exp, ["classification_report.csv", "*classification*report*.csv"], recursive=True)
        if rep_src:
            _copy_if_found(rep_src, dest_dir / "classification_report.csv", self.stdout, "classification_report.csv")
        else:
            self.stdout.write("[SKIP] classification_report.csv not found")

        # 4) Build thresholds.json + EER
        thr_map = {}
        eer_map = {}
        thr_source_msg = None

        eer_csv = _find_first(exp, ["eer_summary.csv"], recursive=True)
        if eer_csv and eer_csv.exists():
            thr_map, eer_map = _build_thresholds_from_eer_csv(eer_csv)
            thr_source_msg = f"eer_summary.csv ({eer_csv.name})"
        else:
            # fallback: thresholds.json anywhere inside exp
            thr_json_src = _find_first(exp, ["thresholds.json", "*thresholds*.json"], recursive=True)
            if thr_json_src and thr_json_src.exists():
                thr_map, eer_map, _st = _load_thresholds_any(thr_json_src)
                thr_source_msg = f"thresholds json ({thr_json_src.name})"
            else:
                thr_source_msg = "No eer_summary.csv or thresholds.json found; thresholds/eer empty"

        self.stdout.write(f"[INFO] Thresholds source: {thr_source_msg}")

        thr_out = {
            "version": 1,
            "model_path": str((dest_dir / "model.pkl").resolve()),
            "score_type": opts["score_type"],
            "policy": "per_user",
            "thresholds": thr_map,
            "eer": eer_map,
            "default_threshold": 0.0,
        }

        (dest_dir / "thresholds.json").write_text(
            json.dumps(thr_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.stdout.write("[OK] Wrote thresholds.json")

        # 5) Write meta.json (merge any meta found in exp)
        meta = {
            "exp_name": opts["name"] or exp.name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target_col": "subject_id",
            "source_exp": str(exp),
            "dest": str(dest_dir),
            "score_type": opts["score_type"],
            "kind": "paragraph" if is_paragraph else "word",
        }

        meta_src = _find_first(exp, ["meta.json", "*meta*.json"], recursive=False)  # prefer root
        if meta_src and meta_src.exists():
            try:
                meta_loaded = json.loads(meta_src.read_text(encoding="utf-8"))
                if isinstance(meta_loaded, dict):
                    # do not overwrite our keys unless needed
                    for k, v in meta_loaded.items():
                        meta.setdefault(k, v)
            except:
                pass

        (dest_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.stdout.write("[OK] Wrote meta.json")

        self.stdout.write(self.style.SUCCESS(f"[DONE] Promoted {exp.name} -> {dest_dir}"))
