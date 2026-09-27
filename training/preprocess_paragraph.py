# training/preprocess_paragraph.py
import os
import json
import joblib
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------- إعدادات بسيطة للفقرة --------------------
MAD_THRESH = 4.5        # حد MAD للشذوذ
MIN_ROWS_FOR_MAD = 2    # نطبق MAD فقط إذا كان للمستخدم صفّان أو أكثر
# --------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE     = RAW_DIR  / "aggregated_paragraph.csv"
OUT_FILE     = PROC_DIR / "processed_paragraph.csv"
DROP_FILE    = PROC_DIR / "drop_report_paragraph.csv"
SCALER_FILE  = PROC_DIR / "scaler_paragraph.pkl"
ENCODER_FILE = PROC_DIR / "label_encoder_paragraph.pkl"
META_FILE    = PROC_DIR / "preprocess_meta_paragraph.json"


def _safe_unlink(path: Path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# دعم خيار --clean لمسح الملفات القديمة
parser = argparse.ArgumentParser()
parser.add_argument(
    "--clean",
    action="store_true",
    help="احذف نواتج المعالجة السابقة ثم أعد التشغيل"
)
args, _ = parser.parse_known_args()
if args.clean:
    for p in [OUT_FILE, DROP_FILE, SCALER_FILE, ENCODER_FILE, META_FILE]:
        _safe_unlink(p)


def mad_outlier_mask(X: np.ndarray, thresh: float = MAD_THRESH) -> np.ndarray:
    """
    يُرجِع قناعًا منطقيًا (True = صف شاذ) بناءً على MAD لكل عمود.
    نعتبر الصف شاذًا إذا تجاوز أي عمود العتبة.
    """
    med = np.median(X, axis=0)
    abs_dev = np.abs(X - med)
    mad = np.median(abs_dev, axis=0)
    mad[mad == 0] = 1.0  # لتجنب القسمة على صفر
    z_mad = 0.6745 * abs_dev / mad
    return (z_mad > thresh).any(axis=1)


def main():
    # 1) قراءة RAW
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"لم يتم العثور على الملف الخام: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    # تأكد من الأعمدة الأساسية
    for col in ["subject", "sessionIndex", "rep"]:
        if col not in df.columns:
            raise RuntimeError(f"العمود الإجباري '{col}' غير موجود في {RAW_FILE}")

    # ضبط الأنواع والفرز (للتنظيم فقط)
    df["subject"] = df["subject"].astype(str)
    df["sessionIndex"] = pd.to_numeric(df["sessionIndex"], errors="coerce").fillna(0).astype(int)
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["subject", "sessionIndex", "rep"],
                        kind="mergesort").reset_index(drop=True)

    drops = []

    def log_drop(row, reason):
        drops.append({
            "subject": row.get("subject", ""),
            "sessionIndex": int(row.get("sessionIndex", 0)),
            "rep": int(row.get("rep", 0)),
            "reason": reason
        })

    # 2) اختيار أعمدة الميزات:
    # نأخذ كل الأعمدة العددية ما عدا المعرفات (subject, sessionIndex, rep)
    id_cols = ["subject", "sessionIndex", "rep"]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in id_cols]

    if not feat_cols:
        raise RuntimeError("لم يتم العثور على أعمدة ميزات رقمية في ملف الفقرة.")

    # 3) معالجة الـ NaN:
    # نحذف فقط الصفوف التي كل الميزات فيها NaN تماماً
    all_nan_mask = df[feat_cols].isna().all(axis=1)
    df.loc[all_nan_mask].apply(lambda r: log_drop(r, "all features NaN"), axis=1)
    df = df[~all_nan_mask]

    # أي NaN متبقي نعتبره 0.0 (الميزة غير موجودة / لم تُحسب)
    df[feat_cols] = df[feat_cols].fillna(0.0)

    # 4) إسقاط صفوف مجموع ميزاتها = 0 (كلها 0، غير مفيد للتعلم)
    zero_mask = (df[feat_cols].sum(axis=1) == 0)
    df.loc[zero_mask].apply(lambda r: log_drop(r, "All features == 0"), axis=1)
    df = df[~zero_mask]

    # لو بعد التنظيف أصبحت الداتا فارغة
    if df.empty:
        # نحفظ ملفات فارغة + scaler و encoder إفتراضيين
        df_empty = pd.DataFrame(columns=id_cols + feat_cols + ["subject_id"])
        df_empty.to_csv(OUT_FILE, index=False)

        if drops:
            df_drops = pd.DataFrame(drops).sort_values(
                ["subject", "sessionIndex", "rep"],
                kind="mergesort"
            ).reset_index(drop=True)
        else:
            df_drops = pd.DataFrame(columns=["subject", "sessionIndex", "rep", "reason"])
        df_drops.to_csv(DROP_FILE, index=False)

        # scaler identity على عدد الميزات (حتى لا ينكسر الكود لاحقاً)
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(len(feat_cols))
        scaler.scale_ = np.ones(len(feat_cols))
        scaler.var_ = np.ones(len(feat_cols))
        scaler.n_features_in_ = len(feat_cols)
        scaler.feature_names_in_ = np.array(feat_cols, dtype=object)
        joblib.dump(scaler, SCALER_FILE)

        le = LabelEncoder()
        le.classes_ = np.array([], dtype=object)
        joblib.dump(le, ENCODER_FILE)

        META_FILE.write_text(json.dumps({
            "raw_file": str(RAW_FILE),
            "processed_file": str(OUT_FILE),
            "drop_report_file": str(DROP_FILE),
            "n_rows_processed": 0,
            "n_features": int(len(feat_cols)),
            "features": feat_cols,
            "settings": {
                "MAD_THRESH": MAD_THRESH,
                "MIN_ROWS_FOR_MAD": MIN_ROWS_FOR_MAD,
                "single_session_per_user": True,
                "nan_handling": "fillna(0.0), drop rows with all-NaN or all-zero features"
            }
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[WARN] No rows left after cleaning. Saved empty processed file.")
        return

    # 5) تنظيف الشذوذ بالـ MAD لكل subject إذا لديه صفين أو أكثر
    keep_idx = []
    for s, grp in df.groupby("subject"):
        if len(grp) < MIN_ROWS_FOR_MAD:
            # لديك جلسة واحدة (أو صف واحد) للشخص -> لا نطبق MAD
            keep_idx.extend(grp.index.tolist())
            continue

        X = grp[feat_cols].values
        bad = mad_outlier_mask(X, thresh=MAD_THRESH)
        grp.loc[bad].apply(
            lambda r: log_drop(r, f"outlier MAD>{MAD_THRESH} (per subject)"),
            axis=1
        )
        keep_idx.extend(grp.loc[~bad].index.tolist())

    df = df.loc[keep_idx].copy()

    # 6) ترميز subject إلى subject_id
    le = LabelEncoder()
    df["subject_id"] = le.fit_transform(df["subject"].astype(str))

    # 7) StandardScaler على أعمدة الميزات فقط
    scaler = StandardScaler()
    df_scaled = df.copy()

    if len(df_scaled) >= 2 and len(feat_cols):
        # عندنا صفّان أو أكثر -> نطبق التقييس بشكل طبيعي
        df_scaled[feat_cols] = scaler.fit_transform(df[feat_cols].values)
    else:
        # صف واحد فقط -> نترك القيم كما هي (raw) ونضبط scaler كـ identity
        scaler.mean_ = np.zeros(len(feat_cols))
        scaler.scale_ = np.ones(len(feat_cols))
        scaler.var_ = np.ones(len(feat_cols))
        scaler.n_features_in_ = len(feat_cols)
        scaler.feature_names_in_ = np.array(feat_cols, dtype=object)
        # لا نغيّر df_scaled[feat_cols] (تبقى القيم الأصلية)

    # 8) الفرز النهائي والحفظ
    df_scaled = df_scaled.sort_values(
        ["subject", "sessionIndex", "rep"],
        kind="mergesort"
    ).reset_index(drop=True)

    df_scaled.to_csv(OUT_FILE, index=False)

    # 9) حفظ تقرير الإسقاط (حتى لو كان فاضي)
    if drops:
        df_drops = (pd.DataFrame(drops)
                    .sort_values(["subject", "sessionIndex", "rep"],
                                 kind="mergesort")
                    .reset_index(drop=True))
    else:
        df_drops = pd.DataFrame(columns=["subject", "sessionIndex", "rep", "reason"])

    df_drops.to_csv(DROP_FILE, index=False)

    # 10) حفظ الأدوات والميتا
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)

    META_FILE.write_text(json.dumps({
        "raw_file": str(RAW_FILE),
        "processed_file": str(OUT_FILE),
        "drop_report_file": str(DROP_FILE),
        "n_rows_processed": int(len(df_scaled)),
        "n_features": int(len(feat_cols)),
        "features": feat_cols,
        "settings": {
            "MAD_THRESH": MAD_THRESH,
            "MIN_ROWS_FOR_MAD": MIN_ROWS_FOR_MAD,
            "single_session_per_user": True,
            "nan_handling": "fillna(0.0), drop rows with all-NaN or all-zero features",
            "adaptive_scaler": "if n_rows>=2 -> StandardScaler, else identity"
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved processed paragraph CSV -> {OUT_FILE}")
    print(f"[OK] Saved paragraph drop report   -> {DROP_FILE}")
    print(f"[OK] Saved scaler_paragraph.pkl    -> {SCALER_FILE}")
    print(f"[OK] Saved label_encoder_paragraph -> {ENCODER_FILE}")


if __name__ == "__main__":
    main()
