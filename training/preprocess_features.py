# training/preprocess_features.py
import os
import json
import joblib
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------- إعدادات قابلة للتعديل --------------------
EXPECTED_USERS = None          # لا نقتطع على مستوى عدد المستخدمين
ALLOWED_SESSIONS = {1, 2}       # نريد جلستين فقط
REP_MIN, REP_MAX = 1, 50        # نطاق rep المنطقي
WARMUP_SKIP = 3                  # تجاهل أول N محاولات بكل جلسة
MAD_THRESH = 4.5              # حد MAD للشذوذ (متين Robust)
BALANCE_BY_TRUNCATE = False      # موازنة عدد التكرارات بين الجميع بالقصّ
MIN_REPS_PER_SESSION = 10        # حد أدنى للتكرارات داخل كل (subject, sessionIndex)
# ---------------------------------------------------------------

# تعريف المسارات
BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE     = RAW_DIR  / "aggregated_tie5Roanl.csv"
OUT_FILE     = PROC_DIR / "processed_keystrokes.csv"
DROP_FILE    = PROC_DIR / "drop_report.csv"
SCALER_FILE  = PROC_DIR / "scaler.pkl"
ENCODER_FILE = PROC_DIR / "label_encoder.pkl"
META_FILE    = PROC_DIR / "preprocess_meta.json"

def _safe_unlink(path: Path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

# دعم خيار --clean من سطر الأوامر
parser = argparse.ArgumentParser()
parser.add_argument("--clean", action="store_true", help="احذف نواتج المعالجة السابقة ثم أعد التشغيل")
args, _ = parser.parse_known_args()
if args.clean:
    for p in [OUT_FILE, DROP_FILE, SCALER_FILE, ENCODER_FILE, META_FILE]:
        _safe_unlink(p)

# --- دالة كشف الشذوذ بالـ MAD (Robust) ---
def mad_outlier_mask(X: np.ndarray, thresh: float = MAD_THRESH) -> np.ndarray:
    """
    يُرجِع قناعًا منطقيًا (True = صف شاذ) بناءً على MAD لكل عمود.
    نعتبر الصف شاذًا إذا تجاوز أي عمود العتبة.
    """
    med = np.median(X, axis=0)
    abs_dev = np.abs(X - med)
    mad = np.median(abs_dev, axis=0)
    mad[mad == 0] = 1.0  # لتجنب القسمة على صفر
    z_mad = 0.6745 * abs_dev / mad  # ثابت التقريب للانحراف المعياري
    return (z_mad > thresh).any(axis=1)

def main():
    # 1) قراءة RAW
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"لم يتم العثور على الملف الخام: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)

    # --- تصحيح الأنواع والفرز المبكر ---
    if not df.empty:
        df["subject"] = df["subject"].astype(str)
        df["sessionIndex"] = pd.to_numeric(df["sessionIndex"], errors="coerce").fillna(0).astype(int)
        df["rep"] = pd.to_numeric(df["rep"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(["subject", "sessionIndex", "rep"], kind="mergesort").reset_index(drop=True)

    # 2) أعمدة الميزات
    feat_cols = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]
    if not feat_cols:
        raise RuntimeError("لم يتم العثور على أعمدة ميزات تبدأ بـ H./DD./UD. في الملف الخام.")

    # --- تجميع أسباب الإسقاط ---
    drops = []
    def log_drop(row, reason):
        drops.append({
            "subject": row.get("subject", ""),
            "sessionIndex": int(row.get("sessionIndex", 0)),
            "rep": int(row.get("rep", 0)),
            "reason": reason
        })

    # 3) إسقاط الصفوف NaN في الميزات
    nan_mask = df[feat_cols].isna().any(axis=1)
    df.loc[nan_mask].apply(lambda r: log_drop(r, "NaN in features"), axis=1)
    df = df[~nan_mask]

    # 4) إسقاط الصفوف التي مجموع ميزاتها = 0
    zero_mask = (df[feat_cols].sum(axis=1) == 0)
    df.loc[zero_mask].apply(lambda r: log_drop(r, "All features == 0"), axis=1)
    df = df[~zero_mask]

    # 5) نطاق rep منطقي
    rep_mask = ~(df["rep"].between(REP_MIN, REP_MAX))
    df.loc[rep_mask].apply(lambda r: log_drop(r, "rep out of [1,50]"), axis=1)
    df = df[~rep_mask]

    # 6) قبول الجلسات 1 و2 فقط
    sess_mask = ~df["sessionIndex"].isin(ALLOWED_SESSIONS)
    df.loc[sess_mask].apply(lambda r: log_drop(r, "session not in {1,2}"), axis=1)
    df = df[~sess_mask]

    # 6.5) تجاهل أول WARMUP_SKIP محاولات بكل جلسة (بعد الفرز)
    warm_idx = df["rep"] <= WARMUP_SKIP
    df.loc[warm_idx].apply(lambda r: log_drop(r, f"warmup skip (rep<={WARMUP_SKIP})"), axis=1)
    df = df[~warm_idx]

    # 7) اختيار 5 مستخدمين فقط (الأكثر اكتمالًا)
    if EXPECTED_USERS is not None:
        users_by_rows = (df.groupby("subject").size().sort_values(ascending=False)).index.tolist()
        chosen_users = users_by_rows[:EXPECTED_USERS]
        not_chosen_mask = ~df["subject"].isin(chosen_users)
        df.loc[not_chosen_mask].apply(lambda r: log_drop(r, f"subject not in chosen top-{EXPECTED_USERS}"), axis=1)
        df = df[~not_chosen_mask]

    # 8) تأكد أن كل مستخدم يملك الجلسات المطلوبة (1 و2)، وإلا احذف المستخدم بالكامل
    keep_subjects = []
    for s, g in df.groupby("subject"):
        sessions = set(g["sessionIndex"].unique().tolist())
        if ALLOWED_SESSIONS.issubset(sessions):
            keep_subjects.append(s)
        else:
            df.loc[df["subject"] == s].apply(lambda r: log_drop(r, "missing one of sessions {1,2}"), axis=1)
    df = df[df["subject"].isin(keep_subjects)]

    # 9) حد أدنى للتكرارات داخل كل (subject, session)
    bad_idx = []
    for (s, si), grp in df.groupby(["subject", "sessionIndex"]):
        if grp["rep"].nunique() < MIN_REPS_PER_SESSION:
            bad_idx.extend(grp.index.tolist())
    if bad_idx:
        df.loc[bad_idx].apply(lambda r: log_drop(r, f"reps per session < {MIN_REPS_PER_SESSION}"), axis=1)
        df = df.drop(index=bad_idx)

    # 10) تنظيف الشذوذ باستخدام MAD لكل (subject, session)
    keep_idx = []
    for (s, si), grp in df.groupby(["subject", "sessionIndex"]):
        X = grp[feat_cols].values
        bad = mad_outlier_mask(X, thresh=MAD_THRESH)  # True = صف شاذ
        # سجل السجلات الشاذة
        grp.loc[bad].apply(lambda r: log_drop(r, f"outlier MAD>{MAD_THRESH} (per subject-session)"), axis=1)
        keep_idx.extend(grp.loc[~bad].index.tolist())
    df = df.loc[keep_idx].copy()

    # 11) موازنة التكرارات بين الجميع (اختياري)
    if BALANCE_BY_TRUNCATE and not df.empty:
        min_reps = []
        for (s, si), grp in df.groupby(["subject", "sessionIndex"]):
            min_reps.append(grp["rep"].nunique())
        if min_reps:
            cap = min(min_reps)
            keep_rows = []
            for (s, si), grp in df.sort_values(["subject", "sessionIndex", "rep"]).groupby(["subject", "sessionIndex"]):
                uniq_reps = sorted(grp["rep"].unique())[:cap]
                keep_rows.append(grp[grp["rep"].isin(uniq_reps)])
                extra_rows = grp[~grp["rep"].isin(uniq_reps)]
                if len(extra_rows):
                    extra_rows.apply(lambda r: log_drop(r, f"truncate to balance (cap={cap})"), axis=1)
            df = pd.concat(keep_rows, ignore_index=True)

    # 12) تشفير subject إلى subject_id + StandardScaler
    le = LabelEncoder()
    df["subject_id"] = le.fit_transform(df["subject"].astype(str))

    scaler = StandardScaler()
    df_scaled = df.copy()

    if len(df_scaled) >= 2 and len(feat_cols):
        # عندنا أكثر من صف → تقييس طبيعي
        df_scaled[feat_cols] = scaler.fit_transform(df[feat_cols].values)
    else:
        # عندنا صف واحد فقط → لا نطبّق التقييس، نترك القيم الأصلية
        # لكن نضع scaler "هوية" حتى لا يتعطل الكود لاحقًا
        scaler.mean_ = np.zeros(len(feat_cols))
        scaler.scale_ = np.ones(len(feat_cols))
        scaler.var_ = np.ones(len(feat_cols))
        scaler.n_features_in_ = len(feat_cols)
        scaler.feature_names_in_ = np.array(feat_cols, dtype=object)

    # 13) الفرز النهائي والحفظ
    df_scaled = df_scaled.sort_values(["subject", "sessionIndex", "rep"], kind="mergesort").reset_index(drop=True)

    # حفظ البيانات المعالجة
    df_scaled.to_csv(OUT_FILE, index=False)

    # حفظ تقرير الإسقاط
    pd.DataFrame(drops).sort_values(["subject", "sessionIndex", "rep"], kind="mergesort").to_csv(DROP_FILE, index=False)

    # حفظ الأدوات
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)

    # ميتاداتا
    META_FILE.write_text(json.dumps({
        "raw_file": str(RAW_FILE),
        "processed_file": str(OUT_FILE),
        "drop_report_file": str(DROP_FILE),
        "n_rows_processed": int(len(df_scaled)),
        "n_features": int(len(feat_cols)),
        "features": feat_cols,
        "chosen_users": keep_subjects if EXPECTED_USERS is None else keep_subjects[:EXPECTED_USERS],
        "settings": {
            "EXPECTED_USERS": EXPECTED_USERS,
            "ALLOWED_SESSIONS": sorted(list(ALLOWED_SESSIONS)),
            "REP_MIN": REP_MIN, "REP_MAX": REP_MAX,
            "WARMUP_SKIP": WARMUP_SKIP,
            "MAD_THRESH": MAD_THRESH,
            "BALANCE_BY_TRUNCATE": BALANCE_BY_TRUNCATE,
            "MIN_REPS_PER_SESSION": MIN_REPS_PER_SESSION
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved processed CSV   -> {OUT_FILE}")
    print(f"[OK] Saved drop report     -> {DROP_FILE}")
    print(f"[OK] Saved scaler.pkl      -> {SCALER_FILE}")
    print(f"[OK] Saved label_encoder   -> {ENCODER_FILE}")

if __name__ == "__main__":
    main()
