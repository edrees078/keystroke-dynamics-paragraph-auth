# collector/runtime_io.py
import csv, time
from .constants import SESSIONS_DIR, DECISIONS_DIR

def save_session_rows(rows, filename_prefix="session"):
    # rows: list of dicts (feature columns + subject/session/rep...)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DIR / f"{filename_prefix}_{ts}.csv"
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return str(path)

def append_decision(record):
    # record: dict {ts, claimed_user, pred_name, score, thr, accepted}
    path = DECISIONS_DIR / "decisions_log.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return str(path)
