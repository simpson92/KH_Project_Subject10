from __future__ import annotations

# ==================================================
# day05_team_make_webcam_log.py
# 촬영 로그 webcam_log.csv (03_웹캠_촬영_계획서 9.1) — manifest · Human QA에서 채울 수 있는 칸을 채움
# ==================================================
# 자동으로 채우는 칸: date · file_name · person_id · split_role · clip_type · action_id · take ·
#                    variation(normal) · angle_deg(0, W1 정면) · hand(right, W5) · resolution · fps · result · issue
# 사람이 채우는 칸 : cam_height_m · cam_distance_m (실측값) · operator (촬영 진행자 S##)
#   → 한 번 적으면 다시 실행해도 지워지지 않음
# 출력: reports/team_projects/team01/main/webcam_log.csv
#
# 실행: uv run python scripts/day05_team_make_webcam_log.py
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"
COLUMNS = ["date", "file_name", "person_id", "split_role", "clip_type", "action_id", "take",
           "variation", "angle_deg", "hand", "resolution", "fps", "cam_height_m", "cam_distance_m",
           "result", "issue", "operator", "note"]
MANUAL = ["cam_height_m", "cam_distance_m", "operator"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "configs" / "team01_main_manifest.csv"))
    parser.add_argument("--review", default=str(MAIN_DIR / "dataset" / "team01_clip_review.csv"))
    parser.add_argument("--split-config", default=str(ROOT / "configs" / "team01_subject_split.json"))
    parser.add_argument("--output", default=str(MAIN_DIR / "webcam_log.csv"))
    return parser.parse_args(argv)


def capture_date(capture_set: str) -> str:
    """main_1117 → 2026-11-17."""
    digits = "".join(ch for ch in str(capture_set) if ch.isdigit())[-4:]
    return f"2026-{digits[:2]}-{digits[2:]}" if len(digits) == 4 else ""


def build_log(manifest: pd.DataFrame, review: pd.DataFrame | None, test_subjects: set[str]) -> pd.DataFrame:
    decision = {}
    note = {}
    if review is not None:
        decision = dict(zip(review["clip_id"], review["human_decision"].str.upper()))
        note = dict(zip(review["clip_id"], review["human_note"]))
    rows = []
    for _, m in manifest.iterrows():
        clip, d = m["clip_id"], decision.get(m["clip_id"], "")
        reason = str(m.get("review_reason", "") or "")
        rows.append({
            "date": capture_date(m.get("capture_set", "")),
            "file_name": m["video_file"],
            "person_id": m["subject_id"],
            "split_role": "test" if m["subject_id"] in test_subjects else "train",
            "clip_type": "daily" if str(m["action_id"]).startswith("D") else "signal",
            "action_id": m["action_id"],
            "take": int(str(m["take_id"]).lstrip("T") or 0),
            "variation": "normal", "angle_deg": 0, "hand": "right",
            "resolution": f"{int(m['width'])}x{int(m['height'])}",
            "fps": round(float(m["fps"]), 2),
            "cam_height_m": "", "cam_distance_m": "",
            "result": "RESHOOT" if d in ("RESHOOT", "EXCLUDE") else "OK",
            "issue": reason if (reason or d in ("RESHOOT", "EXCLUDE")) else "",
            "operator": "",
            "note": note.get(clip, "") if d == "PASS" and reason else "",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False)
    review = pd.read_csv(args.review, encoding="utf-8-sig", keep_default_na=False) if Path(args.review).exists() else None
    tests = set(json.loads(Path(args.split_config).read_text(encoding="utf-8"))["test_subjects"])
    log = build_log(manifest, review, tests)
    out = Path(args.output)
    if out.exists():   # 사람이 적은 실측값 · 진행자 지키기
        old = pd.read_csv(out, encoding="utf-8-sig", keep_default_na=False, dtype=str).set_index("file_name")
        for col in MANUAL:
            filled = log["file_name"].map(old[col]) if col in old else None
            if filled is not None:
                log[col] = filled.where(filled.notna() & (filled != ""), log[col])
    out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False, encoding="utf-8-sig")
    empty = {c: int((log[c].astype(str) == "").sum()) for c in MANUAL}
    print(f"clip {len(log)}개 · RESHOOT {int((log['result'] == 'RESHOOT').sum())} · 비어 있는 실측 칸 {empty}")
    print(f"Saved: {out}")
    return log


if __name__ == "__main__":
    main()
