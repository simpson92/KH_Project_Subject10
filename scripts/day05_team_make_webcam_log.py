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
# 빈 칸 한 번에 채우기 (모든 촬영을 같은 세팅으로 했을 때 — 이미 적힌 칸은 그대로, --force면 덮어씀)
#   uv run python scripts/day05_team_make_webcam_log.py --cam-height 1.3 --cam-distance 2.0 \
#       --operator-map S01=S05,S02=S04,S03=S04,S04=S03,S05=S01 --value-note "실측(11-19, 촬영 위치 재현)"
#   --operator-map : 촬영된 사람=찍어 준 사람. 한 사람이 다 찍었으면 --operator S05
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
    parser.add_argument("--cam-height", default="", help="카메라 높이 m (예: 1.3)")
    parser.add_argument("--cam-distance", default="", help="카메라 ↔ 서는 위치 거리 m (예: 2.0)")
    parser.add_argument("--operator", default="", help="촬영 진행자 S## (전원 같을 때)")
    parser.add_argument("--operator-map", default="", help="촬영된 사람=진행자, 쉼표로 (S01=S05,S02=S04)")
    parser.add_argument("--value-note", default="", help="채운 값의 출처를 note에 남김 (예: 실측 · 계획값)")
    parser.add_argument("--force", action="store_true", help="이미 적힌 칸도 덮어씀")
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


def fill_manual(log: pd.DataFrame, args) -> pd.DataFrame:
    """--cam-height · --cam-distance · --operator(-map) 값으로 빈 칸 채우기."""
    log = log.copy()
    for col, value in (("cam_height_m", args.cam_height), ("cam_distance_m", args.cam_distance)):
        if value:
            float(value)   # 숫자가 아니면 여기서 멈춤
            mask = (log[col].astype(str) == "") | bool(args.force)
            log.loc[mask, col] = value
    mapping = dict(x.split("=") for x in args.operator_map.split(",") if "=" in x)
    if args.operator or mapping:
        ops = log["person_id"].map(mapping).fillna(args.operator) if mapping else pd.Series(args.operator, index=log.index)
        mask = ((log["operator"].astype(str) == "") | bool(args.force)) & (ops != "")
        log.loc[mask, "operator"] = ops[mask]
    if args.value_note and (args.cam_height or args.cam_distance):
        tag = f"높이·거리: {args.value_note}"
        log["note"] = log["note"].astype(str).map(lambda n: n if tag in n else (f"{n} · {tag}" if n else tag))
    return log


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
    log = fill_manual(log, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False, encoding="utf-8-sig")
    empty = {c: int((log[c].astype(str) == "").sum()) for c in MANUAL}
    print(f"clip {len(log)}개 · RESHOOT {int((log['result'] == 'RESHOOT').sum())} · 비어 있는 실측 칸 {empty}")
    print(f"Saved: {out}")
    return log


if __name__ == "__main__":
    main()
