from __future__ import annotations

# ==================================================
# day05_team_make_labels.py
# 구간 라벨 labels.csv 자동 초안 (02_라벨링_가이드 5.2 · 6장)
# ==================================================
# 입력
#   configs/team01_main_manifest.csv
#   reports/team_projects/team01/main/sequences/<clip_id>_sequence.csv   (STEP 4)
#   reports/team_projects/team01/main/dataset/team01_clip_review.csv     (Human QA — EXCLUDE · RESHOOT 반영)
# 출력
#   reports/team_projects/team01/main/labels/team01_labels.csv  (UTF-8, BOM 없음, 1행 = 1구간)
#
# 규칙
#   - 신호 clip: N00 → X → C01 → X → A## → X → N00 (src/team_label_utils.py)
#   - 일상 동작 D01~D03: clip 전체 N00 (R-11)
#   - labeler = auto. 사람이 경계를 고치면 그 clip 행의 labeler를 S## 로 바꿈 → 다시 돌려도 지켜짐
#   - 구간을 못 찾거나 신호가 0.5초보다 짧으면 qa_status = REVIEW
#
# 실행
#   uv run python scripts/day05_team_make_labels.py
#   uv run python scripts/day05_team_validate_labels.py
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_label_utils import COLUMNS, keep_human_rows, label_rows, segment_signal

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"
MANIFEST_CSV = ROOT / "configs" / "team01_main_manifest.csv"
REVIEW_CSV = MAIN_DIR / "dataset" / "team01_clip_review.csv"
LABELS_CSV = MAIN_DIR / "labels" / "team01_labels.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--review", default=str(REVIEW_CSV))
    parser.add_argument("--sequence-dir", default=str(MAIN_DIR / "sequences"))
    parser.add_argument("--output", default=str(LABELS_CSV))
    return parser.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False)
    decisions = {}
    if Path(args.review).exists():
        review = pd.read_csv(args.review, encoding="utf-8-sig", keep_default_na=False)
        decisions = dict(zip(review["clip_id"], review["human_decision"].str.upper()))

    rows, missing = [], []
    for _, item in manifest.iterrows():
        clip = item["clip_id"]
        path = Path(args.sequence_dir) / f"{clip}_sequence.csv"
        if not path.exists():
            missing.append(clip)
            continue
        seq = pd.read_csv(path)
        action = str(item["action_id"])
        seg = None if action.startswith("D") else segment_signal(seq, action)
        decision = decisions.get(clip, "")
        override = decision if decision in ("EXCLUDE", "RESHOOT") else ""
        note = f"Human QA {decision}" if override else ""
        fps = int(round(float(item.get("fps") or 30)))
        rows += label_rows(clip, item["subject_id"], action, len(seq), seg, fps=fps,
                           qa_override=override, note_override=note,
                           action_label=str(item.get("action_label", "")))

    labels = pd.DataFrame(rows, columns=COLUMNS)
    out = Path(args.output)
    previous = pd.read_csv(out, keep_default_na=False) if out.exists() else None
    labels = keep_human_rows(labels, previous)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False, encoding="utf-8", float_format="%.3f")

    per_clip = labels.groupby("clip_id")["qa_status"].first()
    print(f"clip {labels['clip_id'].nunique()}개 · 구간 {len(labels)}행 · Sequence 없음 {len(missing)}개")
    print(per_clip.value_counts().to_string())
    length = (labels["end_frame"] - labels["start_frame"] + 1) / labels["fps"]
    kind = labels["label"].where(~labels["label"].str.match(r"A\d\d"), "A##")
    print("\n[구간 평균 길이 (초)]")
    print(length.groupby(kind).mean().round(2).to_string())
    review = per_clip[per_clip == "REVIEW"].index.tolist()
    if review:
        print("\n[REVIEW]", ", ".join(review))
    print(f"\nSaved: {out}")
    return labels


if __name__ == "__main__":
    main()
