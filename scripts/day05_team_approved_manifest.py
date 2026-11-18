from __future__ import annotations

# ==================================================
# day05_team_approved_manifest.py   (5일차 실습 29번 → 1조 버전)
# Human QA에서 PASS 받은 clip만 모은 Approved Manifest
# ==================================================
# 출력: reports/team_projects/team01/main/dataset/team01_approved_manifest.csv
#   - train_label 열: A01~A15 그대로, D01~D03 → N00 (학습 라벨 16개)
#   - RESHOOT · EXCLUDE · NO_SEQUENCE 는 빠짐
#
# 실행
#   uv run python scripts/day05_team_approved_manifest.py
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_dataset_utils import build_approved

DATASET_DIR = ROOT / "reports" / "team_projects" / "team01" / "main" / "dataset"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-path", default=str(DATASET_DIR / "team01_clip_review.csv"))
    parser.add_argument("--output-path", default=str(DATASET_DIR / "team01_approved_manifest.csv"))
    return parser.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    review = pd.read_csv(args.review_path, encoding="utf-8-sig", keep_default_na=False)
    approved = build_approved(review)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    approved.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Approved clips: {len(approved)} / {len(review)}")
    print("\n[By Subject]")
    print(approved.groupby("subject_id").size().to_string())
    print("\n[By Train Label]")
    print(approved.groupby("train_label").size().to_string())
    print(f"\nSaved: {out}")
    return approved


if __name__ == "__main__":
    main()
