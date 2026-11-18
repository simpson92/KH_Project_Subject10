from __future__ import annotations

# ==================================================
# day05_team_validate_review.py   (5일차 실습 28번 → 1조 버전)
# Human QA 파일 검사 — 모든 clip에 PASS / RESHOOT / EXCLUDE 가 적혔는지
# ==================================================
# 실습 규칙 + 1조 추가 규칙
#   - UNREVIEWED · 빈칸 · 허용하지 않는 값 → 실패
#   - 자동 QA가 CHECK인데 human_note 없이 PASS → 실패 (왜 통과시켰는지 남기기)
#   - NO_SEQUENCE (STEP 4 안 돌린 clip) → 실패 (--allow-no-sequence 이면 통과)
#
# 실행
#   uv run python scripts/day05_team_validate_review.py
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_dataset_utils import normalize_decisions, validate_review

DATASET_DIR = ROOT / "reports" / "team_projects" / "team01" / "main" / "dataset"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-path", default=str(DATASET_DIR / "team01_clip_review.csv"))
    parser.add_argument("--expected-clips", type=int, default=0, help="0이면 검사 안 함")
    parser.add_argument("--allow-no-sequence", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    review_path = Path(args.review_path)
    if not review_path.exists():
        print(f"[ERROR] Human QA 파일이 없습니다: {review_path}")
        return 1
    df = pd.read_csv(review_path, encoding="utf-8-sig", keep_default_na=False)
    errors = validate_review(df, allow_no_sequence=args.allow_no_sequence)
    if args.expected_clips and len(df) != args.expected_clips:
        errors.append(f"clip 수 expected={args.expected_clips}, actual={len(df)}")
    if errors:
        print("[HUMAN QA NOT COMPLETE]")
        for error in errors:
            print(" -", error)
        return 1

    decisions = normalize_decisions(df)
    print("[Human QA Result]")
    print(decisions.value_counts().to_string())
    print()
    print(pd.crosstab(df["qa_status"], decisions).to_string())
    print(f"\nReviewed clips: {len(df)}")
    print("Human review: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
