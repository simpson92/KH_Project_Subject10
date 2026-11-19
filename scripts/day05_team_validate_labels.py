from __future__ import annotations

# ==================================================
# day05_team_validate_labels.py   (02_라벨링_가이드 7장 · validate_labels.py)
# labels.csv 자동 검증 V-01 ~ V-10 — 오류가 하나라도 있으면 학습 투입 금지
# ==================================================
# 실행
#   uv run python scripts/day05_team_validate_labels.py                  # 초안 검사 (REVIEW 허용)
#   uv run python scripts/day05_team_validate_labels.py --for-training   # 학습 직전 (V-10: REVIEW 남으면 오류)
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_label_utils import validate_labels

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(MAIN_DIR / "labels" / "team01_labels.csv"))
    parser.add_argument("--sequence-dir", default=str(MAIN_DIR / "sequences"),
                        help="있으면 마지막 Frame 번호(V-05)까지 확인")
    parser.add_argument("--for-training", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    labels = pd.read_csv(args.labels, keep_default_na=False)
    counts = {}
    seq_dir = Path(args.sequence_dir)
    for clip in labels["clip_id"].unique():
        path = seq_dir / f"{clip}_sequence.csv"
        if path.exists():
            with path.open(encoding="utf-8-sig") as f:
                counts[clip] = sum(1 for _ in f) - 1
    errors, warnings = validate_labels(labels, counts, for_training=args.for_training)
    print(f"clip {labels['clip_id'].nunique()}개 · 구간 {len(labels)}행 · Frame 수 확인 {len(counts)}개")
    for w in warnings:
        print("[WARN]", w)
    for e in errors:
        print("[FAIL]", e)
    if errors:
        print(f"\n오류 {len(errors)}건 → 학습 투입 금지")
        return 1
    print(f"\n경고 {len(warnings)}건 · 오류 0건 → Labels check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
