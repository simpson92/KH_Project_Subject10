from __future__ import annotations

# ==================================================
# day05_team_subject_split.py   (5일차 실습 30번 → 1조 버전)
# Approved Manifest → 사람 단위 train / val / test
# ==================================================
# 설정: configs/team01_subject_split.json (train_subjects · val_subjects · test_subjects)
#   - 같은 사람이 두 split에 들어가면 실패
#   - 각 split에 18개 동작(A01~A15 · D01~D03)이 모두 있어야 함
#
# 출력 (reports/team_projects/team01/main/dataset/splits/)
#   train.csv · val.csv · test.csv
#   split_counts.csv : split × 학습 라벨(16개) clip 수
#
# 실행
#   uv run python scripts/day05_team_subject_split.py
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_dataset_utils import split_by_subject, split_counts

DATASET_DIR = ROOT / "reports" / "team_projects" / "team01" / "main" / "dataset"
SPLIT_CONFIG = ROOT / "configs" / "team01_subject_split.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DATASET_DIR / "team01_approved_manifest.csv"))
    parser.add_argument("--split-config", default=str(SPLIT_CONFIG))
    parser.add_argument("--output-dir", default=str(DATASET_DIR / "splits"))
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    config = json.loads(Path(args.split_config).read_text(encoding="utf-8"))
    approved = pd.read_csv(args.manifest_path, encoding="utf-8-sig", keep_default_na=False)
    splits = split_by_subject(approved, config)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"[{name.upper()}] subjects={sorted(df['subject_id'].unique())} clips={len(df)}")
    counts = split_counts(splits)
    counts.to_csv(out / "split_counts.csv", index=False, encoding="utf-8-sig")
    print()
    print(counts.to_string(index=False))
    print(f"\nSaved: {out}")
    return splits


if __name__ == "__main__":
    main()
