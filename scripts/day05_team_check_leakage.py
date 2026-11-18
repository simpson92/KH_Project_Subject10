from __future__ import annotations

# ==================================================
# day05_team_check_leakage.py   (5일차 실습 31번 → 1조 버전)
# train / val / test 사이에 같은 사람 · 같은 clip · 같은 영상 파일이 없는지
# ==================================================
# 실행
#   uv run python scripts/day05_team_check_leakage.py
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_dataset_utils import check_leakage

SPLIT_DIR = ROOT / "reports" / "team_projects" / "team01" / "main" / "dataset" / "splits"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    splits = {}
    for name in ("train", "val", "test"):
        path = Path(args.split_dir) / f"{name}.csv"
        if not path.exists():
            print(f"[ERROR] 없음: {path}")
            return 1
        splits[name] = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        print(f"{name:5s} subjects={sorted(splits[name]['subject_id'].unique())} clips={len(splits[name])}")

    errors = check_leakage(splits)
    print()
    if errors:
        for error in errors:
            print("[FAIL]", error)
        return 1
    print("[PASS] Subject overlap 없음")
    print("[PASS] Clip · 영상 파일 overlap 없음")
    print("[PASS] split마다 18개 동작 모두 있음")
    print("\nLeakage check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
