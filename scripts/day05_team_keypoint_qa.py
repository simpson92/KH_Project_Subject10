from __future__ import annotations

# ==================================================
# day05_team_keypoint_qa.py   (5일차 실습 27번 → 1조 버전)
# 동작별 중요 관절 자동 QA + Human QA 입력 파일 만들기
# ==================================================
# 입력
#   configs/team01_main_manifest.csv
#   reports/team_projects/team01/main/sequences/<clip_id>_sequence.csv  (STEP 4 결과)
#
# 출력 (reports/team_projects/team01/main/dataset/)
#   team01_clip_qa.csv      : 자동 QA 결과 (AUTO_PASS / CHECK / NO_SEQUENCE + 걸린 기준)
#   team01_clip_review.csv  : 위 표 + human_decision · human_note  ← 사람이 채우는 파일
#
# 실습과 다른 점
#   - 다시 돌려도 review 파일에 이미 적은 human_decision · human_note는 지킴 (450개라 다시 적기 어려움)
#   - Sequence CSV가 없는 clip은 멈추지 않고 NO_SEQUENCE로 남김 (STEP 4 후 다시 실행)
#
# 실행
#   uv run python scripts/day05_team_keypoint_qa.py
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config
from src.team_dataset_utils import build_qa_table, merge_previous_review

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"
MANIFEST_CSV = ROOT / "configs" / "team01_main_manifest.csv"
SEQUENCE_DIR = MAIN_DIR / "sequences"
DATASET_DIR = MAIN_DIR / "dataset"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--sequence-dir", default=str(SEQUENCE_DIR))
    parser.add_argument("--qa-path", default=str(DATASET_DIR / "team01_clip_qa.csv"))
    parser.add_argument("--review-path", default=str(DATASET_DIR / "team01_clip_review.csv"))
    parser.add_argument("--expected-clips", type=int, default=0, help="0이면 검사 안 함")
    return parser.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False)
    if args.expected_clips and len(manifest) != args.expected_clips:
        raise RuntimeError(f"Manifest Clip 수가 다릅니다. expected={args.expected_clips}, actual={len(manifest)}")

    threshold = float(load_config().get("keypoint_confidence", 0.50))
    qa = build_qa_table(manifest, Path(args.sequence_dir), threshold, root=ROOT)

    qa_path, review_path = Path(args.qa_path), Path(args.review_path)
    previous = None
    if review_path.exists():
        previous = pd.read_csv(review_path, encoding="utf-8-sig", keep_default_na=False)
    review = merge_previous_review(qa, previous)

    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa.to_csv(qa_path, index=False, encoding="utf-8-sig")
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    print(f"threshold = {threshold} · clip {len(qa)}개")
    print(qa["qa_status"].value_counts().to_string())
    check = qa[qa["qa_status"] == "CHECK"]
    if not check.empty:
        print("\n[CHECK]")
        print(check[["clip_id", "qa_reason", "pose_ratio", "mean_keypoint_conf", "low_critical_ratio"]]
              .to_string(index=False))
    left = int((review["human_decision"].astype(str).str.upper() == "UNREVIEWED").sum())
    print(f"\nHuman QA 남은 clip: {left}")
    print(f"QA     : {qa_path}")
    print(f"Review : {review_path}")
    return qa


if __name__ == "__main__":
    main()
