from __future__ import annotations

# ==================================================
# day05_team_frame_split_demo.py   (5일차 실습 32번 → 1조 버전)
# "Frame을 무작위로 섞어 나누면 왜 안 되는가" 를 1조 데이터로 확인
# ==================================================
# Approved clip의 Sequence CSV(pose_ok = 1 Frame)를 모두 합친 뒤
# Frame 단위로 train_test_split(test_size=0.2, random_state=42) 하면
#   - 같은 사람 · 같은 clip의 바로 옆 Frame이 train과 test에 함께 들어감
#   - 모델이 "동작"이 아니라 "그 사람 · 그 영상"을 외워도 점수가 높게 나옴 (6일차 Val 점수가 부풀려짐)
# Subject-wise Split(day05_team_subject_split.py)과 비교하는 데모입니다. 학습은 하지 않습니다.
#
# 실행
#   uv run python scripts/day05_team_frame_split_demo.py
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "reports" / "team_projects" / "team01" / "main" / "dataset"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DATASET_DIR / "team01_approved_manifest.csv"))
    parser.add_argument("--sequence-dir", default="", help="비우면 manifest의 sequence_csv 경로 사용")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_frames(approved: pd.DataFrame, sequence_dir: str = "") -> pd.DataFrame:
    frames = []
    for _, item in approved.iterrows():
        path = (Path(sequence_dir) / f"{item['clip_id']}_sequence.csv" if sequence_dir
                else ROOT / item["sequence_csv"])
        df = pd.read_csv(path, usecols=["clip_id", "subject_id", "frame_id", "pose_ok"])
        frames.append(df[df["pose_ok"] == 1])
    return pd.concat(frames, ignore_index=True)


def frame_split_overlap(frames: pd.DataFrame, test_size=0.2, seed=42) -> dict:
    train, test = train_test_split(frames, test_size=test_size, random_state=seed, shuffle=True)
    subj = set(train["subject_id"]) & set(test["subject_id"])
    clips = set(train["clip_id"]) & set(test["clip_id"])
    return {
        "frames": len(frames), "train_frames": len(train), "test_frames": len(test),
        "subject_overlap": len(subj), "clip_overlap": len(clips),
        "test_clips": test["clip_id"].nunique(),
    }


def main(argv=None) -> dict:
    args = parse_args(argv)
    approved = pd.read_csv(args.manifest_path, encoding="utf-8-sig", keep_default_na=False)
    frames = load_frames(approved, args.sequence_dir)
    result = frame_split_overlap(frames, args.test_size, args.seed)
    print("[Frame 무작위 분할 — 하면 안 되는 방법]")
    for key, value in result.items():
        print(f"  {key:16s}: {value}")
    print(f"\n→ test clip {result['test_clips']}개 중 {result['clip_overlap']}개가 train에도 있음 (Clip Leakage)")
    print(f"→ 사람 {result['subject_overlap']}명이 train · test 양쪽에 있음 (Subject Leakage)")
    print("→ 1조는 day05_team_subject_split.py 로 사람 단위 분할")
    return result


if __name__ == "__main__":
    main()
