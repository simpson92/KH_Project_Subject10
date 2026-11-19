from __future__ import annotations

# ==================================================
# day05_team_export_keypoints.py
# STEP 4 Sequence CSV → 5일차 수업 Keypoint CSV 형식 (실습 26번과 같은 열)
# ==================================================
# 왜?  6일차 수업 코드는 5일차 Keypoint CSV(<관절>_x_px · _y_px · _x_norm · _y_norm · _conf)를 읽음.
#      1조는 YOLO를 26번으로 다시 돌리지 않고, 이미 만든 Sequence CSV에서 바로 옮김 (같은 YOLO 결과)
#        x_px · y_px  = Sequence의 raw_<관절>_x · _y
#        x_norm       = x_px / 영상 너비,  y_norm = y_px / 영상 높이  (이미지 정규화 — 골반 · 어깨 정규화와 다름)
#        conf         = conf_<관절>,  person_count = people,  pose_detected = pose_ok
# 입력: reports/team_projects/team01/main/dataset/team01_approved_manifest.csv (keypoint_csv 열 = 저장 위치)
# 출력: data/team_projects/team01/main/keypoints/<S##>/<clip_id>.csv   (data/ = Git 제외)
#
# 실행
#   uv run python scripts/day05_team_export_keypoints.py
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import COCO_KEYPOINT_NAMES

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MAIN_DIR / "dataset" / "team01_approved_manifest.csv"))
    parser.add_argument("--root", default=str(ROOT), help="manifest 경로의 기준 폴더")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def to_keypoint_frame(seq: pd.DataFrame, item: dict) -> pd.DataFrame:
    """Sequence 1개 → 수업 26번 열 순서의 DataFrame."""
    width = seq["image_width"].astype(float)
    height = seq["image_height"].astype(float)
    pose = seq["pose_ok"].fillna(0).astype(int)
    out = pd.DataFrame({
        "clip_id": item["clip_id"],
        "frame_id": seq["frame_id"].astype(int),
        "time_sec": seq["time_sec"].round(4),
        "subject": item["subject_id"],
        "action_id": item["action_id"],
        "action_label": item["action_label"],
        "take": item["take_id"],
        "person_count": seq["people"].fillna(0).astype(int),
        "selected_box_conf": seq["bbox_conf"].where(pose == 1).round(4),
        "pose_detected": pose,
    })
    for name in COCO_KEYPOINT_NAMES:
        x = seq[f"raw_{name}_x"].where(pose == 1)
        y = seq[f"raw_{name}_y"].where(pose == 1)
        out[f"{name}_x_px"] = x.round(2)
        out[f"{name}_y_px"] = y.round(2)
        out[f"{name}_x_norm"] = (x / width).round(6)
        out[f"{name}_y_norm"] = (y / height).round(6)
        out[f"{name}_conf"] = seq[f"conf_{name}"].where(pose == 1).round(4)
    out["train_label"] = item.get("train_label", item["action_id"])   # 1조: D01~D03 → N00
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False)
    saved = skipped = 0
    for _, item in manifest.iterrows():
        out = root / item["keypoint_csv"]
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        seq = pd.read_csv(root / item["sequence_csv"])
        frame = to_keypoint_frame(seq, item.to_dict())
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, encoding="utf-8-sig")
        saved += 1
    print(f"Keypoint CSV 저장 {saved}개 · 이미 있음 {skipped}개 (다시 만들려면 --overwrite)")
    print(f"위치: {root / 'data/team_projects/team01/main/keypoints'}  (Git 제외)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
