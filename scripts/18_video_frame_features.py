from __future__ import annotations

# ==================================================
# 18_video_frame_features.py
# 영상의 각 Frame을 38-D Frame Feature로 바꾸어
# 시간 순서대로 CSV에 저장합니다. (교재 3일차 26절)
#
# 16_make_frame_feature.py가 이미지 1장에 한 계산을
# Frame마다 반복합니다.
#
# 한 행 = 한 Frame
# frame_id · time_sec · valid_pose
# mean_keypoint_conf · low_keypoint_count
# hip_center_x · hip_center_y · shoulder_width
# + 38-D (34 정규화 좌표 + 4 관절각도)
# ==================================================

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ==================================================
# [01] 프로젝트 ROOT 경로 설정 + 공통 함수 가져오기
# ==================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device  # noqa: E402
from src.feature_utils import (  # noqa: E402
    normalize_by_shoulder_width,
    selected_joint_angles,
)
from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402


# ==================================================
# [02] 실행 옵션
# ==================================================
# --source     : 분석할 MP4 (기본 feature_demo.mp4)
# --kp-conf    : LOW 관절을 셀 Confidence 기준 (기본 0.50)
# --max-frames : 최대 처리 Frame 수 (0 = 전체)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(ROOT / "data" / "day03" / "videos" / "feature_demo.mp4"),
    )
    parser.add_argument("--kp-conf", type=float, default=0.50)
    parser.add_argument("--max-frames", type=int, default=150)
    return parser.parse_args()


# ==================================================
# [03] 38개 Feature 이름 (16번과 같은 순서)
# ==================================================
def make_feature_names():
    names = []
    for keypoint_name in COCO_KEYPOINT_NAMES:
        names.append(f"{keypoint_name}_x")
        names.append(f"{keypoint_name}_y")
    names.extend(
        [
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
        ]
    )
    return names


# ==================================================
# [04] Frame 한 장 → 38-D Feature 한 행
# ==================================================
# Pose가 없거나 정규화·각도 계산이 불가능하면
# valid_pose = 0 으로 기록하고 Feature는 빈 값(NaN)으로 둡니다.
# Confidence가 낮아도 Frame을 버리지 않고 품질 정보만 기록합니다.
def frame_to_row(result, frame_id, time_sec, kp_conf, feature_names):
    row = {
        "frame_id": frame_id,
        "time_sec": round(time_sec, 3),
        "valid_pose": 0,
        "mean_keypoint_conf": "",
        "low_keypoint_count": "",
        "hip_center_x": "",
        "hip_center_y": "",
        "shoulder_width": "",
    }
    for name in feature_names:
        row[name] = ""

    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        return row

    # 16번과 같이 첫 번째 사람을 사용합니다.
    keypoints = result.keypoints.xy[0].cpu().numpy()
    if result.keypoints.conf is not None:
        confidence = result.keypoints.conf[0].cpu().numpy()
    else:
        confidence = np.ones(17, dtype=np.float32)

    row["mean_keypoint_conf"] = round(float(np.mean(confidence)), 4)
    row["low_keypoint_count"] = int(np.sum(confidence < kp_conf))

    try:
        normalized, center, scale = normalize_by_shoulder_width(keypoints)
    except ValueError:
        # 어깨 폭이 0에 가까우면 정규화할 수 없습니다.
        return row

    angles = selected_joint_angles(keypoints)

    values = []
    for point in normalized:
        values.extend([float(point[0]), float(point[1])])
    values.extend(
        [
            angles["left_elbow_angle"],
            angles["right_elbow_angle"],
            angles["left_knee_angle"],
            angles["right_knee_angle"],
        ]
    )

    if any(math.isnan(v) for v in values):
        # 각도를 계산할 수 없는 Frame은 유효 Feature로 쓰지 않습니다.
        return row

    row["valid_pose"] = 1
    row["hip_center_x"] = round(float(center[0]), 4)
    row["hip_center_y"] = round(float(center[1]), 4)
    row["shoulder_width"] = round(float(scale), 4)
    for name, value in zip(feature_names, values):
        row[name] = round(float(value), 6)
    return row


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    # ==============================================
    # [05] 설정 · 장치 · 모델 준비
    # ==============================================
    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    # ==============================================
    # [06] 영상 열기
    # ==============================================
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    feature_names = make_feature_names()
    rows = []
    frame_id = 0

    # ==============================================
    # [07] Frame마다 Pose → 38-D 반복
    # ==============================================
    while True:
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        time_sec = (frame_id - 1) / fps

        results = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )
        rows.append(
            frame_to_row(results[0], frame_id, time_sec, args.kp_conf, feature_names)
        )

    cap.release()

    # ==============================================
    # [08] CSV 저장
    # ==============================================
    output_path = (
        ROOT / "reports" / "day03" / "tables" / f"{source.stem}_frame_features.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    valid_count = sum(row["valid_pose"] for row in rows)
    print(f"Source: {source.name}")
    print(f"Total frames: {len(rows)}")
    print(f"Valid pose frames: {valid_count}")
    print(f"Feature count per frame: {len(feature_names)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()