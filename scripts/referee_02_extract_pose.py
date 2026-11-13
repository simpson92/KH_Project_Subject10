from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device
from src.pose_utils import COCO_KEYPOINT_NAMES
from src.referee_utils import (
    KEYPOINT_DIR,
    MANIFEST_CSV,
    POSE_QUALITY_CSV,
)


# ③ 수신호에서 중요한 팔 관절 (어깨·팔꿈치·손목)
ARM_IDS = [5, 6, 7, 8, 9, 10]


def parse_args():
    parser = argparse.ArgumentParser()

    # 처음에는 몇 개만 돌려서 확인할 때 사용 (0이면 전체)
    parser.add_argument("--max-videos", type=int, default=0)

    # 이미 추출한 영상도 다시 추출
    parser.add_argument("--overwrite", action="store_true")

    # 화면에 여러 명이 잡혔을 때 대표 사람을 고르는 기준
    #   area: BBox가 가장 큰 사람 (촬영 대상이 보통 가장 가까움)
    #   conf: BBox Confidence가 가장 높은 사람 (2일차 방식)
    parser.add_argument(
        "--main-person",
        choices=["area", "conf"],
        default="area",
    )

    # LOW 관절 판단 기준
    parser.add_argument("--kp-conf", type=float, default=0.50)

    return parser.parse_args()


def keypoint_columns() -> list[str]:
    """frame_id, time_sec, ... nose_x, nose_y, nose_c, ... right_ankle_c"""
    columns = [
        "frame_id",
        "time_sec",
        "people",
        "box_conf",
        "img_w",
        "img_h",
    ]
    for name in COCO_KEYPOINT_NAMES:
        columns += [f"{name}_x", f"{name}_y", f"{name}_c"]
    return columns


def extract_video(model, config, device, video_path, main_person):
    """영상 하나를 Frame별로 읽어 대표 사람의 17개 관절을 반환합니다."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0

    rows = []
    frame_id = 0

    # ④ Frame마다 YOLO Pose 실행
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_id += 1
        height, width = frame.shape[:2]

        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )[0]

        people = len(result.boxes) if result.boxes is not None else 0

        row = {
            "frame_id": frame_id,
            "time_sec": round((frame_id - 1) / fps, 3),
            "people": people,
            "box_conf": 0.0,
            "img_w": width,
            "img_h": height,
        }

        # 사람이 없는 Frame은 좌표를 비워 두고 Confidence 0
        xy = np.full((17, 2), np.nan)
        conf = np.zeros(17)

        # ⑤ 대표 사람 한 명 선택
        if (
            people > 0
            and result.keypoints is not None
            and result.keypoints.conf is not None
        ):
            boxes = result.boxes.xyxy.cpu().numpy()
            box_conf = result.boxes.conf.cpu().numpy()

            if main_person == "area":
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                index = int(np.argmax(areas))
            else:
                index = int(np.argmax(box_conf))

            row["box_conf"] = round(float(box_conf[index]), 4)
            xy = result.keypoints.xy.cpu().numpy()[index]
            conf = result.keypoints.conf.cpu().numpy()[index]

        for kp_id, name in enumerate(COCO_KEYPOINT_NAMES):
            x, y = xy[kp_id]
            row[f"{name}_x"] = "" if np.isnan(x) else round(float(x), 2)
            row[f"{name}_y"] = "" if np.isnan(y) else round(float(y), 2)
            row[f"{name}_c"] = round(float(conf[kp_id]), 4)

        rows.append(row)

    capture.release()
    return rows


def summarize(stem, info, rows, kp_threshold):
    """영상 하나의 Pose 품질을 한 줄로 요약합니다."""

    conf = np.array(
        [[row[f"{name}_c"] for name in COCO_KEYPOINT_NAMES] for row in rows]
    )
    people = np.array([row["people"] for row in rows])

    frames = len(rows)
    no_person = int(np.sum(people == 0))
    multi_person = int(np.sum(people > 1))

    detected = conf[people > 0]

    if len(detected) == 0:
        mean_conf = arm_conf = arm_low_ratio = 0.0
        weakest_arm = "NO_PERSON"
    else:
        arm = detected[:, ARM_IDS]
        mean_conf = float(np.mean(detected))
        arm_conf = float(np.mean(arm))
        # 팔 관절 중 하나라도 기준 미만인 Frame 비율
        arm_low_ratio = float(np.mean(np.any(arm < kp_threshold, axis=1)))
        weakest_arm = COCO_KEYPOINT_NAMES[ARM_IDS[int(np.argmin(arm.mean(axis=0)))]]

    no_person_ratio = no_person / frames if frames else 1.0

    # ⑥ 간단한 판정 (스모크 테스트용 기준)
    if frames == 0 or no_person_ratio > 0.10 or arm_low_ratio > 0.30:
        status = "CHECK"
    else:
        status = "PASS"

    return {
        "stem": stem,
        "subject": info["subject"],
        "action": info["action"],
        "take": info["take"],
        "frames": frames,
        "no_person_ratio": round(no_person_ratio, 3),
        "multi_person_frames": multi_person,
        "mean_keypoint_conf": round(mean_conf, 4),
        "arm_mean_conf": round(arm_conf, 4),
        "arm_low_ratio": round(arm_low_ratio, 3),
        "weakest_arm_joint": weakest_arm,
        "status": status,
    }


def main() -> None:
    args = parse_args()

    # ⑦ 1번 스크립트가 만든 영상 목록 읽기
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            "먼저 python scripts/referee_01_check_videos.py 를 실행하세요."
        )

    with MANIFEST_CSV.open(encoding="utf-8-sig") as file:
        manifest = [row for row in csv.DictReader(file) if row["status"] == "OK"]

    if args.max_videos > 0:
        manifest = manifest[: args.max_videos]

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)
    columns = keypoint_columns()
    summaries = []

    # ⑧ 영상마다 추출 → CSV 저장 → 품질 요약
    for index, item in enumerate(manifest, start=1):
        stem = item["stem"]
        info = {
            "subject": item["subject"],
            "action": item["action"],
            "take": int(item["take"]),
        }
        out_csv = KEYPOINT_DIR / f"{stem}.csv"

        if out_csv.exists() and not args.overwrite:
            with out_csv.open(encoding="utf-8-sig") as file:
                rows = [
                    {k: (float(v) if k.endswith("_c") else v) for k, v in row.items()}
                    for row in csv.DictReader(file)
                ]
                for row in rows:
                    row["people"] = int(row["people"])
            print(f"[{index}/{len(manifest)}] {stem} (이미 있음, 건너뜀)")
        else:
            rows = extract_video(
                model, config, device, ROOT / item["video"], args.main_person
            )
            with out_csv.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[{index}/{len(manifest)}] {stem}  frames={len(rows)}")

        summaries.append(summarize(stem, info, rows, args.kp_conf))

    # ⑨ 영상별 품질 요약 CSV 저장
    with POSE_QUALITY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    checks = [row for row in summaries if row["status"] == "CHECK"]

    print()
    print(f"PASS: {len(summaries) - len(checks)}개 / CHECK: {len(checks)}개")
    for row in checks:
        print(
            f"  [CHECK] {row['stem']}  "
            f"no_person={row['no_person_ratio']}  "
            f"arm_low={row['arm_low_ratio']}  "
            f"weakest={row['weakest_arm_joint']}"
        )
    print()
    print(f"Keypoint CSV: {KEYPOINT_DIR}")
    print(f"Quality CSV:  {POSE_QUALITY_CSV}")


if __name__ == "__main__":
    main()