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

from src.common import (
    load_config,
    resolve_device,
)
from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
    confidence_state,
)


# ③ 터미널에서 이미지 경로와 Confidence 기준 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    # 분석할 이미지 지정 (기본값: front.jpg)
    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day02"
            / "images"
            / "front.jpg"
        ),
        help="분석할 이미지 파일",
    )

    # 관절을 VALID로 판정할 Confidence 기준
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
        help="VALID로 판단할 Keypoint Confidence 기준",
    )

    return parser.parse_args()


def main() -> None:

    # ④ 입력 이미지 확인 및 크기 가져오기
    args = parse_args()

    source = Path(args.source)

    # 이미지 파일 존재 여부 확인
    if not source.exists():
        raise FileNotFoundError(
            f"이미지가 없습니다: {source}"
        )

    # OpenCV로 이미지 읽기
    image = cv2.imread(str(source))

    if image is None:
        raise RuntimeError(
            f"이미지를 읽지 못했습니다: {source}"
        )

    # 원본 이미지의 높이와 너비 확인
    height, width = image.shape[:2]

    # ⑤ 프로젝트 설정 읽기 및 YOLO Pose 모델 실행
    config = load_config()
    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    # 이미지에서 사람과 17개 관절 추론
    results = model.predict(
        source=image,
        imgsz=config["image_size"],
        conf=config["person_confidence"],
        device=device,
        verbose=False,
    )

    # 첫 번째 이미지의 추론 결과 가져오기
    result = results[0]

    # ⑥ 사람과 Keypoint 검출 여부 확인
    if (
        result.boxes is None
        or result.keypoints is None
        or result.keypoints.xy is None
        or len(result.boxes) == 0
    ):
        print("사람 또는 Keypoint를 찾지 못했습니다.")
        return

    # ⑦ BBox 좌표·신뢰도와 관절 데이터 추출
    # 사람의 Bounding Box 좌표
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()

    # 사람 검출 Confidence
    boxes_conf = result.boxes.conf.cpu().numpy()

    # 17개 관절의 Pixel x, y 좌표
    xy = result.keypoints.xy.cpu().numpy()

    # 관절별 Confidence 가져오기
    if result.keypoints.conf is not None:
        kp_conf = result.keypoints.conf.cpu().numpy()
    else:
        # 관절 Confidence 정보가 없을 때 대체값 사용
        kp_conf = np.ones(
            xy.shape[:2],
            dtype=float,
        )

    # ⑧ BBox와 Skeleton이 표시된 결과 이미지 저장
    output_image = (
        ROOT
        / "reports"
        / "day02"
        / "images"
        / f"{source.stem}_pose.jpg"
    )

    # 결과 저장 폴더 생성
    output_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # YOLO Pose 결과를 이미지로 저장
    saved = cv2.imwrite(
        str(output_image),
        result.plot(),
    )

    if not saved:
        raise RuntimeError(
            f"결과 이미지를 저장하지 못했습니다: {output_image}"
        )

    # ⑨ CSV 저장 경로와 컬럼 이름 준비
    output_csv = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / f"{source.stem}_keypoints.csv"
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CSV에 저장할 데이터 항목 정의
    fieldnames = [
        "person_id",
        "box_conf",
        "keypoint_id",
        "keypoint_name",
        "pixel_x",
        "pixel_y",
        "normalized_x",
        "normalized_y",
        "keypoint_conf",
        "quality",
    ]

    # CSV에 기록할 데이터 목록
    rows = []

    # ⑩ 검출된 사람별 BBox 정보 확인
    person_count = min(
        len(boxes_xyxy),
        len(xy),
    )

    print(f"Detected people: {person_count}")
    print(f"Image size: {width} x {height}")

    # 사람을 한 명씩 처리
    for person_index in range(person_count):

        # 해당 사람의 BBox 좌표
        x1, y1, x2, y2 = boxes_xyxy[person_index]

        print()
        print(f"Person {person_index + 1}")

        print(
            "BBox:",
            f"({x1:.1f}, {y1:.1f})",
            f"({x2:.1f}, {y2:.1f})",
        )

        print(
            f"BBox Confidence: "
            f"{boxes_conf[person_index]:.3f}"
        )

        # ⑪ 관절별 좌표 정규화 및 Confidence 품질 판정
        # 한 사람의 17개 Keypoint를 순서대로 처리
        for keypoint_id, name in enumerate(
            COCO_KEYPOINT_NAMES
        ):

            # 관절의 원본 Pixel 좌표
            x, y = xy[
                person_index,
                keypoint_id,
            ]

            # 해당 관절의 Confidence
            confidence = kp_conf[
                person_index,
                keypoint_id,
            ]

            # Pixel x 좌표를 이미지 너비로 나누기
            normalized_x = (
                float(x) / width
                if width > 0
                else 0.0
            )

            # Pixel y 좌표를 이미지 높이로 나누기
            normalized_y = (
                float(y) / height
                if height > 0
                else 0.0
            )

            # Confidence 기준으로 관절 품질 구분
            quality = confidence_state(
                float(confidence),
                valid_threshold=args.kp_conf,
                warning_threshold=0.30,
            )

            # CSV 한 행에 저장할 관절 정보
            row = {
                "person_id": person_index + 1,
                "box_conf": round(
                    float(boxes_conf[person_index]),
                    4,
                ),
                "keypoint_id": keypoint_id,
                "keypoint_name": name,
                "pixel_x": round(float(x), 2),
                "pixel_y": round(float(y), 2),
                "normalized_x": round(
                    normalized_x,
                    5,
                ),
                "normalized_y": round(
                    normalized_y,
                    5,
                ),
                "keypoint_conf": round(
                    float(confidence),
                    4,
                ),
                "quality": quality,
            }

            # CSV 저장용 목록에 관절 정보 추가
            rows.append(row)

            # 터미널에 관절 분석 결과 출력
            print(
                f"{keypoint_id:02d} "
                f"{name:16s} "
                f"pixel=({x:7.1f}, {y:7.1f}) "
                f"norm=({normalized_x:.4f}, {normalized_y:.4f}) "
                f"conf={confidence:.3f} "
                f"{quality}"
            )

    # ⑫ 관절 분석 결과를 CSV 파일로 저장
    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        # CSV 첫 줄에 컬럼 이름 저장
        writer.writeheader()

        # 사람별·관절별 데이터 저장
        writer.writerows(rows)

    # 최종 결과 파일 경로 출력
    print()
    print(f"Pose image: {output_image}")
    print(f"Keypoint CSV: {output_csv}")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()