from __future__ import annotations

# ① 필요한 라이브러리 불러오기
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
)


# ③ 분석할 이미지 파일 확장자 지정
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def main() -> None:

    # ④ 프로젝트 설정 및 Keypoint Confidence 기준 읽기
    config = load_config()

    # GPU 또는 CPU 선택
    device = resolve_device(config["device"])

    # 관절 신뢰도 기준 (기본값 0.50)
    kp_threshold = config.get(
        "keypoint_confidence",
        0.50,
    )

    # ⑤ 입력 이미지 및 결과 저장 폴더 지정
    # 2일차 이미지가 저장된 폴더
    image_dir = (
        ROOT
        / "data"
        / "day02"
        / "images"
    )

    # Pose 결과 이미지를 저장할 폴더
    output_dir = (
        ROOT
        / "reports"
        / "day02"
        / "images"
        / "comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 전체 이미지의 품질 비교표 CSV 경로
    table_path = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / "image_quality_summary.csv"
    )

    table_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ⑥ 분석할 이미지 목록 가져오기
    # 지정 폴더에서 JPG, JPEG, PNG 파일 찾기
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    # 이미지가 없으면 오류 발생
    if not image_paths:
        raise RuntimeError(
            f"이미지가 없습니다: {image_dir}"
        )

    # ⑦ YOLO Pose 모델 불러오기
    model = YOLO(config["pose_model"])

    # 이미지별 분석 결과를 저장할 목록
    rows = []

    # ⑧ 이미지를 한 장씩 읽고 크기 확인
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        # 이미지를 읽지 못하면 건너뛰기
        if image is None:
            print(f"Skip: {image_path}")
            continue

        # 이미지의 너비, 높이, 전체 면적 계산
        height, width = image.shape[:2]
        image_area = width * height

        # ⑨ 현재 이미지에 YOLO Pose 적용
        results = model.predict(
            source=image,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )

        result = results[0]

        # 검출된 사람 수 확인
        people = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        # 분석 결과의 기본값 설정
        main_box_conf = 0.0
        bbox_area_ratio = 0.0
        mean_kp_conf = 0.0
        low_keypoints = 17
        lowest_name = "NO_PERSON"
        lowest_conf = 0.0

        # ⑩ 대표 사람의 BBox와 관절 품질 분석
        if (
            result.boxes is not None
            and result.keypoints is not None
            and result.keypoints.conf is not None
            and len(result.boxes) > 0
        ):

            # 사람별 BBox Confidence 가져오기
            box_conf = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            # 사람별 BBox 좌표 가져오기
            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            # 사람별 17개 관절 Confidence 가져오기
            kp_conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

            # BBox Confidence가 가장 높은 사람 선택
            main_index = int(
                np.argmax(box_conf)
            )

            # 대표 사람의 BBox Confidence
            main_box_conf = float(
                box_conf[main_index]
            )

            # 대표 사람의 BBox 좌표
            x1, y1, x2, y2 = boxes[main_index]

            # BBox 면적 계산
            bbox_area = max(
                0.0,
                float(x2 - x1),
            ) * max(
                0.0,
                float(y2 - y1),
            )

            # BBox 면적 / 전체 이미지 면적
            # 이미지에서 사람이 차지하는 비율
            if image_area > 0:
                bbox_area_ratio = (
                    bbox_area / image_area
                )

            # 대표 사람의 관절 Confidence
            conf = kp_conf[main_index]

            # 17개 관절의 평균 Confidence
            mean_kp_conf = float(
                np.mean(conf)
            )

            # 기준값보다 Confidence가 낮은 관절 수
            low_keypoints = int(
                np.sum(
                    conf < kp_threshold
                )
            )

            # Confidence가 가장 낮은 관절 번호
            lowest_id = int(
                np.argmin(conf)
            )

            # 해당 관절의 이름 확인
            lowest_name = (
                COCO_KEYPOINT_NAMES[
                    lowest_id
                ]
            )

            # 가장 낮은 관절 Confidence
            lowest_conf = float(
                conf[lowest_id]
            )

        # ⑪ BBox와 Skeleton 결과 이미지 저장
        output_path = (
            output_dir
            / f"{image_path.stem}_pose.jpg"
        )

        saved = cv2.imwrite(
            str(output_path),
            result.plot(),
        )

        if not saved:
            raise RuntimeError(
                f"결과 이미지를 저장하지 못했습니다: {output_path}"
            )

        # ⑫ 현재 이미지의 분석 결과를 비교표에 추가
        rows.append(
            {
                "image": image_path.name,

                "people": people,

                "main_box_conf": round(
                    main_box_conf,
                    4,
                ),

                "bbox_area_ratio": round(
                    bbox_area_ratio,
                    4,
                ),

                "mean_keypoint_conf": round(
                    mean_kp_conf,
                    4,
                ),

                "low_keypoint_count": (
                    low_keypoints
                ),

                "lowest_keypoint": (
                    lowest_name
                ),

                "lowest_keypoint_conf": round(
                    lowest_conf,
                    4,
                ),

                "keypoint_threshold": (
                    kp_threshold
                ),
            }
        )

        # 이미지별 분석 결과를 터미널에 출력
        print(
            image_path.name,
            "people=",
            people,
            "bbox_ratio=",
            f"{bbox_area_ratio:.3f}",
            "mean_kp=",
            f"{mean_kp_conf:.3f}",
            "low_kp=",
            low_keypoints,
        )

    # ⑬ 전체 이미지의 품질 비교 결과를 CSV로 저장
    with table_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        # CSV 컬럼 이름 지정
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "people",
                "main_box_conf",
                "bbox_area_ratio",
                "mean_keypoint_conf",
                "low_keypoint_count",
                "lowest_keypoint",
                "lowest_keypoint_conf",
                "keypoint_threshold",
            ],
        )

        # CSV 첫 줄에 컬럼 이름 저장
        writer.writeheader()

        # 이미지별 분석 결과 저장
        writer.writerows(rows)

    print()
    print(f"Summary saved: {table_path}")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()