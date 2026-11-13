from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
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


# ③ COCO 17개 Keypoint의 이름 정의
# YOLO가 추정한 관절 좌표를 이름과 연결할 때 사용
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# ④ 터미널에서 이미지 경로 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Pose를 적용할 이미지 파일",
    )

    return parser.parse_args()


def main() -> None:

    # ⑤ 이미지 파일 확인 및 공통 설정 읽기
    args = parse_args()

    source = Path(args.source)

    # 이미지 파일이 존재하는지 확인
    if not source.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {source}"
        )

    # project.json의 설정값 불러오기
    config = load_config()

    # GPU 또는 CPU 선택
    device = resolve_device(config["device"])

    # ⑥ YOLO Pose 모델 불러오기 및 추론 실행
    model = YOLO(config["pose_model"])

    results = model.predict(
        source=str(source),
        imgsz=config["image_size"],
        conf=config["person_confidence"],
        device=device,
        verbose=False,
    )

    # 첫 번째 이미지의 추론 결과 가져오기
    result = results[0]

    # 검출된 사람 수 확인
    person_count = (
        len(result.boxes)
        if result.boxes is not None
        else 0
    )

    print(f"Detected people: {person_count}")

    # ⑦ BBox + Keypoint + Skeleton 결과 이미지 저장
    # YOLO 추론 결과를 원본 이미지 위에 표시
    output_image = result.plot()

    # 결과 이미지 저장 경로 지정
    output_path = (
        ROOT
        / "reports"
        / "day01"
        / f"pose_{source.stem}.jpg"
    )

    # 저장 폴더가 없으면 생성
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 이미지 파일로 저장
    saved = cv2.imwrite(
        str(output_path),
        output_image,
    )

    if not saved:
        raise RuntimeError(
            f"결과 이미지를 저장하지 못했습니다: {output_path}"
        )

    print(f"Saved image: {output_path}")

    # ⑧ Keypoint 검출 여부 확인 및 좌표 데이터 가져오기
    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or person_count == 0
    ):
        print("Keypoint가 검출되지 않았습니다.")
        return

    # 관절의 x, y 좌표를 NumPy 배열로 변환
    xy = result.keypoints.xy.cpu().numpy()

    # 관절별 Confidence 가져오기
    confidence = result.keypoints.conf

    if confidence is not None:
        confidence = confidence.cpu().numpy()
    else:
        # Confidence 정보가 없을 때 사용할 대체값
        confidence = np.ones(
            xy.shape[:2],
            dtype=float,
        )

    # ⑨ 사람별 17개 Keypoint의 좌표와 Confidence 출력
    report_lines = []

    # 검출된 사람을 한 명씩 처리
    for person_index in range(len(xy)):
        title = f"Person {person_index + 1}"

        print(f"\n{title}")
        report_lines.append(title)

        # 한 사람의 17개 관절을 순서대로 확인
        for keypoint_index, name in enumerate(
            COCO_KEYPOINT_NAMES
        ):
            # 관절의 x, y 좌표 가져오기
            x, y = xy[
                person_index,
                keypoint_index,
            ]

            # 해당 관절의 Confidence 가져오기
            conf = confidence[
                person_index,
                keypoint_index,
            ]

            # 출력할 문자열 만들기
            line = (
                f"{keypoint_index:02d} "
                f"{name:16s} "
                f"x={x:8.2f} "
                f"y={y:8.2f} "
                f"conf={conf:.3f}"
            )

            print(line)
            report_lines.append(line)

    # ⑩ Keypoint 분석 결과를 TXT 보고서로 저장
    report_path = (
        ROOT
        / "reports"
        / "day01"
        / f"pose_{source.stem}.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"\nKeypoint report: {report_path}")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()