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
from src.pose_utils import (
    COCO_SKELETON,
)


# ③ 분석할 이미지와 Keypoint Confidence 기준 설정
def parse_args():
    parser = argparse.ArgumentParser()

    # 분석할 이미지 경로 (기본값: front.jpg)
    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day02"
            / "images"
            / "front.jpg"
        ),
    )

    # 관절을 표시할 최소 Confidence (기본값: 0.50)
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


def main() -> None:

    # ④ 이미지 파일 확인 및 OpenCV로 읽기
    args = parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(
            f"이미지가 없습니다: {source}"
        )

    image = cv2.imread(str(source))

    if image is None:
        raise RuntimeError(
            f"이미지를 읽지 못했습니다: {source}"
        )

    # ⑤ 프로젝트 설정 읽기 및 YOLO Pose 추론
    config = load_config()
    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    results = model.predict(
        source=image,
        imgsz=config["image_size"],
        conf=config["person_confidence"],
        device=device,
        verbose=False,
    )

    result = results[0]

    # ⑥ Keypoint 검출 여부 확인
    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        print("Keypoint를 찾지 못했습니다.")
        return

    # ⑦ 관절 좌표 및 Confidence 가져오기
    # 관절별 Pixel x, y 좌표
    xy = result.keypoints.xy.cpu().numpy()

    # 관절별 Confidence
    if result.keypoints.conf is not None:
        confidence = (
            result.keypoints.conf
            .cpu()
            .numpy()
        )
    else:
        # Confidence 정보가 없으면 대체값 1.0 사용
        confidence = np.ones(
            xy.shape[:2],
            dtype=float,
        )

    # 원본 이미지 복사 (그림을 그릴 이미지)
    output = image.copy()

    # ⑧ 사람의 Bounding Box 그리기
    if result.boxes is not None:
        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        # 검출된 사람의 BBox를 하나씩 그림
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

    # ⑨ Confidence 기준을 적용하여 Skeleton 선 그리기
    for person_index in range(len(xy)):
        points = xy[person_index]
        confs = confidence[person_index]

        # COCO_SKELETON에 정의된 관절 연결 관계 확인
        for start_id, end_id in COCO_SKELETON:

            # 두 관절 모두 Confidence 기준 이상인가?
            if (
                confs[start_id] >= args.kp_conf
                and confs[end_id] >= args.kp_conf
            ):

                # 연결할 두 관절의 좌표 가져오기
                start = tuple(
                    points[start_id].astype(int)
                )

                end = tuple(
                    points[end_id].astype(int)
                )

                # 두 관절을 초록색 선으로 연결
                cv2.line(
                    output,
                    start,
                    end,
                    (0, 255, 0),
                    2,
                )

        # ⑩ Confidence 기준 이상인 관절 점과 번호 표시
        for keypoint_id, (point, conf) in enumerate(
            zip(points, confs)
        ):

            # 기준보다 Confidence가 낮으면 표시하지 않음
            if conf < args.kp_conf:
                continue

            # 관절의 Pixel 좌표
            x, y = point.astype(int)

            # 관절 위치에 노란색 점 표시
            cv2.circle(
                output,
                (x, y),
                5,
                (0, 255, 255),
                -1,
            )

            # 관절 번호(0~16)를 점 옆에 표시
            cv2.putText(
                output,
                str(keypoint_id),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    # ⑪ Confidence 기준을 파일명에 넣어 결과 이미지 저장
    threshold_name = int(
        args.kp_conf * 100
    )

    output_path = (
        ROOT
        / "reports"
        / "day02"
        / "images"
        / (
            f"{source.stem}_"
            f"skeleton_conf{threshold_name:02d}.jpg"
        )
    )

    # 저장할 폴더 생성
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Skeleton 결과 이미지 저장
    saved = cv2.imwrite(
        str(output_path),
        output,
    )

    if not saved:
        raise RuntimeError(
            f"결과를 저장하지 못했습니다: {output_path}"
        )

    print(f"Saved: {output_path}")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()