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
)


# ③ 터미널에서 영상 경로 및 실행 옵션 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    # 분석할 MP4 영상 (기본값)
    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day02"
            / "videos"
            / "pose_quality_demo.mp4"
        ),
    )

    # LOW 관절을 판단할 Confidence 기준
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    # 최대 처리 Frame 수 (0이면 영상 전체)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="처리할 최대 Frame 수. 0이면 영상 끝까지 처리",
    )

    return parser.parse_args()


def main() -> None:

    # ④ 영상 파일 확인 및 프로젝트 설정 읽기
    args = parse_args()

    source = Path(args.source)

    # 입력 영상이 존재하는지 확인
    if not source.exists():
        raise FileNotFoundError(
            f"영상이 없습니다: {source}"
        )

    # project.json 설정값 불러오기
    config = load_config()

    # GPU 또는 CPU 선택
    device = resolve_device(config["device"])

    # ⑤ YOLO Pose 모델 불러오기 및 MP4 열기
    model = YOLO(config["pose_model"])

    capture = cv2.VideoCapture(str(source))

    # 영상 연결 여부 확인
    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {source}"
        )

    # ⑥ 영상 FPS·해상도 및 첫 Frame 확인
    fps = capture.get(cv2.CAP_PROP_FPS)

    # FPS를 읽지 못하면 30으로 설정
    output_fps = fps if fps > 0 else 30.0

    # 첫 번째 Frame 읽기
    success, frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError(
            "영상의 첫 Frame을 읽지 못했습니다."
        )

    # 원본 영상의 높이와 너비 확인
    height, width = frame.shape[:2]

    # ⑦ Pose 결과 MP4 저장 준비
    output_video = (
        ROOT
        / "reports"
        / "day02"
        / "videos"
        / f"{source.stem}_pose.mp4"
    )

    # 결과 저장 폴더 생성
    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 결과 영상을 저장할 VideoWriter 생성
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "결과 MP4를 생성할 수 없습니다."
        )

    # ⑧ Frame 품질 CSV와 Keypoint CSV 저장 준비
    # Frame별 품질 요약 CSV 경로
    frame_table = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / f"{source.stem}_frame_quality.csv"
    )

    # Frame별 관절 상세 CSV 경로
    keypoint_table = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / f"{source.stem}_keypoints.csv"
    )

    # CSV 저장 폴더 생성
    frame_table.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CSV에 기록할 데이터를 담을 목록
    frame_rows = []
    keypoint_rows = []

    # Frame 번호 초기화
    frame_id = 0

    # ⑨ Frame별 YOLO Pose 추론 및 결과 영상 저장
    while success:

        # 처리 중인 Frame 번호 증가
        frame_id += 1

        # 현재 Frame에 YOLO Pose 적용
        results = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )

        # 현재 Frame의 추론 결과 가져오기
        result = results[0]

        # BBox + Keypoint + Skeleton 표시 후 MP4 저장
        writer.write(result.plot())

        # ⑩ 검출된 사람 수 및 품질 기본값 설정
        people = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        # 사람이나 관절을 찾지 못한 경우의 기본값
        mean_kp_conf = 0.0
        low_keypoints = 17
        lowest_name = "NO_PERSON"
        lowest_conf = 0.0

        # ⑪ 대표 사람의 관절 Confidence 분석
        if (
            result.boxes is not None
            and result.keypoints is not None
            and result.keypoints.xy is not None
            and result.keypoints.conf is not None
            and len(result.boxes) > 0
        ):

            # 사람별 BBox Confidence 가져오기
            box_conf = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            # 사람별 17개 관절 좌표 가져오기
            xy = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            # 사람별 관절 Confidence 가져오기
            conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

            # BBox Confidence가 가장 높은 사람 선택
            main_index = int(
                np.argmax(box_conf)
            )

            # 대표 사람의 관절 좌표 및 Confidence
            main_xy = xy[main_index]
            main_conf = conf[main_index]

            # 17개 관절 Confidence 평균 계산
            mean_kp_conf = float(
                np.mean(main_conf)
            )

            # Threshold보다 Confidence가 낮은 관절 수
            low_keypoints = int(
                np.sum(
                    main_conf < args.kp_conf
                )
            )

            # 가장 Confidence가 낮은 관절 번호
            lowest_id = int(
                np.argmin(main_conf)
            )

            # 가장 신뢰도가 낮은 관절 이름
            lowest_name = (
                COCO_KEYPOINT_NAMES[
                    lowest_id
                ]
            )

            # 해당 관절의 Confidence
            lowest_conf = float(
                main_conf[lowest_id]
            )

            # ⑫ 현재 Frame의 17개 Keypoint 상세 정보 기록
            for keypoint_id, name in enumerate(
                COCO_KEYPOINT_NAMES
            ):

                # 관절별 Pixel x, y 좌표
                x, y = main_xy[keypoint_id]

                # 해당 관절의 Confidence
                confidence = main_conf[keypoint_id]

                # 관절 하나의 정보를 CSV 목록에 추가
                keypoint_rows.append(
                    {
                        "frame_id": frame_id,

                        "time_sec": round(
                            (frame_id - 1)
                            / output_fps,
                            3,
                        ),

                        "keypoint_id": keypoint_id,
                        "keypoint_name": name,

                        "pixel_x": round(
                            float(x),
                            2,
                        ),

                        "pixel_y": round(
                            float(y),
                            2,
                        ),

                        "confidence": round(
                            float(confidence),
                            4,
                        ),
                    }
                )

        # ⑬ 현재 Frame의 Pose 품질 요약 기록
        frame_rows.append(
            {
                # Frame 번호
                "frame_id": frame_id,

                # 현재 Frame의 시간(초)
                "time_sec": round(
                    (frame_id - 1)
                    / output_fps,
                    3,
                ),

                # 검출된 사람 수
                "people": people,

                # 17개 관절 평균 Confidence
                "mean_keypoint_conf": round(
                    mean_kp_conf,
                    4,
                ),

                # 기준 미만 관절 개수
                "low_keypoint_count": (
                    low_keypoints
                ),

                # 가장 신뢰도가 낮은 관절 이름
                "lowest_keypoint": (
                    lowest_name
                ),

                # 가장 낮은 관절 Confidence
                "lowest_keypoint_conf": round(
                    lowest_conf,
                    4,
                ),

                # 이번 분석에 사용한 기준값
                "keypoint_threshold": (
                    args.kp_conf
                ),
            }
        )

        # ⑭ 진행 상황 확인 및 다음 Frame 읽기
        # 30개 Frame마다 진행 상황 출력
        if frame_id % 30 == 0:
            print(
                f"Processed frame: {frame_id}"
            )

        # 지정한 최대 Frame 수에 도달했는지 확인
        if (
            args.max_frames > 0
            and frame_id >= args.max_frames
        ):
            break

        # 다음 Frame 읽기
        success, frame = capture.read()

    # ⑮ 영상 자원 해제 및 Frame 품질 CSV 저장
    # 원본 영상 및 결과 영상 파일 닫기
    capture.release()
    writer.release()

    # Frame별 품질 요약 CSV 작성
    with frame_table.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer_csv = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "time_sec",
                "people",
                "mean_keypoint_conf",
                "low_keypoint_count",
                "lowest_keypoint",
                "lowest_keypoint_conf",
                "keypoint_threshold",
            ],
        )

        # 컬럼 이름 저장
        writer_csv.writeheader()

        # Frame별 품질 데이터 저장
        writer_csv.writerows(frame_rows)

    # ⑯ Keypoint 상세 CSV 저장 및 최종 결과 출력
    with keypoint_table.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer_csv = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "time_sec",
                "keypoint_id",
                "keypoint_name",
                "pixel_x",
                "pixel_y",
                "confidence",
            ],
        )

        # 컬럼 이름 저장
        writer_csv.writeheader()

        # Frame별 17개 관절 데이터 저장
        writer_csv.writerows(keypoint_rows)

    # 최종 처리 결과 출력
    print()
    print(f"Frames processed: {frame_id}")
    print(f"Pose video: {output_video}")
    print(f"Frame table: {frame_table}")
    print(f"Keypoint table: {keypoint_table}")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()