from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
)


# ③ 터미널에서 영상 경로와 처리할 Frame 수 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Pose를 적용할 MP4 파일",
    )

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

    # 영상 파일이 존재하는지 확인
    if not source.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {source}"
        )

    # project.json 설정값 불러오기
    config = load_config()

    # GPU 또는 CPU 선택
    device = resolve_device(config["device"])

    # ⑤ YOLO Pose 모델 불러오기
    model = YOLO(config["pose_model"])

    # ⑥ MP4 영상 열기 및 영상 정보 확인
    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {source}"
        )

    # 원본 영상의 FPS 확인
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

    # 영상의 높이와 너비 확인
    height, width = frame.shape[:2]

    # ⑦ Pose 결과 영상 저장 준비
    output_path = (
        ROOT
        / "reports"
        / "day01"
        / "pose_demo_output.mp4"
    )

    # 저장 폴더가 없으면 생성
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # MP4 영상 저장에 사용할 코덱 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 원본 해상도와 FPS를 기준으로 출력 영상 생성
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "결과 MP4 파일을 생성할 수 없습니다."
        )

    # ⑧ Frame별 YOLO Pose 추론 및 결과 저장
    frame_number = 0

    while success:

        # 처리한 Frame 수 증가
        frame_number += 1

        # 현재 Frame에 YOLO Pose 적용
        results = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )

        # BBox + Keypoint + Skeleton 표시
        plotted = results[0].plot()

        # Pose가 표시된 Frame을 결과 영상에 저장
        writer.write(plotted)

        # ⑨ 최대 Frame 수 확인 및 다음 Frame 읽기
        if (
            args.max_frames > 0
            and frame_number >= args.max_frames
        ):
            break

        # 다음 Frame 읽기
        success, frame = capture.read()

    # ⑩ 영상 자원 해제 및 최종 결과 출력
    capture.release()
    writer.release()

    print(f"Processed frames: {frame_number}")
    print(f"Saved: {output_path}")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()