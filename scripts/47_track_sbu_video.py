# =====================================================
# ① 실습에 필요한 라이브러리 준비
# =====================================================
# argparse는 실행 옵션을 받습니다. csv는 추적 로그를 저장하고, sys와 Path는 경로를 관리합니다.
# OpenCV는 영상 입출력과 글자 표시, NumPy는 Keypoint 신뢰도의 평균 계산에 사용합니다.
# YOLO는 Pose 검출과 여러 사람의 Tracking을 담당합니다.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =====================================================
# ② 프로젝트 ROOT와 모듈 검색 경로 설정
# =====================================================
# 이 프로그램이 scripts 폴더에 있다는 전제로 상위 프로젝트 폴더를 ROOT로 정합니다.
# sys.path.insert()로 ROOT를 추가하여 src 폴더의 공통 모듈을 가져올 수 있게 합니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


# =====================================================
# ③ 프로젝트 공통 설정 함수 불러오기
# =====================================================
# load_config()는 프로젝트 공통 설정을 읽고, resolve_device()는 설정값에 맞는 CPU/GPU 실행 장치를 결정합니다.

from src.common import load_config, resolve_device


# =====================================================
# ④ 결과 MP4와 CSV의 기본 저장 폴더 지정
# =====================================================
# 추적 결과 영상은 reports/day08/tracking에, 사람별 추적 기록은 reports/day08/logs에 저장하도록 기본 경로를 정의합니다.

DEFAULT_OUTPUT_VIDEO_DIR = (
    ROOT / "reports" / "day08" / "tracking"
)

DEFAULT_OUTPUT_LOG_DIR = (
    ROOT / "reports" / "day08" / "logs"
)


# =====================================================
# ⑤ 터미널 실행 옵션을 받는 함수 정의
# =====================================================
# argparse를 이용하면 원본 MP4와 추적기, 저장 폴더를 명령어에서 선택할 수 있습니다.

def parse_args():
    parser = argparse.ArgumentParser()

    # =====================================================
    # ⑥ 입력 영상과 Tracker 옵션 설정
    # =====================================================
    # --source는 필수 입력 MP4 경로입니다.
    # --tracker의 기본값은 bytetrack.yaml로, 사람에게 Track ID를 배정하고 연결할 추적기 설정입니다.

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
    )

    # =====================================================
    # ⑦ 결과 영상·로그 폴더 옵션 설정
    # =====================================================
    # --output-video-dir와 --output-log-dir 옵션으로 저장 위치를 바꿀 수 있습니다.
    # 옵션을 생략하면 위에서 정의한 기본 경로를 사용하고 parse_args()가 값을 돌려줍니다.

    parser.add_argument(
        "--output-video-dir",
        default=str(
            DEFAULT_OUTPUT_VIDEO_DIR
        ),
    )

    parser.add_argument(
        "--output-log-dir",
        default=str(
            DEFAULT_OUTPUT_LOG_DIR
        ),
    )

    return parser.parse_args()


# =====================================================
# ⑧ main 함수 시작 및 실행 옵션 읽기
# =====================================================
# main()은 입력 검증부터 영상 추적, 결과 저장까지 실행합니다.
# parse_args()로 사용자가 지정한 옵션을 먼저 읽습니다.

def main() -> None:
    args = parse_args()

    # =====================================================
    # ⑨ 입력 MP4의 경로 확인과 존재 검사
    # =====================================================
    # 입력 경로가 상대경로라면 ROOT를 앞에 붙입니다.
    # 파일이 없으면 FileNotFoundError를 발생시켜 실습을 중단합니다.

    source = Path(args.source)

    if not source.is_absolute():
        source = ROOT / source

    if not source.exists():
        raise FileNotFoundError(source)

    # =====================================================
    # ⑩ 출력 경로를 Path로 변환하고 ROOT 기준으로 정리
    # =====================================================
    # 영상·로그 폴더를 각각 Path 객체로 만들고, 상대경로인 경우 ROOT를 연결합니다.
    # 입력과 출력 경로를 명확히 나누어 저장 위치를 예측 가능하게 합니다.

    output_video_dir = Path(
        args.output_video_dir
    )
    output_log_dir = Path(
        args.output_log_dir
    )

    if not output_video_dir.is_absolute():
        output_video_dir = (
            ROOT / output_video_dir
        )

    if not output_log_dir.is_absolute():
        output_log_dir = (
            ROOT / output_log_dir
        )

    # =====================================================
    # ⑪ 설정값으로 CPU/GPU 및 YOLO Pose 모델 준비
    # =====================================================
    # 프로젝트 공통 설정에서 device와 pose_model을 읽습니다.
    # resolve_device()로 실행 장치를 확정하고 YOLO Pose 모델을 메모리에 불러옵니다.
    # 이 스크립트는 6일차 Random Forest Action 모델을 사용하지 않습니다.

    config = load_config()
    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    # =====================================================
    # ⑫ OpenCV로 원본 MP4 열기
    # =====================================================
    # VideoCapture가 원본 영상을 열 수 있는지 isOpened()로 확인합니다.
    # 열리지 않으면 즉시 오류를 발생시킵니다.

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {source}"
        )

    # =====================================================
    # ⑬ 원본 영상의 FPS와 해상도 읽기
    # =====================================================
    # FPS는 초당 Frame 수이며, width·height는 한 Frame의 가로·세로 픽셀 수입니다.
    # 결과 MP4를 원본과 같은 FPS 및 해상도로 저장하기 위해 필요합니다.

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # =====================================================
    # ⑭ FPS 및 해상도의 유효성 검사
    # =====================================================
    # FPS가 0 이하이면 시간 계산을 할 수 없습니다.
    # 너비 또는 높이가 0 이하이면 VideoWriter의 영상 크기를 지정할 수 없습니다.
    # 오류 발생 전 capture를 해제합니다.

    if fps <= 0:
        capture.release()
        raise RuntimeError("영상 FPS를 확인할 수 없습니다.")

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("영상 해상도를 확인할 수 없습니다.")

    # =====================================================
    # ⑮ 결과 저장 폴더 생성
    # =====================================================
    # parents=True는 중간 폴더도 생성하고 exist_ok=True는 이미 폴더가 있어도 오류를 내지 않습니다.

    output_video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # ⑯ 입력 파일명으로 결과 MP4 및 CSV 이름 만들기
    # =====================================================
    # source.stem은 확장자를 제외한 파일명입니다.
    # 예를 들어 01_sbu_shaking_hands.mp4를 입력하면 _tracked.mp4와 _tracking.csv를 만듭니다.
    # 같은 이름으로 다시 실행하면 결과 파일을 덮어쓸 수 있습니다.

    output_video = (
        output_video_dir
        / f"{source.stem}_tracked.mp4"
    )

    output_log = (
        output_log_dir
        / f"{source.stem}_tracking.csv"
    )

    # =====================================================
    # ⑰ 추적 결과 MP4를 기록할 VideoWriter 생성
    # =====================================================
    # mp4v 코덱, 원본 FPS, 원본 해상도로 저장 객체를 생성합니다.
    # 현재 코드에는 원본 Frame을 화면에 띄우는 cv2.imshow()가 없으며, 결과를 파일로 저장합니다.

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # =====================================================
    # ⑱ 결과 영상 저장 가능 여부 확인
    # =====================================================
    # VideoWriter가 열리지 않으면 영상을 기록할 수 없습니다.
    # 입력 capture를 해제한 후 RuntimeError로 종료합니다.

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"결과 영상을 저장할 수 없습니다: {output_video}"
        )

    # =====================================================
    # ⑲ CSV 로그에 기록할 컬럼 정의
    # =====================================================
    # frame_id·time_sec·person_count로 Frame 정보를 기록합니다.
    # track_id는 추적된 사람의 ID, box_conf는 사람 검출 신뢰도, bbox_x1~y2는 Bounding Box의 두 꼭짓점입니다.
    # mean_keypoint_conf는 해당 사람 관절 신뢰도의 평균입니다.

    fieldnames = [
        "frame_id",
        "time_sec",
        "person_count",
        "track_id",
        "box_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mean_keypoint_conf",
    ]

    # =====================================================
    # ⑳ 로그 목록과 Frame 번호 초기화
    # =====================================================
    # rows에 사람별 추적 결과를 누적합니다.
    # frame_id=0부터 시작하며 하나의 Frame에 여러 사람이 있으면 CSV 행도 여러 개 생깁니다.

    rows = []
    frame_id = 0

    # =====================================================
    # ㉑ 영상 Frame을 끝까지 반복해서 읽기
    # =====================================================
    # capture.read()로 한 장씩 읽습니다.
    # 영상이 끝나거나 Frame을 읽지 못하면 반복문을 종료합니다.

    while True:
        success, frame = capture.read()

        if not success:
            break

        # =====================================================
        # ㉒ YOLO Pose + ByteTrack으로 사람별 ID 추적
        # =====================================================
        # model.track()은 각 Frame의 사람을 검출하고 Pose 및 Track ID를 추정합니다.
        # persist=True는 연속된 Frame을 처리할 때 추적기 상태를 이어 가게 합니다.
        # tracker는 bytetrack.yaml 기본값이며 conf·imgsz·device는 프로젝트 설정을 사용합니다.
        # Track ID는 추적기 출력값이므로 영상 전체에서 항상 유지된다고 보장되지 않습니다.

        result = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            imgsz=int(config["image_size"]),
            conf=float(config["person_confidence"]),
            device=device,
            verbose=False,
        )[0]

        # =====================================================
        # ㉓ 추적 영상과 사람·관절 검출 결과 분리
        # =====================================================
        # result.plot()으로 Bounding Box·Pose 등이 그려진 Frame을 만듭니다.
        # boxes와 keypoints를 분리하여 사람 수와 신뢰도·좌표를 기록할 준비를 합니다.
        # person_count는 검출 박스 수이며 실제 추적 ID가 있는 사람 수와 다를 수 있습니다.

        annotated = result.plot()
        boxes = result.boxes
        keypoints = result.keypoints

        person_count = len(boxes) if boxes is not None else 0
        tracked_rows = 0

        # =====================================================
        # ㉔ 추적 ID가 있는 사람만 상세 로그 대상 선정
        # =====================================================
        # 검출 박스가 하나 이상 있고 boxes.id가 있을 때 사람별 상세 정보를 추출합니다.
        # Track ID가 아직 배정되지 않았다면 아래의 -1 표시 행에서 기록합니다.

        if (
            boxes is not None
            and len(boxes) > 0
            and boxes.id is not None
        ):
            # =====================================================
            # ㉕ Bounding Box·신뢰도·Track ID를 NumPy 배열로 변환
            # =====================================================
            # YOLO의 Tensor 값을 detach()→cpu()→numpy() 순서로 변환합니다.
            # xyxy는 (x1,y1,x2,y2), box_conf는 박스별 검출 신뢰도, track_ids는 정수 ID입니다.

            xyxy = boxes.xyxy.detach().cpu().numpy()
            box_conf = boxes.conf.detach().cpu().numpy()
            track_ids = (
                boxes.id.detach().cpu().numpy().astype(int)
            )

            # =====================================================
            # ㉖ Keypoint 신뢰도가 있을 때만 읽기
            # =====================================================
            # Keypoint 결과에 conf가 있으면 각 사람의 관절 신뢰도를 NumPy 배열로 바꿉니다.
            # 없을 때는 None으로 두어 평균 계산을 건너뛰게 합니다.

            kp_conf_all = None

            if (
                keypoints is not None
                and keypoints.conf is not None
            ):
                kp_conf_all = (
                    keypoints.conf.detach().cpu().numpy()
                )

            # =====================================================
            # ㉗ 사람별 로그 반복 개수 정하기
            # =====================================================
            # Track ID·Bounding Box·박스 Confidence 배열의 길이 중 가장 작은 값을 사용합니다.
            # 배열 길이 불일치로 인덱스 오류가 나는 상황을 줄입니다.

            item_count = min(
                len(track_ids),
                len(xyxy),
                len(box_conf),
            )

            for index in range(item_count):
                x1, y1, x2, y2 = xyxy[index]
                track_id = int(track_ids[index])

                # =====================================================
                # ㉘ 각 사람의 평균 Keypoint Confidence 계산
                # =====================================================
                # 해당 사람의 관절 Confidence가 있으면 np.mean()으로 평균을 구해 소수 넷째 자리까지 저장합니다.
                # 관절 Confidence가 없다면 빈 문자열로 둡니다.
                # 이 값은 관절 위치 검출 신뢰도이지 Action 분류 Confidence가 아닙니다.

                mean_kp_conf = ""

                if (
                    kp_conf_all is not None
                    and index < len(kp_conf_all)
                ):
                    mean_kp_conf = round(
                        float(np.mean(kp_conf_all[index])),
                        4,
                    )

                # =====================================================
                # ㉙ Frame·사람별 Tracking 결과를 CSV 행으로 기록
                # =====================================================
                # 한 사람당 한 행을 rows에 추가합니다.
                # frame_id/fps는 영상의 초 단위 위치이고 bbox 좌표는 픽셀 기준입니다.
                # 동일 Frame에 세 명이 추적되면 같은 frame_id로 세 행이 기록됩니다.

                rows.append(
                    {
                        "frame_id": frame_id,
                        "time_sec": round(frame_id / fps, 4),
                        "person_count": person_count,
                        "track_id": track_id,
                        "box_conf": round(
                            float(box_conf[index]),
                            4,
                        ),
                        "bbox_x1": round(float(x1), 2),
                        "bbox_y1": round(float(y1), 2),
                        "bbox_x2": round(float(x2), 2),
                        "bbox_y2": round(float(y2), 2),
                        "mean_keypoint_conf": mean_kp_conf,
                    }
                )

                tracked_rows += 1

                # =====================================================
                # ㉚ 영상에 사람별 Track ID 글자 표시
                # =====================================================
                # Bounding Box의 왼쪽 위 부근에 Track ID를 글자로 추가합니다.
                # max(25, y1-10)은 글자가 영상의 위쪽 경계를 벗어나지 않도록 최소 높이를 보장합니다.

                cv2.putText(
                    annotated,
                    f"Track ID {track_id}",
                    (
                        int(x1),
                        max(25, int(y1) - 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # =====================================================
        # ㉛ 추적 ID가 없는 Frame도 빠짐없이 로그에 남기기
        # =====================================================
        # 이번 Frame에서 ID가 있는 사람을 한 명도 기록하지 못하면 track_id=-1인 행을 생성합니다.
        # person_count는 Box 검출 수일 수 있으므로 -1이 반드시 사람 0명을 뜻하지는 않습니다.
        # 박스·Keypoint 필드는 빈 문자열로 저장합니다.

        if tracked_rows == 0:
            rows.append(
                {
                    "frame_id": frame_id,
                    "time_sec": round(frame_id / fps, 4),
                    "person_count": person_count,
                    "track_id": -1,
                    "box_conf": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "bbox_x2": "",
                    "bbox_y2": "",
                    "mean_keypoint_conf": "",
                }
            )

        # =====================================================
        # ㉜ 추적 결과가 표시된 Frame을 MP4에 저장
        # =====================================================
        # YOLO가 표시한 Skeleton·Bounding Box와 추가 Track ID 텍스트를 결과 영상에 한 Frame씩 기록합니다.

        writer.write(annotated)

        # =====================================================
        # ㉝ 10 Frame마다 현재 Track ID를 터미널에 출력
        # =====================================================
        # frame_id가 10의 배수일 때 현재 Track ID 목록을 출력합니다.
        # 초기 Frame 0도 출력 대상이며, IDs=[]는 해당 Frame에서 확인된 ID가 없다는 뜻입니다.

        if frame_id % 10 == 0:
            current_ids = []

            if (
                boxes is not None
                and boxes.id is not None
            ):
                current_ids = (
                    boxes.id
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )

            print(
                f"Frame {frame_id:3d} "
                f"IDs={current_ids}"
            )

        # =====================================================
        # ㉞ 다음 Frame으로 이동
        # =====================================================
        # 현재 Frame의 영상 저장과 로그 기록을 마친 뒤 번호를 하나 올립니다.
        # 반복문 처음으로 돌아가 다음 Frame을 읽습니다.

        frame_id += 1

    # =====================================================
    # ㉟ 입력 영상과 결과 영상 저장 객체 닫기
    # =====================================================
    # VideoCapture와 VideoWriter를 닫아 파일이 올바르게 마무리되도록 합니다.
    # 정상적인 반복 종료 후에 실행됩니다.

    capture.release()
    writer.release()

    # =====================================================
    # ㊱ 한 Frame도 처리하지 않았는지 검사
    # =====================================================
    # 처리된 Frame 수가 0이면 결과를 분석할 수 없으므로 RuntimeError를 발생시킵니다.

    if frame_id == 0:
        raise RuntimeError("처리한 Frame이 없습니다.")

    # =====================================================
    # ㊲ 추적 결과 CSV 헤더와 모든 행 저장
    # =====================================================
    # CSV 파일을 UTF-8-SIG로 열고, fieldnames 순서로 헤더와 rows를 저장합니다.
    # 같은 파일명이 있으면 CSV는 덮어씁니다.

    with output_log.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    # =====================================================
    # ㊳ 처리 Frame 수와 결과 파일 경로 출력
    # =====================================================
    # 완료 후 총 처리 Frame 수, 결과 MP4 경로, CSV 경로를 화면에 출력합니다.

    print()
    print(f"Frames: {frame_id}")
    print(f"Video : {output_video}")
    print(f"Log   : {output_log}")


# =====================================================
# ㊴ 직접 실행 시 main 함수 호출
# =====================================================
# python scripts/47_track_multi_person.py로 직접 실행하면 main()을 호출합니다.
# 다른 Python 파일에서 import할 때는 자동 실행하지 않습니다.

if __name__ == "__main__":
    main()