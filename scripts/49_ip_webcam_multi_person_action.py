from __future__ import annotations

# ==========================================================
# ① 실시간 다중 인물 추론에 필요한 라이브러리 준비
# ==========================================================
# argparse는 실행 옵션, csv·json은 데이터 저장·설정 읽기, time·datetime은 처리 시각을 담당합니다.
# OpenCV는 IP Webcam 영상과 화면 표시, joblib은 기존 분류 모델, YOLO는 Pose와 Tracking을 사용합니다.
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
from ultralytics import YOLO


# ==========================================================
# ② 프로젝트 ROOT와 내부 모듈 검색 경로 준비
# ==========================================================
# 스크립트가 scripts 폴더에 있다면 parents[1]이 프로젝트 ROOT입니다.
# sys.path에 ROOT를 추가하여 src 폴더의 공통 Python 함수를 불러올 수 있게 합니다.
ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


# ==========================================================
# ③ Pose·다중 인물 상태·실시간 예측 공통 함수 불러오기
# ==========================================================
# src.common은 프로젝트 설정과 실행 장치를 담당합니다.
# MultiPersonStateManager는 Track ID별 상태, feature_from_keypoints는 관절좌표의 Feature 변환을 담당합니다.
# FPSMeter·predict_from_window·validate_runtime_values는 처리속도·Action 예측·설정 검사를 담당합니다.
from src.common import load_config, resolve_device

from src.multi_person_utils import (
    MultiPersonStateManager,
    feature_from_keypoints,
)

from src.realtime_utils import (
    FPSMeter,
    predict_from_window,
    validate_runtime_values,
)


# ==========================================================
# ④ 7일차·8일차 설정과 6일차 모델·로그 경로 지정
# ==========================================================
# 7일차 JSON은 Window·Confidence·안정화 기준을, 8일차 JSON은 Tracker와 Lost Timeout을 담습니다.
# 6일차 학습 정보와 Random Forest 모델은 재학습하지 않고 불러옵니다.
# 결과 로그는 기본적으로 reports/day08/ip_webcam 폴더에 저장됩니다.
DAY07_CONFIG_PATH = (
    ROOT / "configs" / "day07_realtime.json"
)

DAY08_CONFIG_PATH = (
    ROOT / "configs" / "day08_multi_person.json"
)

TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
)

DEFAULT_OUTPUT_DIR = (
    ROOT / "reports" / "day08" / "ip_webcam"
)


# ==========================================================
# ⑤ 명령행 옵션에서 IP Webcam URL을 필수로 받기
# ==========================================================
# ArgumentParser가 터미널 입력을 읽습니다.
# --url은 스마트폰 IP Webcam 영상 스트림 주소이며 반드시 입력해야 합니다.
# 주소는 같은 네트워크에서 접근할 수 있는 실제 스트림 주소여야 합니다.
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        required=True,
        help=(
            "예: http://192.168.0.10:8080/video"
        ),
    )

    # ==========================================================
    # ⑥ Runtime 로그 출력 폴더 옵션 지정
    # ==========================================================
    # --output-dir은 CSV를 저장할 폴더를 바꾸는 선택 옵션입니다.
    # 입력하지 않으면 DEFAULT_OUTPUT_DIR을 사용하고 parse_args()가 입력값을 반환합니다.
    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    return parser.parse_args()


# ==========================================================
# ⑦ JSON 설정을 읽는 공통 함수 작성
# ==========================================================
# 전달받은 JSON 파일을 UTF-8로 열고 Python 딕셔너리 등으로 변환합니다.
# 이 함수로 7일차·8일차 설정과 6일차 학습 정보를 읽습니다.
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================================
# ⑧ 사람별 Track ID에 사용할 구분 색상 선택
# ==========================================================
# 6가지 BGR 색상 팔레트에서 ID에 대응하는 색을 선택합니다.
# track_id % len(palette)는 ID별 색상 인덱스를 만들며, ID가 많으면 색상이 반복될 수 있습니다.
# 색상은 화면 표시용이며 추적 정확도에는 영향을 주지 않습니다.
def track_color(track_id: int) -> tuple[int, int, int]:
    """Track ID마다 눈에 잘 구분되는 BGR 색상을 반환합니다."""
    palette = [
        (255, 170, 0),
        (0, 200, 255),
        (120, 220, 80),
        (220, 120, 255),
        (255, 120, 120),
        (80, 220, 220),
    ]
    return palette[int(track_id) % len(palette)]


# ==========================================================
# ⑨ 텍스트 배경을 가진 Badge 그리기 함수 시작
# ==========================================================
# BBox 안에 읽기 쉬운 글자를 표시하기 위한 함수입니다.
# cv2.getTextSize()는 글자의 너비·높이와 기준선(baseline)을 계산합니다.
# 글자 크기를 먼저 알아야 배경 박스 크기를 결정할 수 있습니다.
def draw_badge(
    image,
    text: str,
    x: int,
    y: int,
    background_color,
    font_scale: float = 0.58,
    thickness: int = 2,
) -> None:
    """작은 배경 박스를 가진 텍스트 라벨을 그립니다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    # ==========================================================
    # ⑩ Badge 여백과 화면 밖 좌표 방지
    # ==========================================================
    # 가로·세로 여백을 설정한 뒤 이미지 크기를 읽습니다.
    # x·y를 0부터 영상의 마지막 픽셀 범위로 제한합니다.
    # 배경 박스도 영상 크기를 넘지 않도록 끝 좌표를 조정합니다.
    pad_x = 7
    pad_y = 5

    image_height, image_width = image.shape[:2]

    x = max(0, min(int(x), image_width - 1))
    y = max(0, min(int(y), image_height - 1))

    box_x1 = x
    box_y1 = y
    box_x2 = min(
        image_width - 1,
        box_x1 + text_width + pad_x * 2,
    )
    box_y2 = min(
        image_height - 1,
        box_y1 + text_height + baseline + pad_y * 2,
    )

    # ==========================================================
    # ⑪ Badge 배경 박스와 흰 글자 표시
    # ==========================================================
    # cv2.rectangle()에 thickness=-1을 주어 배경 사각형을 채웁니다.
    # 여백을 더한 위치에 cv2.putText()로 흰 글자를 그립니다.
    # 학생은 배경이 있어 밝은 영상에서도 글자가 잘 보이는 이유를 확인할 수 있습니다.
    cv2.rectangle(
        image,
        (box_x1, box_y1),
        (box_x2, box_y2),
        background_color,
        -1,
    )

    text_x = box_x1 + pad_x
    text_y = min(
        image_height - 1,
        box_y1 + pad_y + text_height,
    )

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ==========================================================
# ⑫ 한 사람의 Bounding Box·ID·Action 표시 함수 정의
# ==========================================================
# 사람 한 명의 BBox, Track ID, 안정화 Action, Confidence를 받습니다.
# BBox 좌표를 정수로 바꾸고 영상 경계로 제한하여 화면 밖 그리기를 줄입니다.
def draw_person_ui(
    image,
    bbox,
    track_id: int,
    stable_label: str,
    confidence: float,
) -> None:
    """
    한 사람의 BBox, ID, Action을 겹치지 않도록 표시합니다.

    - ID: BBox 왼쪽 위 내부
    - Action/Confidence: BBox 왼쪽 아래 내부
    """
    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = [int(value) for value in bbox]

    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height - 1))

    # ==========================================================
    # ⑬ Track ID별 Bounding Box와 ID Badge 그리기
    # ==========================================================
    # 사람마다 색상을 선택하여 BBox를 그립니다.
    # ID Badge는 BBox의 왼쪽 위에 배치합니다.
    # 같은 ID가 연속 Frame에서 유지되면 같은 색상으로 보입니다.
    color = track_color(track_id)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
        cv2.LINE_AA,
    )

    draw_badge(
        image,
        f"ID {track_id}",
        x1 + 4,
        y1 + 4,
        color,
        font_scale=0.62,
        thickness=2,
    )

    # ==========================================================
    # ⑭ Stable Action과 Confidence를 사람 아래쪽에 표시
    # ==========================================================
    # 안정화된 Action 이름과 Confidence를 문자열로 합칩니다.
    # BBox의 아래쪽 위치를 계산하여 어두운 Badge 위에 표시합니다.
    # 여기서 Confidence는 Action 모델의 출력값이며 추적 ID의 신뢰도와는 다릅니다.
    action_text = (
        f"{stable_label}  {confidence:.2f}"
    )

    action_y = max(
        y1 + 40,
        y2 - 34,
    )

    draw_badge(
        image,
        action_text,
        x1 + 4,
        action_y,
        (35, 35, 35),
        font_scale=0.50,
        thickness=1,
    )


# ==========================================================
# ⑮ 전체 추적 상태 Panel의 표시 항목 준비
# ==========================================================
# seen_ids는 현재 Frame에서 관찰된 ID, runtime_ids는 유지 중인 사람별 상태 ID입니다.
# FPS는 처리 속도를 뜻하며 removed_ids가 있으면 제거된 상태 ID도 보여 줍니다.
# 즉 Seen과 States는 서로 다를 수 있습니다.
def draw_status_panel(
    image,
    seen_ids: set[int],
    runtime_ids: list[int],
    fps: float,
    removed_ids: list[int],
) -> None:
    """화면 왼쪽 위에 최소한의 Runtime 상태만 표시합니다."""
    lines = [
        f"Seen: {sorted(seen_ids)}",
        f"States: {len(runtime_ids)}",
        f"FPS: {fps:.1f}",
    ]

    if removed_ids:
        lines.append(
            f"Removed: {removed_ids}"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    line_height = 24
    padding = 10

    # ==========================================================
    # ⑯ 상태 Panel 크기 계산과 반투명 배경 만들기
    # ==========================================================
    # 각 줄의 글자 너비를 계산하여 필요한 Panel 크기를 정합니다.
    # overlay는 원본 이미지를 복사한 것이며 cv2.addWeighted()로 어두운 배경을 반투명하게 합칩니다.
    widths = [
        cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )[0][0]
        for line in lines
    ]

    panel_width = max(widths) + padding * 2
    panel_height = (
        len(lines) * line_height
        + padding
    )

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.58,
        image,
        0.42,
        0,
        image,
    )

    # ==========================================================
    # ⑰ 상태 Panel에 여러 줄을 순서대로 표시
    # ==========================================================
    # enumerate(lines)는 줄 번호와 문구를 같이 가져옵니다.
    # 줄 번호마다 y좌표를 늘려 Seen·States·FPS 등을 겹치지 않게 그립니다.
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (
                18,
                29 + index * line_height,
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


# ==========================================================
# ⑱ Q 종료·R 초기화 조작키 안내 표시
# ==========================================================
# 화면 왼쪽 아래에 R: reset, Q: quit을 표시합니다.
# 배경 영역을 반투명하게 만들고 그 위에 글자를 그려 쉽게 확인하도록 합니다.
# 실제 키 입력의 동작은 main()의 cv2.waitKey()에서 처리합니다.
def draw_controls(image) -> None:
    """화면 왼쪽 아래에 조작키를 작게 표시합니다."""
    text = "R: reset   Q: quit"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    height, _ = image.shape[:2]
    x = 12
    y = height - 12

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (
            x - 5,
            y - text_height - 8,
        ),
        (
            x + text_width + 7,
            y + baseline + 4,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        image,
        0.45,
        0,
        image,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ==========================================================
# ⑲ main 시작·실행 옵션과 출력 경로 확정
# ==========================================================
# parse_args()에서 --url과 --output-dir을 가져옵니다.
# 출력 경로가 상대경로이면 프로젝트 ROOT를 붙입니다.
# 실시간 실행 전체를 main()에서 순서대로 처리합니다.
def main() -> None:
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    # ==========================================================
    # ⑳ 프로젝트·Runtime·Tracking 설정과 학습 정보 읽기
    # ==========================================================
    # 프로젝트 설정, 7일차 Runtime JSON, 8일차 Tracking JSON, 6일차 Training Info를 읽습니다.
    # Track ID 관리와 Action 추론은 이 설정값들을 공유합니다.
    project_config = load_config()
    day07_config = load_json(DAY07_CONFIG_PATH)
    day08_config = load_json(DAY08_CONFIG_PATH)
    training_info = load_json(TRAIN_INFO_PATH)

    # ==========================================================
    # ㉑ Sequence 길이·Window·Confidence·투표 기준 결정
    # ==========================================================
    # sequence_steps는 학습 모델의 시간 Step 수입니다.
    # window_seconds는 몇 초간의 Frame을 모을지 결정합니다.
    # confidence_threshold와 안정화 Buffer/최소 투표 수는 예측 표시 방식을 결정합니다.
    sequence_steps = int(training_info["sequence_steps"])

    window_seconds = float(
        day07_config["window_seconds"]
    )
    confidence_threshold = float(
        day07_config["confidence_threshold"]
    )
    stabilization_buffer = int(
        day07_config["stabilization_buffer"]
    )
    stabilization_min_votes = int(
        day07_config["stabilization_min_votes"]
    )

    # ==========================================================
    # ㉒ 7일차 실시간 설정값 유효성 검사
    # ==========================================================
    # Window 길이, Confidence Threshold, 안정화 Buffer, 최소 투표 수가 허용 범위인지 공통 함수로 확인합니다.
    # 잘못된 설정을 그대로 추론에 사용하지 않도록 합니다.
    validate_runtime_values(
        window_seconds=window_seconds,
        confidence_threshold=confidence_threshold,
        stabilization_buffer=stabilization_buffer,
        stabilization_min_votes=(
            stabilization_min_votes
        ),
    )

    # ==========================================================
    # ㉓ CPU·GPU 선택 및 6일차 Action 모델 불러오기
    # ==========================================================
    # 프로젝트의 device 설정으로 실행 장치를 선택합니다.
    # joblib.load()는 6일차에 저장한 Random Forest 모델을 메모리로 불러옵니다.
    # joblib은 신뢰할 수 있는 모델 파일에만 사용해야 합니다.
    device = resolve_device(
        project_config["device"]
    )

    action_model = joblib.load(MODEL_PATH)

    # ==========================================================
    # ㉔ 학습 당시 Random Forest 입력 Feature 개수 검사
    # ==========================================================
    # 기본 구조에서는 Step당 38-D Feature를 사용합니다.
    # 따라서 예상 입력 개수는 sequence_steps × 38입니다.
    # 모델의 n_features_in_와 다르면 추론을 시작하지 않고 오류를 발생시킵니다.
    expected_features = sequence_steps * 38

    if int(action_model.n_features_in_) != expected_features:
        raise RuntimeError(
            "6일차 모델 Feature 수와 "
            "현재 sequence_steps가 다릅니다."
        )

    # ==========================================================
    # ㉕ YOLO Pose 모델 불러오기
    # ==========================================================
    # 사람 검출과 Keypoint 추출에 사용할 YOLO Pose 모델을 준비합니다.
    # Random Forest는 Action 분류, YOLO는 Pose·Tracking을 담당하므로 역할이 다릅니다.
    pose_model = YOLO(project_config["pose_model"])

    # ==========================================================
    # ㉖ IP Webcam 연결 및 첫 Frame 검사
    # ==========================================================
    # 사용자가 입력한 스트림 URL을 OpenCV로 엽니다.
    # 영상 연결에 실패하거나 첫 Frame을 읽지 못하면 오류를 발생시킵니다.
    # 첫 Frame으로 해상도를 확인할 수 있습니다.
    capture = cv2.VideoCapture(args.url)

    if not capture.isOpened():
        raise RuntimeError(
            "IP Webcam Stream을 열 수 없습니다."
        )

    success, frame = capture.read()

    if not success or frame is None:
        capture.release()
        raise RuntimeError(
            "IP Webcam 첫 Frame을 읽지 못했습니다."
        )

    # ==========================================================
    # ㉗ 해상도 표시와 키보드 조작 방법 안내
    # ==========================================================
    # 첫 Frame의 세로·가로 크기를 얻고 Connected 메시지로 출력합니다.
    # Q는 종료, R은 사람별 Runtime 상태를 초기화하는 키입니다.
    height, width = frame.shape[:2]

    print(f"Connected: {width} x {height}")
    print("Q: quit")
    print("R: reset all person states")
    print()

    # ==========================================================
    # ㉘ Track ID마다 독립적인 Runtime 상태 관리자 생성
    # ==========================================================
    # MultiPersonStateManager가 사람마다 Feature Buffer·예측 안정화 상태를 관리합니다.
    # window_seconds와 투표 기준은 각 사람의 Window와 Stabilizer에 사용합니다.
    # lost_timeout_seconds는 보이지 않는 사람의 상태를 언제 정리할지 결정합니다.
    state_manager = MultiPersonStateManager(
        window_seconds=window_seconds,
        stabilization_buffer=stabilization_buffer,
        stabilization_min_votes=(
            stabilization_min_votes
        ),
        lost_timeout_seconds=float(
            day08_config["lost_timeout_seconds"]
        ),
    )

    # ==========================================================
    # ㉙ FPS 측정기·실행 시간·Frame 로그 초기화
    # ==========================================================
    # FPSMeter는 화면 처리 속도를 계산합니다.
    # perf_counter()는 세션 경과 시간 계산에 사용됩니다.
    # frame_id는 0부터 시작하고, log_rows에는 사람별 Runtime 기록이 누적됩니다.
    fps_meter = FPSMeter()
    session_start = time.perf_counter()
    frame_id = 0
    log_rows = []

    # ==========================================================
    # ㉚ 실시간 Frame 반복 처리와 경과 시간 계산
    # ==========================================================
    # 실시간으로 Frame을 계속 처리하는 반복문입니다.
    # 각 반복에서 시작 시각부터 지금까지의 경과 초를 계산합니다.
    # 이 경과 시간은 모든 사람의 시간 Window에 공통 시간축으로 쓰입니다.
    while True:
        elapsed = time.perf_counter() - session_start

        # ==========================================================
        # ㉛ YOLO Pose + ByteTrack으로 여러 사람 추적
        # ==========================================================
        # model.track()은 현재 Frame의 사람과 Pose를 추출하고 Track ID를 붙입니다.
        # persist=True로 연속 Frame 사이의 Tracker 상태를 유지합니다.
        # Tracker 종류와 이미지 크기·사람 검출 기준은 설정 JSON에서 가져옵니다.
        result = pose_model.track(
            source=frame,
            persist=True,
            tracker=day08_config["tracker"],
            imgsz=int(project_config["image_size"]),
            conf=float(
                project_config["person_confidence"]
            ),
            device=device,
            verbose=False,
        )[0]

        # Ultralytics 기본 BBox/라벨은 숨기고
        # Skeleton만 그립니다.
        # 아래에서 Track ID와 Action 라벨을 직접 그려
        # 글자 중복을 방지합니다.
        # ==========================================================
        # ㉜ Skeleton 표시용 Frame과 현재 처리 FPS 생성
        # ==========================================================
        # Ultralytics의 기본 BBox와 Label을 숨기고 Skeleton만 시각화합니다.
        # 이후 draw_person_ui()가 사람마다 BBox·ID·Action을 직접 표시하므로 글자 중복을 줄입니다.
        # 현재 처리 FPS도 측정합니다.
        annotated = result.plot(
            boxes=False,
            labels=False,
        )
        current_fps = fps_meter.update(
            time.perf_counter()
        )

        # ==========================================================
        # ㉝ 현재 Frame의 BBox·Keypoint·Track ID 추출
        # ==========================================================
        # 결과에서 Bounding Box와 Keypoint 객체를 가져옵니다.
        # seen_ids는 이 Frame에서 관찰한 Track ID를 담을 집합입니다.
        # ID와 Keypoint 좌표가 모두 있을 때만 Tensor를 NumPy 배열로 변환해 개인별 처리로 들어갑니다.
        boxes = result.boxes
        keypoints = result.keypoints
        seen_ids: set[int] = set()

        if (
            boxes is not None
            and boxes.id is not None
            and keypoints is not None
            and keypoints.xy is not None
        ):
            track_ids = (
                boxes.id.detach().cpu().numpy().astype(int)
            )
            xyxy = boxes.xyxy.detach().cpu().numpy()
            kp_xy = (
                keypoints.xy.detach().cpu().numpy()
            )

            # ==========================================================
            # ㉞ 서로 대응하는 사람 자료 개수만큼 반복
            # ==========================================================
            # Track ID, BBox, Keypoint 배열의 길이 중 최솟값만 사용합니다.
            # 같은 index에 있는 세 정보를 한 사람의 정보로 취급합니다.
            # 각 사람을 순회하면서 Track ID를 seen_ids에 추가합니다.
            item_count = min(
                len(track_ids),
                len(xyxy),
                len(kp_xy),
            )

            for index in range(item_count):
                track_id = int(track_ids[index])
                seen_ids.add(track_id)

                # ==========================================================
                # ㉟ 현재 Track ID의 상태 가져오기 또는 새로 생성
                # ==========================================================
                # 이미 관리 중인 ID면 기존 State를 가져오고, 처음 본 ID면 새 State를 생성합니다.
                # 이렇게 해야 사람 A와 사람 B의 Feature Window가 서로 섞이지 않습니다.
                # 각 State에는 Buffer·Stabilizer·최근 예측 시간 등이 들어 있습니다.
                state = state_manager.get_or_create(
                    track_id,
                    elapsed,
                )

                # ==========================================================
                # ㊱ 사람별 관절좌표를 Frame Feature로 변환
                # ==========================================================
                # 현재 사람의 Keypoint 좌표 배열을 공통 변환 함수에 전달합니다.
                # 이 함수의 반환 Feature는 해당 사람의 Action 분류 Window 입력으로 사용됩니다.
                # 실제 결측점 처리·정규화 규칙은 src.multi_person_utils 구현에 따릅니다.
                feature = feature_from_keypoints(
                    kp_xy[index]
                )

                # ==========================================================
                # ㊲ 개인별 시간 Window에 Feature 추가하고 준비 여부 검사
                # ==========================================================
                # 경과 시간과 Feature를 현재 사람의 Buffer에 추가합니다.
                # is_ready()는 Window 충족 비율과 유효 Feature 조건을 확인합니다.
                # 조건을 만족해야 Action 예측을 시작합니다.
                state.feature_buffer.append(
                    elapsed,
                    feature,
                )

                ready = state.feature_buffer.is_ready(
                    min_window_fill_ratio=float(
                        day07_config[
                            "min_window_fill_ratio"
                        ]
                    ),
                    min_valid_feature_ratio=float(
                        day07_config[
                            "min_valid_feature_ratio"
                        ]
                    ),
                    min_valid_features=int(
                        day07_config[
                            "min_valid_features"
                        ]
                    ),
                )

                # ==========================================================
                # ㊳ 예측 갱신 주기를 확인하고 Random Forest로 Action 예측
                # ==========================================================
                # 이번 Frame에서 새 예측 여부를 0으로 시작합니다.
                # Window가 준비되었고 마지막 예측 후 설정 간격이 지났을 때만 predict_from_window()를 실행합니다.
                # Buffer를 학습 Step 수에 맞춘 입력으로 변환해 Raw Label·Threshold Label·Confidence를 얻습니다.
                prediction_updated = 0

                if (
                    ready
                    and (
                        elapsed
                        - state.last_prediction_time
                    )
                    >= float(
                        day07_config[
                            "prediction_interval_sec"
                        ]
                    )
                ):
                    (
                        threshold_label,
                        raw_label,
                        confidence,
                        _,
                    ) = predict_from_window(
                        model=action_model,
                        feature_buffer=(
                            state.feature_buffer
                        ),
                        sequence_steps=sequence_steps,
                        confidence_threshold=(
                            confidence_threshold
                        ),
                    )

                    # ==========================================================
                    # ㊴ Raw·Threshold·Confidence 저장 및 Stable Label 갱신
                    # ==========================================================
                    # 예측한 Raw Label과 Confidence 기준을 적용한 Threshold Label을 개인 State에 저장합니다.
                    # Stabilizer에 Threshold Label을 전달하여 Stable Label을 구합니다.
                    # 최근 예측 시각을 갱신하고 prediction_updated를 1로 설정합니다.
                    state.raw_label = raw_label
                    state.threshold_label = (
                        threshold_label
                    )
                    state.confidence = confidence
                    state.stable_label = (
                        state.stabilizer.update(
                            threshold_label
                        )
                    )
                    state.last_prediction_time = elapsed
                    prediction_updated = 1

                # ==========================================================
                # ㊵ Window가 부족하면 WARMING_UP 상태 표시
                # ==========================================================
                # Window가 준비되지 않으면 세 가지 Label을 WARMING_UP으로 바꾸고 Confidence를 0으로 합니다.
                # Window가 준비되었지만 예측 간격이 아직 지나지 않았다면 이전 예측 상태를 그대로 유지합니다.
                elif not ready:
                    state.raw_label = "WARMING_UP"
                    state.threshold_label = "WARMING_UP"
                    state.stable_label = "WARMING_UP"
                    state.confidence = 0.0

                x1, y1, x2, y2 = xyxy[index]

                # 사람마다 한 개의 BBox만 그리고,
                # ID와 Action을 위/아래로 분리해 표시합니다.
                # ==========================================================
                # ㊶ 사람별 BBox·Track ID·Action·Confidence 표시
                # ==========================================================
                # 현재 사람의 BBox 좌표와 State의 Stable Label·Confidence를 draw_person_ui()에 전달합니다.
                # 사람마다 추적 ID와 분류 결과가 영상에 따로 표시됩니다.
                draw_person_ui(
                    annotated,
                    (x1, y1, x2, y2),
                    track_id,
                    state.stable_label,
                    state.confidence,
                )

                # ==========================================================
                # ㊷ 현재 Frame·Track ID별 Runtime 로그 한 행 누적
                # ==========================================================
                # Frame 번호·세션 경과 시간·Track ID를 기록합니다.
                # 개인 Buffer 길이·유효 비율·Raw/Threshold/Stable·Confidence·예측 갱신 여부·FPS도 기록합니다.
                # 사람이 2명이면 한 Frame에 로그가 2행 생길 수 있습니다.
                log_rows.append(
                    {
                        "frame_id": frame_id,
                        "elapsed_sec": round(
                            elapsed,
                            4,
                        ),
                        "track_id": track_id,
                        "buffer_duration_sec": round(
                            state.feature_buffer.duration(),
                            4,
                        ),
                        "valid_feature_ratio": round(
                            state.feature_buffer.valid_ratio(),
                            4,
                        ),
                        "raw_label": state.raw_label,
                        "threshold_label": (
                            state.threshold_label
                        ),
                        "stable_label": (
                            state.stable_label
                        ),
                        "confidence": round(
                            state.confidence,
                            4,
                        ),
                        "prediction_updated": (
                            prediction_updated
                        ),
                        "processing_fps": round(
                            current_fps,
                            3,
                        ),
                    }
                )

        # ==========================================================
        # ㊸ 이번 Frame에 없는 사람을 표시하고 Timeout으로 정리
        # ==========================================================
        # seen_ids에 없는 관리 ID는 이번 Frame에서 보이지 않은 상태로 처리합니다.
        # remove_lost()는 설정된 Lost Timeout에 따라 오래 사라진 개인 State를 제거합니다.
        # 이 정리는 개인 Runtime 상태의 제거이며 실제 사람의 퇴장을 확정하는 판단은 아닙니다.
        state_manager.mark_missing(
            elapsed,
            seen_ids,
        )

        removed_ids = state_manager.remove_lost(
            elapsed
        )

        # 상태 정보는 반투명 패널 하나에 간단히 표시합니다.
        # ==========================================================
        # ㊹ 전체 상태 Panel과 조작키를 그린 후 화면 출력
        # ==========================================================
        # Seen ID 목록, 관리 중인 State 수, FPS, 제거 ID를 반투명 Panel에 그립니다.
        # R/Q 조작키도 표시한 후 cv2.imshow()로 결과를 보여 줍니다.
        draw_status_panel(
            annotated,
            seen_ids,
            state_manager.active_ids(),
            current_fps,
            removed_ids,
        )

        draw_controls(annotated)

        cv2.imshow(
            "Subject10 Multi-Person Action",
            annotated,
        )

        # ==========================================================
        # ㊺ Q 종료·R 상태 초기화 키 처리
        # ==========================================================
        # Q를 누르면 반복문을 종료합니다.
        # R을 누르면 MultiPersonStateManager의 사람별 Runtime State를 초기화합니다.
        # R은 모델의 ByteTrack 내부 Track ID까지 반드시 초기화하는 동작은 아닙니다.
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        if key in (ord("r"), ord("R")):
            state_manager.clear()
            print("All person runtime states reset")

        # ==========================================================
        # ㊻ 다음 IP Webcam Frame 읽기 및 반복 종료 조건
        # ==========================================================
        # 다음 Frame을 받아 성공하면 frame_id를 증가시키고 반복합니다.
        # 읽기에 실패하거나 Frame이 None이면 메시지를 출력하고 종료합니다.
        # 원본 코드는 이 단계에서 재시도 없이 종료합니다.
        success, frame = capture.read()

        if not success or frame is None:
            print("다음 IP Webcam Frame을 읽지 못했습니다.")
            break

        frame_id += 1

    # ==========================================================
    # ㊼ 카메라와 창을 닫고 Runtime 로그 폴더 준비
    # ==========================================================
    # 반복문이 끝나면 VideoCapture를 해제하고 OpenCV 창을 닫습니다.
    # 출력 폴더가 없다면 부모 폴더까지 생성합니다.
    capture.release()
    cv2.destroyAllWindows()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================================
    # ㊽ 실행 시각으로 로그 CSV 파일명 만들기
    # ==========================================================
    # 날짜와 시각을 YYYYMMDD_HHMMSS 형식으로 만든 뒤 ip_multi_<시각>.csv를 구성합니다.
    # 실험마다 결과를 다른 파일명으로 남기기 위한 방식입니다.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_log = (
        output_dir
        / f"ip_multi_{timestamp}.csv"
    )

    # ==========================================================
    # ㊾ 사람별 로그가 있으면 CSV 저장, 없으면 안내
    # ==========================================================
    # log_rows가 있으면 DictWriter로 컬럼 헤더와 전체 행을 UTF-8-SIG CSV에 기록합니다.
    # 사람 Track ID가 기록된 행이 없다면 CSV를 생성하지 않고 안내 메시지를 출력합니다.
    # 영상 자체를 MP4로 저장하는 코드는 포함되어 있지 않습니다.
    if log_rows:
        with output_log.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(log_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(log_rows)

        print()
        print(f"Saved: {output_log}")
    else:
        print()
        print(
            "Track ID가 있는 Runtime Log가 없습니다. "
            "두 사람이 충분히 보이는지 확인하세요."
        )


# ==========================================================
# ㊿ Python 파일 직접 실행 시 main 호출
# ==========================================================
# python scripts/49_run_ip_webcam_multi_person_action.py처럼 파일을 직접 실행하면 main()을 실행합니다.
# 다른 모듈에서 import하면 자동으로 main()을 시작하지 않습니다.
if __name__ == "__main__":
    main()