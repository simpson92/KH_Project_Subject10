from __future__ import annotations

# ======================================================
# ① 필요한 라이브러리 가져오기
# ======================================================
# argparse는 --url 같은 터미널 옵션을 읽습니다.
# csv·json은 로그 저장 및 설정 파일 읽기에 사용합니다.
# sys·Path는 프로젝트 내부 모듈과 경로를 관리합니다.
# time·datetime은 처리 시간과 파일명 시각을 기록합니다.
# OpenCV는 IP Webcam 영상을 받고, joblib은 학습 모델을 읽습니다.
# YOLO는 사람의 Pose Keypoint를 추출합니다.

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


# ======================================================
# ② 프로젝트 ROOT와 Python 검색 경로 설정
# ======================================================
# 현재 스크립트가 scripts 폴더 안에 있다고 가정합니다.
# parents[1]은 프로젝트의 최상위(ROOT) 폴더입니다.
# sys.path.insert(0, ...)는 src 모듈을 import할 수 있게 합니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ======================================================
# ③ 프로젝트 설정 및 CPU·GPU 선택 함수 가져오기
# ======================================================
# load_config()는 src.common의 프로젝트 설정을 읽습니다.
# resolve_device()는 설정값에 맞는 실행 장치를 결정합니다.
# 이 코드에서 학습 모델을 새로 학습하지는 않습니다.

from src.common import (
    load_config,
    resolve_device,
)

# ======================================================
# ④ 실시간 추론에 필요한 공통 함수 가져오기
# ======================================================
# TimedFeatureBuffer는 시간 Window의 Feature를 보관합니다.
# PredictionStabilizer는 최근 예측들을 모아 Label을 안정화합니다.
# FPSMeter는 처리 속도를 기록합니다.
# frame_feature_from_pose()는 Pose를 Frame Feature로 만듭니다.
# predict_from_window()는 Window를 Sequence로 바꿔 Action을 예측합니다.
# validate_runtime_values()는 실행 옵션의 범위를 검사합니다.

from src.realtime_utils import (
    FPSMeter,
    PredictionStabilizer,
    TimedFeatureBuffer,
    frame_feature_from_pose,
    predict_from_window,
    validate_runtime_values,
)


# ======================================================
# ⑤ 7일차 설정·6일차 모델·출력 폴더 경로 지정
# ======================================================
# day07_realtime.json은 Window와 예측 주기 등의 설정입니다.
# rf_training_info.json은 6일차 학습 시 사용한 Sequence 길이를 담습니다.
# rf_action_baseline.joblib은 학습 완료된 Random Forest 모델입니다.
# 결과는 기본적으로 reports/day07/ip_webcam 폴더에 CSV로 기록됩니다.
# 이 프로그램은 결과 MP4가 아니라 런타임 CSV를 저장합니다.

DAY07_CONFIG_PATH = (
    ROOT
    / "configs"
    / "day07_realtime.json"
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
    ROOT
    / "reports"
    / "day07"
    / "ip_webcam"
)

# ======================================================
# ⑥ 숫자키 1~6과 Action 이름 연결
# ======================================================
# 키보드에서 1~6을 누르면 사람이 현재 수행할 목표 Action을 선택합니다.
# 예: 6키 → wave.
# ord()는 키 문자를 OpenCV 키 코드와 비교하기 위한 정수로 바꿉니다.
# 여기서 Target은 참조 정답이며 모델 예측과는 다릅니다.

KEY_TO_ACTION = {
    ord("1"): "bend_return",
    ord("2"): "leg_raise_lower",
    ord("3"): "walk_turn_walk",
    ord("4"): "drink_return",
    ord("5"): "sit_stand",
    ord("6"): "wave",
}


# ======================================================
# ⑦ 명령행 인자 파서 생성 및 IP Webcam 주소 필수 입력
# ======================================================
# parse_args()는 실행할 때 입력한 옵션을 읽습니다.
# --url은 필수 옵션입니다.
# 예: --url http://192.168.0.10:8080/video
# 주소는 스마트폰·앱의 실제 스트림 주소로 바꿔야 합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        required=True,
        help=(
            "예: "
            "http://192.168.0.10:"
            "8080/video"
        ),
    )

    # ======================================================
    # ⑧ Window 길이·Confidence·안정화 버퍼 옵션 설정
    # ======================================================
    # --window-seconds는 최근 몇 초를 볼지 정합니다.
    # --confidence는 UNKNOWN 처리에 쓰는 Confidence 기준입니다.
    # --stabilization-buffer는 최근 예측 몇 개를 모을지 정합니다.
    # 옵션을 생략하면 day07_realtime.json의 값을 사용합니다.

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--stabilization-buffer",
        type=int,
        default=None,
    )

    # ======================================================
    # ⑨ Target 변경 시 Buffer 유지 옵션과 출력 폴더 설정
    # ======================================================
    # --keep-buffer-on-target을 입력하면 목표 Action 키를 바꾸어도
    # 기존 시간 Window와 안정화 기록을 그대로 유지합니다.
    # 이 옵션이 없으면 목표 Action을 변경할 때 Buffer를 초기화합니다.
    # --output-dir은 런타임 로그 CSV를 저장할 폴더입니다.
    # 마지막에 parse_args()로 해석된 옵션을 반환합니다.

    parser.add_argument(
        "--keep-buffer-on-target",
        action="store_true",
        help=(
            "Target 키를 바꿀 때 "
            "Sliding Window를 유지하여 "
            "연속 Action 전환을 관찰합니다."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Runtime Log를 "
            "저장할 폴더"
        ),
    )

    return parser.parse_args()


# ======================================================
# ⑩ JSON 파일을 읽는 공통 함수 정의
# ======================================================
# 전달받은 경로를 UTF-8 형식으로 열어 설정을 읽습니다.
# json.load()가 Python 딕셔너리 형태로 변환합니다.
# 7일차 설정과 학습 정보를 이 함수로 불러옵니다.

def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ======================================================
# ⑪ IP Webcam Frame 읽기 재시도 함수 정의
# ======================================================
# IP Webcam의 MJPEG 스트림은 일시적으로 Frame 읽기에 실패할 수 있습니다.
# read_frame_with_retry()는 기본 최대 10회 읽기를 시도합니다.
# 중간에 성공하면 즉시 (True, Frame)을 반환합니다.
# 실패 시 기본 0.05초 쉬고 다시 읽으며, 끝까지 실패하면 (False, None)을 반환합니다.
# 이 기능은 연결 끊김을 무한히 복구하는 재접속 기능은 아닙니다.

def read_frame_with_retry(
    capture: cv2.VideoCapture,
    max_retries: int = 10,
    retry_delay_sec: float = 0.05,
):
    """
    IP Webcam의 MJPEG Stream에서 한 번의 일시적인 read 실패가
    발생해도 바로 종료하지 않고 짧게 재시도합니다.
    """
    for attempt in range(1, max_retries + 1):
        success, frame = capture.read()

        if success and frame is not None:
            return True, frame

        if attempt < max_retries:
            time.sleep(retry_delay_sec)

    return False, None


# ======================================================
# ⑫ main 시작: 입력 옵션 및 세 설정 읽기
# ======================================================
# main()은 실시간 Action Recognition을 실행하는 중심 함수입니다.
# parse_args()로 터미널 옵션을 읽습니다.
# src.common의 load_config()로 프로젝트 설정을 읽습니다.
# day07_realtime.json과 6일차 Training Info를 읽습니다.

def main() -> None:
    args = parse_args()

    project_config = load_config()
    runtime_config = load_json(
        DAY07_CONFIG_PATH
    )
    training_info = load_json(
        TRAIN_INFO_PATH
    )

    # ======================================================
    # ⑬ 학습 당시 Sequence Step 수와 Window 길이 결정
    # ======================================================
    # sequence_steps는 6일차 학습 모델에 맞는 시간 Step 수입니다.
    # 기본 수업에서는 30 Step을 사용합니다.
    # window_seconds는 최근 몇 초의 움직임을 모을지 정합니다.
    # --window-seconds 옵션이 있으면 그 값을, 없으면 JSON 값을 사용합니다.
    # Window 길이(초)와 Sequence Step 수(개)는 서로 다른 개념입니다.

    sequence_steps = int(
        training_info[
            "sequence_steps"
        ]
    )

    window_seconds = (
        float(
            args.window_seconds
        )
        if args.window_seconds
        is not None
        else float(
            runtime_config[
                "window_seconds"
            ]
        )
    )

    # ======================================================
    # ⑭ Confidence Threshold와 안정화 Buffer 크기 결정
    # ======================================================
    # confidence_threshold는 모델 예측에 Confidence 기준을 적용할 때 사용합니다.
    # stabilization_buffer는 최근 몇 회 예측을 모아 판단할지 정합니다.
    # 각 항목은 명령행 옵션이 있으면 그 값을 우선하고,
    # 그렇지 않으면 day07_realtime.json을 사용합니다.

    confidence_threshold = (
        float(
            args.confidence
        )
        if args.confidence
        is not None
        else float(
            runtime_config[
                "confidence_threshold"
            ]
        )
    )

    stabilization_buffer = (
        int(
            args.stabilization_buffer
        )
        if args.stabilization_buffer
        is not None
        else int(
            runtime_config[
                "stabilization_buffer"
            ]
        )
    )

    # ======================================================
    # ⑮ 안정화에 필요한 최소 투표 수 결정
    # ======================================================
    # Buffer 크기를 명령행에서 직접 바꾼 경우 과반수 기준을 적용합니다.
    # 예: Buffer가 5이면 최소 투표 수는 5//2+1 = 3입니다.
    # Buffer 크기를 직접 변경하지 않았다면 JSON의 min_votes를 사용합니다.
    # Buffer에 모이는 예측값들의 합의 정도를 결정하는 설정입니다.

    if (
        args.stabilization_buffer
        is not None
    ):
        stabilization_min_votes = (
            stabilization_buffer
            // 2
            + 1
        )
    else:
        stabilization_min_votes = int(
            runtime_config[
                "stabilization_min_votes"
            ]
        )

    # ======================================================
    # ⑯ 출력 경로 확정 및 Runtime 옵션 값 검사
    # ======================================================
    # --output-dir이 절대경로라면 그대로 사용합니다.
    # 상대경로라면 프로젝트 ROOT와 연결합니다.
    # validate_runtime_values()로 Window 길이·Confidence·Buffer·최소 투표 수의
    # 값이 허용되는 범위에 있는지 검사합니다.

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.is_absolute():
        output_dir = (
            ROOT
            / output_dir
        )

    validate_runtime_values(
        window_seconds=(
            window_seconds
        ),
        confidence_threshold=(
            confidence_threshold
        ),
        stabilization_buffer=(
            stabilization_buffer
        ),
        stabilization_min_votes=(
            stabilization_min_votes
        ),
    )

    # ======================================================
    # ⑰ Buffer 유지 방식·실행 장치·학습 모델 준비
    # ======================================================
    # --keep-buffer-on-target 입력 여부를 bool로 보관합니다.
    # resolve_device()는 프로젝트 설정에 따라 CPU/GPU를 선택합니다.
    # joblib.load()는 6일차 Random Forest 모델을 복원합니다.
    # 주의: 신뢰할 수 없는 joblib 파일은 불러오지 않습니다.

    keep_buffer_on_target = bool(
        args.keep_buffer_on_target
    )

    device = resolve_device(
        project_config["device"]
    )

    action_model = joblib.load(
        MODEL_PATH
    )

    # ======================================================
    # ⑱ YOLO Pose 모델 준비 및 IP Webcam 연결
    # ======================================================
    # 프로젝트 설정에 적힌 Pose 모델을 YOLO로 불러옵니다.
    # cv2.VideoCapture(args.url)로 IP Webcam 영상 스트림을 엽니다.
    # isOpened()가 False이면 연결 실패 오류를 발생시킵니다.
    # 스마트폰과 PC가 통신 가능한 네트워크에 있어야 합니다.

    pose_model = YOLO(
        project_config[
            "pose_model"
        ]
    )

    capture = cv2.VideoCapture(
        args.url
    )

    if not capture.isOpened():
        raise RuntimeError(
            "IP Webcam Stream을 "
            "열 수 없습니다."
        )

    # ======================================================
    # ⑲ IP Webcam 첫 Frame을 읽고 오류 검사
    # ======================================================
    # read_frame_with_retry()를 호출해 첫 Frame을 가져옵니다.
    # 여러 번 재시도해도 실패하면 capture를 닫고 중단합니다.
    # 첫 Frame이 있어야 영상 해상도와 화면 구성을 결정할 수 있습니다.

    success, frame = (
        read_frame_with_retry(
            capture
        )
    )

    if (
        not success
        or frame is None
    ):
        capture.release()
        raise RuntimeError(
            "IP Webcam 첫 Frame을 "
            "읽지 못했습니다."
        )

    # ======================================================
    # ⑳ 영상 해상도와 OpenCV 표시 창 생성
    # ======================================================
    # 첫 Frame의 shape[:2]에서 높이(height)와 너비(width)를 가져옵니다.
    # 오른쪽에 붙일 정보 Panel의 너비를 380픽셀로 정합니다.
    # namedWindow()로 OpenCV 창을 만들고,
    # resizeWindow()로 카메라 영상과 Panel이 함께 보일 크기로 설정합니다.
    # 그래픽 화면이 없는 환경에서는 imshow 창이 뜨지 않을 수 있습니다.

    height, width = (
        frame.shape[:2]
    )

    WINDOW_NAME = "Subject10 IP Webcam Action"
    PANEL_WIDTH = 380

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        width + PANEL_WIDTH,
        height,
    )

    # ======================================================
    # ㉑ 실시간 조작 키 안내 출력
    # ======================================================
    # 연결 해상도와 키보드 조작법을 터미널에 출력합니다.
    # 1~6: 목표 Action 선택 / 0: 목표 해제.
    # R: Buffer 초기화 / Q: 프로그램 종료.
    # Target 변경 시 Buffer를 KEEP할지 RESET할지도 표시합니다.

    print(
        f"Connected: "
        f"{width} x {height}"
    )
    print()
    print(
        "1 bend_return"
    )
    print(
        "2 leg_raise_lower"
    )
    print(
        "3 walk_turn_walk"
    )
    print(
        "4 drink_return"
    )
    print(
        "5 sit_stand"
    )
    print(
        "6 wave"
    )
    print(
        "0 target clear"
    )
    print(
        "R buffer reset"
    )
    print(
        "Q quit"
    )
    print()
    print(
        "Target mode:",
        (
            "KEEP BUFFER"
            if keep_buffer_on_target
            else "RESET BUFFER"
        ),
    )

    # ======================================================
    # ㉒ 시간 Window Buffer와 예측 안정화 도구 생성
    # ======================================================
    # TimedFeatureBuffer는 Frame Feature와 측정 시각을 모읍니다.
    # PredictionStabilizer는 최근 예측을 받아 안정된 Action Label을 만듭니다.
    # 두 도구는 현재 Window 길이·버퍼 크기·최소 투표 수로 초기화됩니다.

    feature_buffer = (
        TimedFeatureBuffer(
            window_seconds
        )
    )

    stabilizer = (
        PredictionStabilizer(
            buffer_size=(
                stabilization_buffer
            ),
            min_votes=(
                stabilization_min_votes
            ),
        )
    )

    # ======================================================
    # ㉓ FPS 측정기 및 초기 예측 상태 설정
    # ======================================================
    # FPSMeter는 실행 중 처리 속도를 측정합니다.
    # last_prediction_time을 매우 작은 값(-1e9)으로 설정하여
    # Window가 처음 준비되면 첫 예측을 허용합니다.
    # Raw·Threshold·Stable Label은 WARMING_UP으로 시작합니다.
    # Confidence는 0, 목표 Action은 빈 문자열로 시작합니다.

    fps_meter = FPSMeter()

    last_prediction_time = (
        -1e9
    )

    raw_label = "WARMING_UP"
    threshold_label = (
        "WARMING_UP"
    )
    stable_label = (
        "WARMING_UP"
    )
    confidence = 0.0
    expected_label = ""

    # ======================================================
    # ㉔ 세션 시작 시각과 로그·Frame·구간 번호 초기화
    # ======================================================
    # time.perf_counter()를 기준으로 경과 시간을 계산합니다.
    # log_rows는 Frame별 실행 기록을 모으는 목록입니다.
    # frame_id는 처리한 Frame 번호입니다.
    # segment_id는 사용자가 Target을 선택·전환한 구간을 구별합니다.

    session_start = (
        time.perf_counter()
    )

    log_rows = []
    frame_id = 0
    segment_id = 0

    # ======================================================
    # ㉕ 실시간 처리 반복문과 현재 경과 시간 계산
    # ======================================================
    # while True 반복문은 영상이 들어오는 동안 동작합니다.
    # 현재 시각에서 세션 시작 시각을 빼 elapsed를 계산합니다.
    # 저장영상의 frame_id/FPS와 달리 실제 실행 중 흐른 시간입니다.

    while True:
        now = time.perf_counter()
        elapsed = (
            now
            - session_start
        )

        # ======================================================
        # ㉖ 현재 Frame에서 YOLO Pose 추론
        # ======================================================
        # 현재 스마트폰 Frame을 pose_model.predict()에 전달합니다.
        # image_size·person_confidence는 프로젝트 설정을 사용합니다.
        # 선택한 device에서 사람의 Pose Keypoint를 추론합니다.
        # [0]은 예측 결과 목록 중 첫 번째 이미지 결과입니다.
        # 이 단계는 Random Forest Action 분류가 아닌 Pose 검출입니다.

        result = (
            pose_model.predict(
                source=frame,
                imgsz=int(
                    project_config[
                        "image_size"
                    ]
                ),
                conf=float(
                    project_config[
                        "person_confidence"
                    ]
                ),
                device=device,
                verbose=False,
            )[0]
        )

        # ======================================================
        # ㉗ Pose를 Frame Feature로 바꾸고 시간 Buffer에 추가
        # ======================================================
        # frame_feature_from_pose()는 현재 Pose를 모델 입력용 Frame Feature로 만듭니다.
        # 6일차 수업의 기본 구성은 38-D Feature입니다.
        # feature_buffer.append(elapsed, feature)는 생성한 값을 시각과 함께 보관합니다.
        # Pose를 제대로 얻지 못한 경우의 처리 방식은 공통 함수 구현을 따릅니다.

        feature = (
            frame_feature_from_pose(
                result
            )
        )

        feature_buffer.append(
            elapsed,
            feature,
        )

        # ======================================================
        # ㉘ Window가 Action 예측을 수행할 만큼 준비되었는지 검사
        # ======================================================
        # Window가 충분히 채워졌는지 확인합니다.
        # min_window_fill_ratio: 필요한 시간 Window 충족 비율입니다.
        # min_valid_feature_ratio: 유효 Feature 비율의 기준입니다.
        # min_valid_features: 최소 유효 Feature 수입니다.
        # 세 조건을 공통 Buffer 객체의 is_ready()로 검사합니다.

        ready = (
            feature_buffer.is_ready(
                min_window_fill_ratio=float(
                    runtime_config[
                        "min_window_fill_ratio"
                    ]
                ),
                min_valid_feature_ratio=float(
                    runtime_config[
                        "min_valid_feature_ratio"
                    ]
                ),
                min_valid_features=int(
                    runtime_config[
                        "min_valid_features"
                    ]
                ),
            )
        )

        # ======================================================
        # ㉙ 예측 실행 주기 확인 및 갱신 여부 표시
        # ======================================================
        # prediction_updated=0은 현재 Frame에서 새 예측이 없음을 뜻합니다.
        # Window가 준비되어 있고 마지막 예측으로부터
        # prediction_interval_sec 이상 경과했을 때만 새 예측을 수행합니다.
        # 모든 Frame마다 Random Forest를 호출하지 않도록 제어합니다.

        prediction_updated = 0

        if (
            ready
            and (
                elapsed
                - last_prediction_time
            )
            >= float(
                runtime_config[
                    "prediction_interval_sec"
                ]
            )
        ):
            # ======================================================
            # ㉚ 시간 Window로 Raw 및 Threshold Action 예측
            # ======================================================
            # predict_from_window()가 Buffer를 학습 당시의 Sequence Step 수로 맞춥니다.
            # Random Forest 모델에 입력하여 Raw Label을 얻습니다.
            # Confidence 기준을 반영한 Threshold Label도 반환합니다.
            # 네 번째 반환값은 현재 코드에서 사용하지 않아 _로 받습니다.

            (
                threshold_label,
                raw_label,
                confidence,
                _,
            ) = predict_from_window(
                model=action_model,
                feature_buffer=(
                    feature_buffer
                ),
                sequence_steps=(
                    sequence_steps
                ),
                confidence_threshold=(
                    confidence_threshold
                ),
            )

            # ======================================================
            # ㉛ 예측 안정화 및 마지막 예측 시각 갱신
            # ======================================================
            # stabilizer.update(threshold_label)는 최근 예측을 종합합니다.
            # stable_label에 안정화된 결과를 저장합니다.
            # 예측을 수행한 시각을 last_prediction_time에 기록합니다.
            # 이번 Frame에서 새 예측을 했으므로 prediction_updated=1로 바꿉니다.

            stable_label = (
                stabilizer.update(
                    threshold_label
                )
            )

            last_prediction_time = (
                elapsed
            )
            prediction_updated = 1

        # ======================================================
        # ㉜ Window가 준비되지 않았을 때 WARMING_UP 처리
        # ======================================================
        # ready=False이면 모델 예측 결과를 표시하지 않습니다.
        # Raw·Threshold·Stable Label을 모두 WARMING_UP으로 바꿉니다.
        # Confidence를 0으로 되돌립니다.
        # Window가 준비되어 있지만 예측 주기가 안 됐다면 이전 Label은 유지됩니다.

        elif not ready:
            raw_label = (
                "WARMING_UP"
            )
            threshold_label = (
                "WARMING_UP"
            )
            stable_label = (
                "WARMING_UP"
            )
            confidence = 0.0

        # ======================================================
        # ㉝ Pose 시각화 Frame 생성 및 Processing FPS 측정
        # ======================================================
        # result.plot()은 Pose가 표시된 Frame을 만듭니다.
        # FPSMeter.update()는 현재 처리 시각으로 FPS를 계산합니다.
        # 표시되는 FPS는 스마트폰 원본 FPS가 아니라 프로그램 처리 속도입니다.

        annotated = (
            result.plot()
        )

        processing_fps = (
            fps_meter.update(
                time.perf_counter()
            )
        )

        # ======================================================
        # ㉞ 카메라 오른쪽에 별도 Runtime Panel 붙이기
        # ======================================================
        # copyMakeBorder()로 영상 오른쪽에 380픽셀 Panel을 추가합니다.
        # 원본 영상 위에 검은 글상자를 덮지 않으므로 Pose를 보기 쉽습니다.
        # Panel 시작 x좌표를 정하고 제목 RUNTIME STATUS를 그립니다.

        # -------------------------------------------------
        # 카메라 영상 오른쪽에 Runtime 정보 Panel을 붙입니다.
        # 카메라 원본에는 검은 박스나 텍스트를 덮지 않으므로
        # 사람의 머리부터 발끝까지 Pose를 그대로 확인할 수 있습니다.
        # -------------------------------------------------
        display_frame = cv2.copyMakeBorder(
            annotated,
            0,
            0,
            0,
            PANEL_WIDTH,
            cv2.BORDER_CONSTANT,
            value=(25, 25, 25),
        )

        panel_x = width + 20

        cv2.putText(
            display_frame,
            "RUNTIME STATUS",
            (panel_x, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

        # ======================================================
        # ㉟ Target·Raw·Threshold·Stable 등 표시 문자열 준비
        # ======================================================
        # Target은 사용자가 키로 지정한 참조 Action입니다.
        # Raw는 모델의 초기 예측, Threshold는 Confidence 적용 결과,
        # Stable은 안정화한 최종 표시 Label입니다.
        # Confidence·Window 초·Processing FPS·KEEP/RESET 모드도 표시합니다.
        # Target과 모델의 예측을 구분하여 확인합니다.

        texts = [
            f"Target: {expected_label or '-'}",
            f"Raw: {raw_label}",
            f"Threshold: {threshold_label}",
            f"Stable: {stable_label}",
            f"Confidence: {confidence:.2f}",
            f"Window: {window_seconds:.1f}s",
            f"FPS: {processing_fps:.1f}",
            (
                "Mode: "
                + (
                    "KEEP"
                    if keep_buffer_on_target
                    else "RESET"
                )
            ),
        ]

        # ======================================================
        # ㊱ Runtime 상태 문자열을 Panel에 차례로 그리기
        # ======================================================
        # start_y와 line_gap으로 표시 위치·행 간격을 정합니다.
        # for 반복문에서 cv2.putText()로 상태 문구를 순서대로 출력합니다.
        # 이 부분은 터미널 print가 아니라 화면 Frame에 글자를 그립니다.

        start_y = 72
        line_gap = 32

        for index, text in enumerate(
            texts
        ):
            cv2.putText(
                display_frame,
                text,
                (
                    panel_x,
                    start_y
                    + index * line_gap,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # ======================================================
        # ㊲ 구분선과 키보드 조작 안내 표시
        # ======================================================
        # 수평 구분선을 그린 뒤 1~6·0·R·Q 조작법을 표시합니다.
        # 사용자는 OpenCV 영상 창을 보며 현재 기능을 확인할 수 있습니다.
        # Panel은 영상 오른쪽에 별도로 위치합니다.

        separator_y = 340

        cv2.line(
            display_frame,
            (
                width + 10,
                separator_y,
            ),
            (
                width
                + PANEL_WIDTH
                - 10,
                separator_y,
            ),
            (120, 120, 120),
            1,
        )

        controls = [
            "1-6 : Target",
            "0   : Clear",
            "R   : Reset",
            "Q   : Quit",
        ]

        for index, text in enumerate(
            controls
        ):
            cv2.putText(
                display_frame,
                text,
                (
                    panel_x,
                    370 + index * 24,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        # ======================================================
        # ㊳ IP Webcam 화면과 Runtime Panel을 창에 표시
        # ======================================================
        # cv2.imshow()로 결합된 영상을 OpenCV 창에 표시합니다.
        # 이 코드는 FastAPI 웹 페이지가 아니라 PC의 OpenCV GUI 창에서 동작합니다.
        # 창이 활성화되어 있어야 cv2.waitKey()의 키 입력을 받을 수 있습니다.

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        # ======================================================
        # ㊴ Frame별 Runtime 실행 정보 로그에 추가
        # ======================================================
        # log_rows에 현재 Frame의 번호와 경과 시간, Target·Segment를 기록합니다.
        # Window 설정·Buffer 준비 상태·유효 Feature 비율도 저장합니다.
        # Raw·Threshold·Stable·Confidence·prediction_updated·FPS를 기록합니다.
        # 이 로그는 종료할 때 CSV로 저장하며 영상 MP4는 저장하지 않습니다.
        # 키 입력 처리 전에 기록하므로 키 변경 내용은 다음 Frame부터 반영됩니다.

        log_rows.append(
            {
                "frame_id": frame_id,
                "elapsed_sec": round(
                    elapsed,
                    4,
                ),
                "expected_label": (
                    expected_label
                ),
                "segment_id": (
                    segment_id
                ),
                "window_seconds": (
                    window_seconds
                ),
                "confidence_threshold": (
                    confidence_threshold
                ),
                "stabilization_buffer": (
                    stabilization_buffer
                ),
                "stabilization_min_votes": (
                    stabilization_min_votes
                ),
                "keep_buffer_on_target": int(
                    keep_buffer_on_target
                ),
                "buffer_duration_sec": round(
                    feature_buffer.duration(),
                    4,
                ),
                "valid_feature_ratio": round(
                    feature_buffer.valid_ratio(),
                    4,
                ),
                "raw_label": raw_label,
                "threshold_label": (
                    threshold_label
                ),
                "stable_label": (
                    stable_label
                ),
                "confidence": round(
                    confidence,
                    4,
                ),
                "prediction_updated": (
                    prediction_updated
                ),
                "processing_fps": round(
                    processing_fps,
                    3,
                ),
            }
        )

        # ======================================================
        # ㊵ 키보드 입력을 읽고 Q를 누르면 종료
        # ======================================================
        # cv2.waitKey(1)로 OpenCV 창의 키 입력을 확인합니다.
        # & 0xFF는 하위 8비트 키 코드만 사용합니다.
        # q 또는 Q를 누르면 while 반복문을 끝내고 로그 저장 단계로 이동합니다.

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key in (
            ord("q"),
            ord("Q"),
        ):
            break

        # ======================================================
        # ㊶ 1~6 숫자키로 Target Action과 구간 번호 변경
        # ======================================================
        # 키가 KEY_TO_ACTION에 있으면 해당 Action을 expected_label로 설정합니다.
        # segment_id를 1 증가시켜 새로운 Target 구간을 표시합니다.
        # Target은 정답 참조용이며 Random Forest의 예측을 바꾸지 않습니다.
        # KEEP/RESET 설정에 따라 다음 단계에서 Window 처리 방식이 달라집니다.

        if key in KEY_TO_ACTION:
            expected_label = (
                KEY_TO_ACTION[key]
            )
            segment_id += 1

            # ======================================================
            # ㊷ RESET 모드에서는 Target 변경 시 Buffer 초기화
            # ======================================================
            # --keep-buffer-on-target 옵션이 없는 기본 RESET 모드라면
            # Feature Buffer와 Stabilizer 기록을 모두 비웁니다.
            # 마지막 예측 시각과 Label·Confidence를 초기화하여
            # 새 Target의 움직임이 충분히 쌓인 뒤 예측하도록 합니다.
            # KEEP 모드에서는 이 블록을 실행하지 않습니다.

            if not keep_buffer_on_target:
                feature_buffer.clear()
                stabilizer.clear()
                last_prediction_time = (
                    -1e9
                )
                raw_label = (
                    "WARMING_UP"
                )
                threshold_label = (
                    "WARMING_UP"
                )
                stable_label = (
                    "WARMING_UP"
                )
                confidence = 0.0

            print(
                "Target:",
                expected_label,
                "| segment:",
                segment_id,
                "| buffer:",
                (
                    "KEEP"
                    if keep_buffer_on_target
                    else "RESET"
                ),
            )

        # ======================================================
        # ㊸ 0키로 Target 해제하고 Buffer도 초기화
        # ======================================================
        # 0키는 expected_label을 빈 값으로 바꿉니다.
        # segment_id를 증가시키고 Feature Buffer와 Stabilizer를 초기화합니다.
        # KEEP 모드여도 0키는 Buffer를 초기화합니다.
        # Label을 WARMING_UP, Confidence를 0으로 돌립니다.

        elif key == ord("0"):
            expected_label = ""
            segment_id += 1
            feature_buffer.clear()
            stabilizer.clear()
            last_prediction_time = (
                -1e9
            )
            raw_label = (
                "WARMING_UP"
            )
            threshold_label = (
                "WARMING_UP"
            )
            stable_label = (
                "WARMING_UP"
            )
            confidence = 0.0

            print(
                "Target cleared"
            )

        # ======================================================
        # ㊹ R키로 시간 Window와 안정화 기록 초기화
        # ======================================================
        # r/R키를 누르면 Feature Buffer와 Stabilizer를 비웁니다.
        # Label과 Confidence도 초기 상태로 되돌립니다.
        # Target이 설정되어 있으면 segment_id를 증가시키지만 Target 자체는 유지합니다.
        # 다시 Window가 채워져야 새 예측이 가능합니다.

        elif key in (
            ord("r"),
            ord("R"),
        ):
            feature_buffer.clear()
            stabilizer.clear()
            last_prediction_time = (
                -1e9
            )
            raw_label = (
                "WARMING_UP"
            )
            threshold_label = (
                "WARMING_UP"
            )
            stable_label = (
                "WARMING_UP"
            )
            confidence = 0.0

            if expected_label:
                segment_id += 1

            print(
                "Buffers reset"
            )

        # ======================================================
        # ㊺ 다음 Frame 재시도 및 Stream 끊김 처리
        # ======================================================
        # read_frame_with_retry()를 다시 호출해 다음 Frame을 받습니다.
        # 여러 번 실패한 경우 안내 메시지를 출력하고 반복문을 종료합니다.
        # Frame을 정상적으로 받았다면 frame_id를 1 증가시킵니다.
        # 현재 Frame은 한 번만 처리하고 다음 반복에서 새 Frame을 사용합니다.

        success, frame = (
            read_frame_with_retry(
                capture
            )
        )

        if (
            not success
            or frame is None
        ):
            print(
                "IP Webcam Frame을 여러 번 "
                "재시도했지만 읽지 못했습니다."
            )
            break

        frame_id += 1

    # ======================================================
    # ㊻ 영상 자원 해제 및 로그 존재 여부 검사
    # ======================================================
    # capture.release()로 IP Webcam 스트림 객체를 닫습니다.
    # cv2.destroyAllWindows()로 OpenCV 표시 창을 닫습니다.
    # 처리한 Frame 로그가 없으면 RuntimeError를 발생시킵니다.

    capture.release()
    cv2.destroyAllWindows()

    if not log_rows:
        raise RuntimeError(
            "저장할 Runtime Log가 "
            "없습니다."
        )

    # ======================================================
    # ㊼ 로그 출력 폴더·시각·실험 조건 파일명 생성
    # ======================================================
    # 출력 폴더를 만든 뒤 현재 날짜·시간을 파일명에 넣습니다.
    # Window 길이(w), Confidence(c), Buffer 크기(s)를 문자열로 만듭니다.
    # 파일명 예: ip_webcam_w2_c0.4_s5_20261110_151200.csv
    # 각 실험 로그를 구분하기 위한 이름이며 정확한 시간은 실행 시 달라집니다.

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    option_name = (
        f"w{window_seconds:g}_"
        f"c{confidence_threshold:g}_"
        f"s{stabilization_buffer}"
    )

    output_log = (
        output_dir
        / (
            "ip_webcam_"
            f"{option_name}_"
            f"{timestamp}.csv"
        )
    )

    # ======================================================
    # ㊽ Frame별 Runtime 로그를 CSV로 저장
    # ======================================================
    # csv.DictWriter를 사용해 로그 딕셔너리 목록을 저장합니다.
    # 첫 번째 행의 Key를 CSV 헤더로 사용합니다.
    # UTF-8-SIG로 저장하여 Excel에서 한글을 읽기 쉽게 합니다.
    # 일일 실행 상태와 키로 지정한 Target 구간을 다시 분석할 수 있습니다.

    with output_log.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                log_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            log_rows
        )

    # ======================================================
    # ㊾ 유효 FPS 평균 계산 및 결과 위치 출력
    # ======================================================
    # processing_fps가 0보다 큰 Frame만 평균에 포함합니다.
    # 유효한 값이 없다면 mean_fps를 0으로 설정합니다.
    # 평균 FPS와 저장한 CSV 경로를 출력합니다.
    # 이는 영상이 실제 처리된 속도이며 IP Webcam의 카메라 FPS와 다릅니다.

    fps_values = [
        float(
            row[
                "processing_fps"
            ]
        )
        for row in log_rows
        if float(
            row[
                "processing_fps"
            ]
        ) > 0
    ]

    mean_fps = (
        sum(fps_values)
        / len(fps_values)
        if fps_values
        else 0.0
    )

    print()
    print(
        "Mean processing FPS: "
        f"{mean_fps:.2f}"
    )
    print(
        f"Saved: {output_log}"
    )


# ======================================================
# ㊿ Python 파일을 직접 실행할 때 main 호출
# ======================================================
# python scripts/44_run_ip_webcam_action.py로 직접 실행하면 main()을 호출합니다.
# IP Webcam에서 Pose를 읽고 Action을 예측한 뒤 CSV 로그를 저장합니다.
# 이 파일을 다른 Python 모듈에서 import하는 경우 main()은 자동 실행되지 않습니다.

if __name__ == "__main__":
    main()