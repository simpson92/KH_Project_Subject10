from __future__ import annotations

# ===================================================
# ① 필요한 라이브러리 가져오기
# ===================================================
# argparse는 터미널 옵션을 읽고 csv·json은 결과 기록과 설정을 처리합니다.
# re는 MP4 파일명에서 Action ID를 찾아냅니다.
# time는 처리 시간을 측정하고 OpenCV는 영상 입출력을 담당합니다.
# joblib은 6일차 분류 모델, YOLO는 사람 Pose 모델을 불러옵니다.
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import joblib
from ultralytics import YOLO


# ===================================================
# ② 프로젝트 ROOT와 Python 모듈 경로 지정
# ===================================================
# 현재 파일이 scripts/ 아래에 있다고 가정합니다.
# parents[1]은 프로젝트 루트 폴더입니다.
# sys.path에 ROOT를 추가해 src/ 공통 모듈을 불러올 수 있습니다.
ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ===================================================
# ③ 프로젝트 공통 설정·실행 장치 함수 가져오기
# ===================================================
# load_config()로 project.json을 읽습니다.
# resolve_device()로 설정에 따라 CPU 또는 GPU 실행 장치를 결정합니다.
from src.common import (
    load_config,
    resolve_device,
)

# ===================================================
# ④ 실시간 Action Recognition 공통 함수 가져오기
# ===================================================
# TimedFeatureBuffer는 시간 Window에 Frame Feature를 쌓습니다.
# PredictionStabilizer는 여러 예측을 모아 표시 결과를 안정화합니다.
# FPSMeter는 처리 FPS를 계산합니다.
# 나머지 함수는 Pose→38-D Feature, Window 예측, 옵션 검증을 담당합니다.
from src.realtime_utils import (
    FPSMeter,
    PredictionStabilizer,
    TimedFeatureBuffer,
    frame_feature_from_pose,
    predict_from_window,
    validate_runtime_values,
)


# ===================================================
# ⑤ 7일차 설정 및 6일차 모델 파일 경로 지정
# ===================================================
# day07_realtime.json은 Window·Confidence·안정화 설정입니다.
# rf_training_info.json에는 모델 학습 시 Sequence Step 수가 기록됩니다.
# rf_action_baseline.joblib은 35번에서 만든 학습 모델입니다.
# 이번 코드는 모델을 새로 학습하지 않고 가져와 사용합니다.
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

# ===================================================
# ⑥ 기본 출력 폴더와 Action 정답 이름 정의
# ===================================================
# 기본 출력 폴더는 reports/day07/saved_video/입니다.
# ACTION_MAP은 A001~A006을 실제 Action 이름에 대응시킵니다.
# 이 목록은 파일명에 적힌 정답을 화면에 표시할 때 활용합니다.
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "reports"
    / "day07"
    / "saved_video"
)

ACTION_MAP = {
    "A001": "bend_return",
    "A002": "leg_raise_lower",
    "A003": "walk_turn_walk",
    "A004": "drink_return",
    "A005": "sit_stand",
    "A006": "wave",
}

# ===================================================
# ⑦ MP4 파일명 형식 확인용 정규식 준비
# ===================================================
# S02_A003_T01.mp4와 같은 이름에서 Action ID를 찾기 위한 패턴입니다.
# 형식이 맞지 않아도 영상 처리 자체를 멈추지는 않습니다.
# 그 경우 Expected Label은 빈 문자열로 남습니다.
FILENAME_PATTERN = re.compile(
    r"^S\d+_(A\d+)_T\d+\.mp4$",
    re.IGNORECASE,
)


# ===================================================
# ⑧ 터미널 옵션을 읽는 함수 만들기
# ===================================================
# argparse.ArgumentParser()를 사용해 실행 옵션을 정의합니다.
# --source는 필수 MP4 경로이며, 나머지 옵션은 선택사항입니다.
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )
    # ===================================================
    # ⑨ Window·Confidence·안정화 버퍼 실험 옵션 설정
    # ===================================================
    # --window-seconds는 시간 Window 길이입니다.
    # --confidence는 분류 결과를 수용할 Confidence 기준입니다.
    # --stabilization-buffer는 최근 예측을 모을 개수입니다.
    # 값을 생략하면 day07_realtime.json의 설정을 사용합니다.
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

    # ===================================================
    # ⑩ 출력 폴더 옵션을 추가하고 인자 반환
    # ===================================================
    # --output-dir은 결과 MP4와 CSV를 저장할 폴더입니다.
    # 값이 없으면 기본 출력 폴더를 사용합니다.
    # parse_args()로 실제 입력값을 반환합니다.
    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "결과 MP4와 CSV를 "
            "저장할 폴더"
        ),
    )

    return parser.parse_args()


# ===================================================
# ⑪ JSON 파일을 읽는 공통 함수 작성
# ===================================================
# Path로 받은 JSON 파일을 UTF-8로 엽니다.
# json.load()로 Python 딕셔너리 등의 데이터로 반환합니다.
def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ===================================================
# ⑫ MP4 파일명에서 기대 Action Label 찾기
# ===================================================
# 파일명이 정규식과 맞으면 A001 같은 Action ID를 추출합니다.
# ACTION_MAP에서 대응하는 Action 이름을 가져옵니다.
# 형식이 맞지 않거나 없는 ID라면 빈 문자열을 반환합니다.
# Expected는 파일명 기준 참조값이며 모델 예측이 아닙니다.
def infer_expected_label(
    source: Path,
) -> str:
    match = FILENAME_PATTERN.match(
        source.name
    )

    if match is None:
        return ""

    action_id = (
        match.group(1)
        .upper()
    )

    return ACTION_MAP.get(
        action_id,
        "",
    )


# ===================================================
# ⑬ main 시작 및 입력 MP4 존재 여부 확인
# ===================================================
# 터미널에서 받은 --source를 Path로 변환합니다.
# 해당 파일이 없다면 FileNotFoundError로 중단합니다.
def main() -> None:
    args = parse_args()

    source = Path(
        args.source
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    # ===================================================
    # ⑭ 프로젝트·7일차·학습 정보 설정 읽기
    # ===================================================
    # src.common.load_config()로 프로젝트 설정을 읽습니다.
    # day07_realtime.json과 rf_training_info.json도 읽습니다.
    # 이 세 설정이 영상·Pose·Action 추론 조건을 결정합니다.
    project_config = load_config()

    runtime_config = load_json(
        DAY07_CONFIG_PATH
    )

    training_info = load_json(
        TRAIN_INFO_PATH
    )

    # ===================================================
    # ⑮ 학습 Sequence 길이와 시간 Window 결정
    # ===================================================
    # sequence_steps는 6일차 모델의 학습 정보에서 가져옵니다.
    # --window-seconds를 입력했다면 그 값을 우선 사용하고,
    # 아니면 7일차 JSON의 window_seconds를 사용합니다.
    # Window 길이와 30-Step Resampling은 서로 다른 개념입니다.
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

    # ===================================================
    # ⑯ Confidence와 안정화 버퍼 크기 결정
    # ===================================================
    # 선택 옵션 --confidence와 --stabilization-buffer를 확인합니다.
    # 직접 지정하지 않았다면 7일차 JSON 기본값을 사용합니다.
    # 두 값은 예측 수용 기준과 결과의 흔들림에 영향을 줍니다.
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

    # ===================================================
    # ⑰ 안정화 투표 최소 개수 정하기
    # ===================================================
    # 버퍼 크기를 명령행에서 바꾸면 과반수인 size//2+1을 사용합니다.
    # 버퍼를 바꾸지 않으면 JSON의 stabilization_min_votes를 따릅니다.
    # 이는 최근 예측 중 같은 Label이 얼마나 모여야 하는지 정하는 기준입니다.
    if (
        args.stabilization_buffer
        is not None
    ):
        # Buffer 크기를 바꾸는 실험에서는
        # 과반수 기준도 함께 맞춥니다.
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

    # ===================================================
    # ⑱ 출력 경로 확정 및 Runtime 옵션 검증
    # ===================================================
    # 상대경로 output-dir은 ROOT를 기준으로 변환합니다.
    # validate_runtime_values()로 Window, Confidence, Buffer, Vote의
    # 허용 범위를 검사한 뒤 잘못된 설정이면 중단합니다.
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

    # ===================================================
    # ⑲ 실행 장치와 학습 모델·Pose 모델 준비
    # ===================================================
    # 프로젝트 설정의 device에 맞춰 CPU/GPU를 정합니다.
    # joblib.load()로 6일차 Random Forest를 불러옵니다.
    # YOLO()로 Pose 모델을 불러옵니다.
    # Pose 추출과 Action 분류는 서로 다른 모델이 담당합니다.
    device = resolve_device(
        project_config["device"]
    )

    action_model = joblib.load(
        MODEL_PATH
    )

    pose_model = YOLO(
        project_config[
            "pose_model"
        ]
    )

    # ===================================================
    # ⑳ 입력 MP4 열기 및 영상 정보 가져오기
    # ===================================================
    # cv2.VideoCapture()로 저장영상 파일을 엽니다.
    # 열기에 실패하면 예외가 발생합니다.
    # 원본 FPS, 영상 가로·세로 크기를 읽습니다.
    capture = cv2.VideoCapture(
        str(source)
    )

    if not capture.isOpened():
        raise RuntimeError(
            "영상을 열 수 없습니다: "
            f"{source}"
        )

    source_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # ===================================================
    # ㉑ 비정상 FPS와 해상도 처리
    # ===================================================
    # 원본 FPS가 0 이하이면 30.0을 대신 사용합니다.
    # 가로나 세로가 0 이하이면 VideoCapture를 닫고 중단합니다.
    # FPS 대체는 원본 영상의 실제 속도를 복원했다는 의미는 아닙니다.
    if source_fps <= 0:
        source_fps = 30.0

    if (
        width <= 0
        or height <= 0
    ):
        capture.release()
        raise RuntimeError(
            "영상 해상도를 확인할 수 "
            f"없습니다: {source}"
        )

    # ===================================================
    # ㉒ 결과 폴더와 실험별 MP4·CSV 파일명 생성
    # ===================================================
    # 출력 폴더를 만들고 Window(w), Confidence(c), Buffer(s)를
    # 파일명에 붙여 서로 다른 실험 결과를 구분합니다.
    # 예: S02_A001_T01_w2_c0.4_s5.mp4 및 같은 이름의 CSV
    # 실제 파일명 숫자는 실행 옵션에 따라 달라집니다.
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    option_name = (
        f"w{window_seconds:g}_"
        f"c{confidence_threshold:g}_"
        f"s{stabilization_buffer}"
    )

    output_video = (
        output_dir
        / (
            f"{source.stem}_"
            f"{option_name}.mp4"
        )
    )

    output_log = (
        output_dir
        / (
            f"{source.stem}_"
            f"{option_name}.csv"
        )
    )

    # ===================================================
    # ㉓ 출력 영상 VideoWriter 만들고 열기 확인
    # ===================================================
    # mp4v 코덱, 입력 FPS, 원본 해상도로 결과 MP4를 작성합니다.
    # VideoWriter 열기에 실패하면 입력 자원을 닫고 중단합니다.
    # 이 FPS는 저장 MP4의 재생 속도이며, 실제 추론 처리 FPS와 다릅니다.
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        source_fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "결과 영상을 저장할 수 "
            f"없습니다: {output_video}"
        )

    # ===================================================
    # ㉔ 시간 Window·예측 안정화 도구 생성
    # ===================================================
    # TimedFeatureBuffer는 최근 window_seconds 구간의 Feature를 보관합니다.
    # PredictionStabilizer는 buffer_size와 min_votes로 최근 예측을 안정화합니다.
    # 두 Buffer는 역할이 다릅니다: 하나는 Frame Feature, 하나는 예측 Label입니다.
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

    # ===================================================
    # ㉕ FPS·초기 Label·정답 참조·로그 상태 초기화
    # ===================================================
    # FPSMeter와 마지막 예측 시간을 준비합니다.
    # 처음에는 raw/threshold/stable을 WARMING_UP으로 표시합니다.
    # expected_label은 파일명에서 읽은 참조 정답입니다.
    # Frame 번호, 로그 목록, 전체 실행 시간 측정을 시작합니다.
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

    expected_label = (
        infer_expected_label(
            source
        )
    )

    frame_id = 0
    log_rows = []

    process_start = (
        time.perf_counter()
    )

    # ===================================================
    # ㉖ 영상 Frame을 순서대로 읽고 영상 시각 계산
    # ===================================================
    # while 반복문으로 저장영상의 Frame을 하나씩 읽습니다.
    # 끝에 도달하면 반복을 종료합니다.
    # video_time=frame_id/source_fps로 원본 영상에서의 시각을 계산합니다.
    # loop_start는 현재 Frame 처리에 걸리는 시간을 측정합니다.
    while True:
        success, frame = (
            capture.read()
        )

        if not success:
            break

        video_time = (
            frame_id
            / source_fps
        )

        loop_start = (
            time.perf_counter()
        )

        # ===================================================
        # ㉗ YOLO Pose 모델로 현재 Frame 추론
        # ===================================================
        # Frame을 YOLO Pose 모델에 입력합니다.
        # image_size와 person_confidence는 project.json에서 가져옵니다.
        # 결과 첫 항목([0])을 현재 Frame의 Pose 결과로 사용합니다.
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

        # ===================================================
        # ㉘ Pose를 38-D Feature로 만들고 시간 Buffer에 추가
        # ===================================================
        # frame_feature_from_pose()가 Pose 결과를 Frame Feature로 변환합니다.
        # 사용 가능한 Pose가 없다면 공통 함수의 정의에 따라 유효하지 않은 값이 나올 수 있습니다.
        # Feature와 video_time을 시간 Window에 차례로 넣습니다.
        feature = (
            frame_feature_from_pose(
                result
            )
        )

        feature_buffer.append(
            video_time,
            feature,
        )

        # ===================================================
        # ㉙ Window 예측 준비 상태 판단
        # ===================================================
        # is_ready()는 Window가 최소 길이를 채웠는지,
        # 유효 Feature 비율과 유효 Feature 개수가 기준을 충족하는지 검사합니다.
        # ready가 False이면 아직 Action 예측을 진행할 수 없습니다.
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

        # ===================================================
        # ㉚ 예측 실행 주기를 확인
        # ===================================================
        # 현재 Frame에서 prediction_updated를 0으로 초기화합니다.
        # Window가 준비되었고 마지막 예측으로부터 설정된
        # prediction_interval_sec 이상 지났을 때만 새 Action 예측을 진행합니다.
        # 이 간격은 실제 처리 시간 아닌 video_time 기준입니다.
        prediction_updated = 0

        if (
            ready
            and (
                video_time
                - last_prediction_time
            )
            >= float(
                runtime_config[
                    "prediction_interval_sec"
                ]
            )
        ):
            # ===================================================
            # ㉛ Window Sequence를 Random Forest로 분류
            # ===================================================
            # predict_from_window()가 Feature Buffer를 학습된 Sequence Step에 맞춰
            # Random Forest에 넣어 Action을 예측합니다.
            # raw_label은 원래 예측, threshold_label은 Confidence 기준 적용 결과입니다.
            # confidence는 모델의 예측 Confidence입니다.
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

            # ===================================================
            # ㉜ 예측 안정화 및 최근 갱신 시각 기록
            # ===================================================
            # stabilizer.update()로 최근 Threshold Label을 종합합니다.
            # stable_label에 화면 표시용 안정화 결과를 저장합니다.
            # last_prediction_time을 갱신하고 prediction_updated를 1로 표시합니다.
            stable_label = (
                stabilizer.update(
                    threshold_label
                )
            )

            last_prediction_time = (
                video_time
            )

            prediction_updated = 1

        # ===================================================
        # ㉝ Window가 준비되지 않았으면 WARMING_UP 유지
        # ===================================================
        # ready=False일 때 Raw·Threshold·Stable을 WARMING_UP으로 표시합니다.
        # Confidence도 0으로 바꿉니다.
        # Window가 준비되었지만 예측 간격이 안 지났다면 이전 값을 유지합니다.
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

        # ===================================================
        # ㉞ Pose 시각화와 실제 처리 FPS 계산
        # ===================================================
        # result.plot()으로 Pose가 그려진 Frame을 만듭니다.
        # FPSMeter는 처리 진행 시각을 이용해 처리 FPS를 계산합니다.
        # 입력 영상의 source_fps와 처리 FPS는 서로 다를 수 있습니다.
        annotated = (
            result.plot()
        )

        processing_fps = (
            fps_meter.update(
                time.perf_counter()
            )
        )

        # ===================================================
        # ㉟ 검사 결과를 표시할 배경과 문구 만들기
        # ===================================================
        # 영상 위에 검은 배경 사각형을 그리고 정보를 배치합니다.
        # Expected=파일명 정답, Raw=모델 원시 결과,
        # Threshold=기준 적용, Stable=안정화 결과입니다.
        # Confidence·Window 길이·처리 FPS도 문구에 포함합니다.
        cv2.rectangle(
            annotated,
            (10, 10),
            (690, 190),
            (0, 0, 0),
            -1,
        )

        texts = [
            (
                f"Expected: "
                f"{expected_label or '-'}"
            ),
            f"Raw: {raw_label}",
            (
                f"Threshold: "
                f"{threshold_label}"
            ),
            (
                f"Stable: "
                f"{stable_label}"
            ),
            (
                f"Conf: "
                f"{confidence:.2f}  "
                f"Window: "
                f"{window_seconds:.1f}s  "
                f"FPS: "
                f"{processing_fps:.1f}"
            ),
        ]

        # ===================================================
        # ㊱ 텍스트를 영상 프레임에 그리기
        # ===================================================
        # texts 목록을 반복하며 cv2.putText()로 영상에 표시합니다.
        # index에 따라 세로 위치를 조절합니다.
        # 한글이 아닌 영문 Label을 OpenCV 글꼴로 기록합니다.
        for index, text in enumerate(
            texts
        ):
            cv2.putText(
                annotated,
                text,
                (
                    25,
                    38
                    + index * 34,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
            )

        # ===================================================
        # ㊲ 주석이 표시된 Frame을 MP4에 저장
        # ===================================================
        # VideoWriter에 현재 Frame을 넣습니다.
        # 모든 Frame을 순서대로 저장해 Pose와 Action 결과가 표시된 MP4를 만듭니다.
        writer.write(
            annotated
        )

        # ===================================================
        # ㊳ Frame별 상세 진단 정보를 로그 목록에 기록
        # ===================================================
        # Frame 번호·영상 시각·기대 정답과 예측 Label을 기록합니다.
        # Window 길이·유효 Feature 비율·Confidence·예측 갱신 여부도 포함합니다.
        # 처리 FPS와 현재 Frame 처리 시간까지 저장해 나중에 분석할 수 있습니다.
        log_rows.append(
            {
                "frame_id": frame_id,
                "video_time_sec": round(
                    video_time,
                    4,
                ),
                "expected_label": (
                    expected_label
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
                "frame_processing_sec": round(
                    time.perf_counter()
                    - loop_start,
                    5,
                ),
            }
        )

        # ===================================================
        # ㊴ Frame 번호를 올리고 30 Frame마다 진행 상황 출력
        # ===================================================
        # Frame 번호를 1씩 증가시킵니다.
        # 30개 Frame마다 Stable Label·Confidence·처리 FPS를 출력합니다.
        # 실행 도중 모델이 어떤 결과를 내는지 간단히 확인할 수 있습니다.
        frame_id += 1

        if frame_id % 30 == 0:
            print(
                f"Frame {frame_id} "
                f"Stable="
                f"{stable_label} "
                f"Conf="
                f"{confidence:.2f} "
                f"FPS="
                f"{processing_fps:.1f}"
            )

    # ===================================================
    # ㊵ 영상 자원 해제와 Frame 처리 여부 확인
    # ===================================================
    # 입력 VideoCapture와 출력 VideoWriter를 닫습니다.
    # 처리한 Frame이 하나도 없다면 RuntimeError를 발생시킵니다.
    # 정상 종료 후에 로그 파일을 작성합니다.
    capture.release()
    writer.release()

    if not log_rows:
        raise RuntimeError(
            "처리한 Frame이 없습니다."
        )

    # ===================================================
    # ㊶ 총 처리 시간 측정 및 Frame 로그 CSV 저장
    # ===================================================
    # 전체 경과 시간을 계산합니다.
    # csv.DictWriter로 컬럼명을 작성한 뒤 Frame별 log_rows를 CSV로 저장합니다.
    # utf-8-sig는 Excel에서 한글을 읽기 쉽게 하는 인코딩입니다.
    total_elapsed = (
        time.perf_counter()
        - process_start
    )

    with output_log.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer_csv = (
            csv.DictWriter(
                file,
                fieldnames=list(
                    log_rows[0].keys()
                ),
            )
        )
        writer_csv.writeheader()
        writer_csv.writerows(
            log_rows
        )

    # ===================================================
    # ㊷ 전체 처리 FPS와 실제 예측 갱신 횟수 계산
    # ===================================================
    # overall_fps=처리한 Frame 수/실제 총 처리 시간입니다.
    # prediction_updated가 1인 로그 행을 합쳐 새 예측 횟수를 계산합니다.
    # 영상 FPS, 처리 FPS, 예측 갱신 횟수는 서로 다른 지표입니다.
    overall_fps = (
        frame_id
        / total_elapsed
        if total_elapsed > 0
        else 0.0
    )

    prediction_updates = sum(
        int(
            row[
                "prediction_updated"
            ]
        )
        for row in log_rows
    )

    # ===================================================
    # ㊸ 최종 통계·경고·저장 파일 경로 출력
    # ===================================================
    # 총 Frame 수와 예측 갱신 횟수를 출력합니다.
    # 예측 갱신이 0회면 긴 영상 또는 짧은 Window를 권장하는 경고를 표시합니다.
    # 마지막으로 처리 FPS, 결과 MP4, 로그 CSV 경로를 출력합니다.
    # 이 경고는 실제 Window 길이 외 유효 Feature 부족 등도 함께 살펴야 합니다.
    print()
    print(
        f"Frames: {frame_id}"
    )
    print(
        "Prediction updates: "
        f"{prediction_updates}"
    )

    if prediction_updates == 0:
        print(
            "[WARNING] Window가 충분히 "
            "채워지기 전에 영상이 끝났습니다."
        )
        print(
            "더 긴 S02 영상 또는 더 짧은 "
            "Window로 다시 확인하세요."
        )
    print(
        "Overall processing FPS: "
        f"{overall_fps:.2f}"
    )
    print(
        f"Video: {output_video}"
    )
    print(
        f"Log  : {output_log}"
    )


# ===================================================
# ㊹ Python 파일 직접 실행 시 main 호출
# ===================================================
# python scripts/42_run_saved_video_action.py --source ... 형식으로 실행합니다.
# 직접 실행할 때만 main()을 호출하며, import할 때는 자동 실행하지 않습니다.
if __name__ == "__main__":
    main()