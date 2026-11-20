from __future__ import annotations

# =================================================
# ① 라이브러리 가져오기
# =================================================
# csv는 검사 결과를 CSV로 저장합니다.
# json은 설정과 학습 정보를 읽습니다.
# sys와 Path는 프로젝트 경로를 관리합니다.
# cv2는 MP4의 FPS·Frame 수·첫 Frame을 확인합니다.
# joblib은 6일차 학습 모델을 불러옵니다.

import csv
import json
import sys
from pathlib import Path

import cv2
import joblib


# =================================================
# ② 프로젝트 ROOT를 Python 모듈 검색 경로에 추가
# =================================================
# 이 파일이 scripts 폴더에 있을 때 parents[1]은 프로젝트 ROOT입니다.
# sys.path.insert(0, str(ROOT))로 ROOT를 검색 경로 맨 앞에 넣습니다.
# 이후 src.sequence_utils를 import할 수 있도록 준비합니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# =================================================
# ③ 6일차 Feature 이름 생성 함수 가져오기
# =================================================
# sequence_feature_names()는 Step 수에 맞는 Feature 컬럼 이름을 생성합니다.
# 7일차 실행 환경과 6일차 학습 당시 Feature 이름·순서가
# 완전히 일치하는지 검사할 때 사용합니다.

from src.sequence_utils import (
    sequence_feature_names,
)


# =================================================
# ④ 6일차 학습 모델 및 기록 파일 경로 지정
# =================================================
# MODEL_PATH: 35번 코드에서 저장한 Random Forest 모델입니다.
# FEATURE_LIST_PATH: 학습에 사용한 Feature 이름·순서 JSON입니다.
# TRAIN_INFO_PATH: 클래스·Feature 수·Step 수 등의 학습 정보입니다.
# 세 파일이 같은 학습 결과인지 이어서 검사합니다.

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
)

FEATURE_LIST_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_feature_columns.json"
)

TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)

# =================================================
# ⑤ 프로젝트 및 7일차 실시간 설정 경로 지정
# =================================================
# project.json: 프로젝트 설정 파일입니다. 이 코드에서는 존재 여부만 확인합니다.
# day07_realtime.json: Window 길이, 최소 충족 비율,
# 예측 주기, Confidence 기준, 안정화 투표 설정 등을 담습니다.

PROJECT_CONFIG_PATH = (
    ROOT
    / "configs"
    / "project.json"
)

DAY07_CONFIG_PATH = (
    ROOT
    / "configs"
    / "day07_realtime.json"
)

# =================================================
# ⑥ S02 원본 MP4 폴더와 검사 결과 CSV 경로 지정
# =================================================
# VIDEO_ROOT에는 S02의 Action MP4 6개가 있다고 가정합니다.
# OUTPUT_PATH에는 영상별 FPS·Frame 수·길이·Warm-up 가능 여부를
# day07_input_check.csv로 기록합니다.

VIDEO_ROOT = (
    ROOT
    / "data"
    / "day05"
    / "videos"
    / "S02"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day07"
    / "tables"
    / "day07_input_check.csv"
)

# =================================================
# ⑦ 6개 Action ID와 정답 이름 연결
# =================================================
# A001~A006 각각에 학습 Label 이름을 연결합니다.
# 이 Label들은 저장된 모델과 Training Info의 Class 구성 검사에 사용됩니다.
# 또한 S02_A001_T01.mp4부터 S02_A006_T01.mp4까지 검사합니다.

EXPECTED_ACTIONS = {
    "A001": "bend_return",
    "A002": "leg_raise_lower",
    "A003": "walk_turn_walk",
    "A004": "drink_return",
    "A005": "sit_stand",
    "A006": "wave",
}

# =================================================
# ⑧ 학습 정보 JSON의 필수 Key 지정
# =================================================
# 7일차 연결에 꼭 필요한 학습 정보는 다음 세 가지입니다.
# classes: 학습한 Action 이름 목록
# feature_count: 학습 모델의 Feature 수
# sequence_steps: 동작을 표현한 시간 Step 수

REQUIRED_TRAIN_INFO_KEYS = {
    "classes",
    "feature_count",
    "sequence_steps",
}


# =================================================
# ⑨ JSON 파일을 읽는 공통 함수 정의
# =================================================
# path로 전달받은 JSON을 UTF-8로 열어 json.load()로 읽습니다.
# 학습 Feature 목록·학습 정보·실시간 설정을 같은 방식으로 읽습니다.

def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# =================================================
# ⑩ 실시간 설정의 필수 항목 누락 검사
# =================================================
# validate_runtime_config()는 day07_realtime.json의 필수 Key 8개를 검사합니다.
# required - set(config)로 누락된 항목을 구합니다.
# 누락 Key가 있다면 목록을 보여주고 RuntimeError로 중단합니다.

def validate_runtime_config(
    config: dict,
) -> None:
    required = {
        "window_seconds",
        "min_window_fill_ratio",
        "min_valid_feature_ratio",
        "min_valid_features",
        "prediction_interval_sec",
        "confidence_threshold",
        "stabilization_buffer",
        "stabilization_min_votes",
    }

    missing = (
        required
        - set(config)
    )

    if missing:
        raise RuntimeError(
            "day07_realtime.json에 "
            "필수 설정이 없습니다: "
            + ", ".join(
                sorted(missing)
            )
        )

    # =================================================
    # ⑪ 실시간 설정값을 숫자형으로 변환
    # =================================================
    # Window 길이·충족 비율·유효 Feature 조건·예측 주기·Confidence·
    # 안정화 버퍼·최소 득표수를 float 또는 int로 변환합니다.
    # 이후 설정값이 허용 범위인지 확인합니다.

    window_seconds = float(
        config["window_seconds"]
    )
    fill_ratio = float(
        config[
            "min_window_fill_ratio"
        ]
    )
    valid_ratio = float(
        config[
            "min_valid_feature_ratio"
        ]
    )
    min_valid_features = int(
        config[
            "min_valid_features"
        ]
    )
    prediction_interval = float(
        config[
            "prediction_interval_sec"
        ]
    )
    confidence = float(
        config[
            "confidence_threshold"
        ]
    )
    buffer_size = int(
        config[
            "stabilization_buffer"
        ]
    )
    min_votes = int(
        config[
            "stabilization_min_votes"
        ]
    )

    # =================================================
    # ⑫ Window 길이와 두 비율의 범위 검사
    # =================================================
    # window_seconds는 0보다 커야 합니다.
    # min_window_fill_ratio와 min_valid_feature_ratio는 0 초과 1 이하입니다.
    # 조건을 위반하면 설정 오류를 발생시킵니다.

    if window_seconds <= 0:
        raise RuntimeError(
            "window_seconds는 "
            "0보다 커야 합니다."
        )

    if not (
        0 < fill_ratio <= 1
    ):
        raise RuntimeError(
            "min_window_fill_ratio는 "
            "0 초과 1 이하여야 합니다."
        )

    if not (
        0 < valid_ratio <= 1
    ):
        raise RuntimeError(
            "min_valid_feature_ratio는 "
            "0 초과 1 이하여야 합니다."
        )

    # =================================================
    # ⑬ 최소 유효 Feature 수와 예측 주기 검사
    # =================================================
    # min_valid_features는 2 이상이어야 합니다.
    # prediction_interval_sec는 0보다 커야 합니다.
    # 두 기준 중 하나라도 어기면 RuntimeError를 발생시킵니다.

    if min_valid_features < 2:
        raise RuntimeError(
            "min_valid_features는 "
            "2 이상이어야 합니다."
        )

    if prediction_interval <= 0:
        raise RuntimeError(
            "prediction_interval_sec는 "
            "0보다 커야 합니다."
        )

    # =================================================
    # ⑭ Confidence 및 안정화 투표 설정 검사
    # =================================================
    # confidence_threshold는 0~1 범위여야 합니다.
    # stabilization_buffer는 1 이상이어야 합니다.
    # stabilization_min_votes는 1 이상이며 버퍼 크기를 초과할 수 없습니다.

    if not (
        0 <= confidence <= 1
    ):
        raise RuntimeError(
            "confidence_threshold는 "
            "0~1 범위여야 합니다."
        )

    if buffer_size < 1:
        raise RuntimeError(
            "stabilization_buffer는 "
            "1 이상이어야 합니다."
        )

    if not (
        1
        <= min_votes
        <= buffer_size
    ):
        raise RuntimeError(
            "stabilization_min_votes는 "
            "1 이상이고 "
            "stabilization_buffer보다 "
            "클 수 없습니다."
        )


# =================================================
# ⑮ MP4 정보를 읽는 함수와 영상 열기 검사
# =================================================
# cv2.VideoCapture()로 지정한 MP4 파일을 엽니다.
# 열리지 않으면 open_ok=0, FPS·Frame 수·길이를 0으로 반환합니다.
# 이 단계는 영상 재생이 아니라 입력 상태 점검입니다.

def read_video_info(
    path: Path,
) -> dict:
    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        return {
            "open_ok": 0,
            "fps": 0.0,
            "frames": 0,
            "duration_sec": 0.0,
        }

    # =================================================
    # ⑯ 영상의 FPS와 전체 Frame 수 읽기
    # =================================================
    # CAP_PROP_FPS에서 초당 Frame 수를 가져옵니다.
    # CAP_PROP_FRAME_COUNT에서 전체 Frame 수를 가져옵니다.
    # 두 값은 뒤에서 영상의 재생 시간을 계산하는 데 사용됩니다.

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )
    frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # =================================================
    # ⑰ 첫 Frame 읽기 및 VideoCapture 자원 해제
    # =================================================
    # capture.read()로 첫 Frame 하나를 읽습니다.
    # capture.release()로 영상 객체를 닫습니다.
    # 전체 Frame을 순차적으로 디코딩하는 검사는 아닙니다.

    success, frame = capture.read()
    capture.release()

    # =================================================
    # ⑱ 첫 Frame·FPS·Frame 수의 유효성 확인
    # =================================================
    # 첫 Frame을 읽지 못했거나 FPS/전체 Frame 수가 0 이하라면
    # open_ok=0을 반환합니다. 재생 시간은 0.0초로 기록합니다.
    # 정상이라면 뒤에서 duration_sec를 계산합니다.

    if (
        not success
        or frame is None
        or fps <= 0
        or frames <= 0
    ):
        return {
            "open_ok": 0,
            "fps": fps,
            "frames": frames,
            "duration_sec": 0.0,
        }

    # =================================================
    # ⑲ 영상 재생 시간 계산 및 정보 반환
    # =================================================
    # duration_sec = frames / fps로 추정 재생 시간을 구합니다.
    # FPS와 길이는 소수점 셋째 자리까지 반올림합니다.
    # 정상이면 open_ok=1과 FPS·Frame 수·재생 시간을 반환합니다.

    duration = (
        frames / fps
    )

    return {
        "open_ok": 1,
        "fps": round(
            fps,
            3,
        ),
        "frames": frames,
        "duration_sec": round(
            duration,
            3,
        ),
    }


# =================================================
# ⑳ main() 시작 및 필수 파일 5개 존재 확인
# =================================================
# 6일차 모델·Feature 목록·학습 정보, project.json,
# day07_realtime.json이 모두 존재하는지 확인합니다.
# 하나라도 없다면 FileNotFoundError가 발생합니다.

def main() -> None:
    required_files = [
        MODEL_PATH,
        FEATURE_LIST_PATH,
        TRAIN_INFO_PATH,
        PROJECT_CONFIG_PATH,
        DAY07_CONFIG_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    # =================================================
    # ㉑ 학습 모델과 세 JSON 읽기
    # =================================================
    # joblib.load()로 학습 모델을 불러옵니다.
    # Feature 목록, Training Info, day07 실시간 설정을 JSON으로 읽습니다.
    # project.json은 이 코드에서 열거나 검사하지 않고 존재만 확인합니다.
    # 주의: 신뢰할 수 없는 joblib 파일은 불러오지 않아야 합니다.

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = load_json(
        FEATURE_LIST_PATH
    )

    training_info = load_json(
        TRAIN_INFO_PATH
    )

    runtime_config = load_json(
        DAY07_CONFIG_PATH
    )

    # =================================================
    # ㉒ Training Info 필수 Key 검사
    # =================================================
    # classes·feature_count·sequence_steps가 모두 있는지 검사합니다.
    # 빠진 값이 있다면 오류 메시지를 보여주고
    # 6일차 최종 35번 모델 학습 코드를 다시 실행하도록 안내합니다.

    missing_train_info = (
        REQUIRED_TRAIN_INFO_KEYS
        - set(training_info)
    )

    if missing_train_info:
        raise RuntimeError(
            "rf_training_info.json에 "
            "7일차에 필요한 정보가 없습니다: "
            + ", ".join(
                sorted(
                    missing_train_info
                )
            )
            + "\n6일차 최종 수정본의 "
            "35_train_rf_baseline.py를 "
            "다시 실행하세요."
        )

    # =================================================
    # ㉓ 실시간 설정 검증과 Sequence Step 수 확인
    # =================================================
    # validate_runtime_config()로 8개 실시간 설정을 검사합니다.
    # Training Info에서 sequence_steps를 가져와 정수로 바꿉니다.
    # Step 수가 2보다 작으면 오류를 발생시킵니다.

    validate_runtime_config(
        runtime_config
    )

    sequence_steps = int(
        training_info[
            "sequence_steps"
        ]
    )

    if sequence_steps < 2:
        raise RuntimeError(
            "sequence_steps는 "
            "2 이상이어야 합니다."
        )

    # =================================================
    # ㉔ 학습 Feature 컬럼의 이름과 순서 일치 검사
    # =================================================
    # sequence_feature_names(sequence_steps)로 현재 코드의 컬럼 목록을 만듭니다.
    # 저장한 rf_feature_columns.json과 목록 전체를 비교합니다.
    # 이름뿐 아니라 순서까지 같아야 통과합니다.

    expected_feature_columns = (
        sequence_feature_names(
            sequence_steps
        )
    )

    if (
        feature_columns
        != expected_feature_columns
    ):
        raise RuntimeError(
            "6일차 모델의 Feature Column과 "
            "현재 sequence_utils.py의 "
            "Feature 순서가 다릅니다."
        )

    # =================================================
    # ㉕ 세 출처의 Feature 개수 구하기
    # =================================================
    # 예상 Feature 수는 sequence_steps × 38입니다.
    # Training Info의 feature_count와 모델의 n_features_in_를 읽습니다.
    # 학습 모델에 입력한 숫자 Feature 수를 교차 검증하기 위한 준비입니다.

    expected_feature_count = (
        sequence_steps
        * 38
    )

    training_feature_count = int(
        training_info[
            "feature_count"
        ]
    )

    model_feature_count = int(
        getattr(
            model,
            "n_features_in_",
            -1,
        )
    )

    # =================================================
    # ㉖ Feature 목록·학습 정보·모델 입력 차원 비교
    # =================================================
    # ① Feature 목록 길이 ② Training Info의 Feature 수
    # ③ 실제 모델의 n_features_in_가 예상값과 모두 같아야 합니다.
    # 하나라도 다르면 각각의 RuntimeError를 발생시킵니다.

    if (
        len(feature_columns)
        != expected_feature_count
    ):
        raise RuntimeError(
            "Feature Column 수가 "
            f"{expected_feature_count}개가 "
            f"아닙니다."
        )

    if (
        training_feature_count
        != expected_feature_count
    ):
        raise RuntimeError(
            "rf_training_info.json의 "
            "feature_count가 "
            "sequence_steps × 38과 "
            "다릅니다."
        )

    if (
        model_feature_count
        != expected_feature_count
    ):
        raise RuntimeError(
            "Random Forest 입력 Feature 수가 "
            "6일차 설정과 다릅니다."
        )

    # =================================================
    # ㉗ 모델 및 학습 정보의 Action Class 비교
    # =================================================
    # 학습 모델과 Training Info의 Action Class를 집합으로 만들고
    # EXPECTED_ACTIONS에 등록된 6개 Label 집합과 비교합니다.
    # 정확히 같지 않으면 7일차와 연결할 수 없으므로 중단합니다.

    model_classes = set(
        map(
            str,
            model.classes_,
        )
    )

    expected_classes = set(
        EXPECTED_ACTIONS.values()
    )

    training_classes = set(
        map(
            str,
            training_info[
                "classes"
            ],
        )
    )

    if (
        model_classes
        != expected_classes
        or training_classes
        != expected_classes
    ):
        raise RuntimeError(
            "6일차 모델 또는 Training Info의 "
            "Action Class가 A001~A006과 "
            "일치하지 않습니다."
        )

    # =================================================
    # ㉘ Window를 채우는 데 필요한 최소 영상 길이 계산
    # =================================================
    # required_duration = window_seconds × min_window_fill_ratio입니다.
    # 예를 들어 2초 Window에 최소 충족 비율 0.75면 1.5초입니다.
    # 이 값은 저장영상의 Warm-up 가능 여부를 판단하는 기준입니다.

    required_duration = (
        float(
            runtime_config[
                "window_seconds"
            ]
        )
        * float(
            runtime_config[
                "min_window_fill_ratio"
            ]
        )
    )

    # =================================================
    # ㉙ 영상별 결과 목록 준비 및 S02 MP4 경로 만들기
    # =================================================
    # rows에는 영상별 검사 결과를 모읍니다.
    # runtime_ready_count는 길이 기준을 충족한 영상 개수를 셉니다.
    # A001~A006을 돌며 S02_A001_T01.mp4 형식으로 경로를 생성합니다.

    rows = []
    runtime_ready_count = 0

    for action_id, action_label in (
        EXPECTED_ACTIONS.items()
    ):
        video_path = (
            VIDEO_ROOT
            / (
                f"S02_{action_id}_"
                "T01.mp4"
            )
        )

        # =================================================
        # ㉚ S02 MP4 존재 및 첫 Frame 읽기 검사
        # =================================================
        # 파일이 없으면 FileNotFoundError가 발생합니다.
        # read_video_info()로 FPS·Frame 수·재생 시간을 읽습니다.
        # open_ok가 1이 아니면 RuntimeError로 중단합니다.

        if not video_path.exists():
            raise FileNotFoundError(
                video_path
            )

        info = read_video_info(
            video_path
        )

        if info["open_ok"] != 1:
            raise RuntimeError(
                "S02 영상을 정상적으로 "
                "읽을 수 없습니다: "
                f"{video_path}"
            )

        # =================================================
        # ㉛ 영상 길이가 Warm-up 기준을 충족하는지 계산
        # =================================================
        # duration_sec >= required_duration이면 runtime_ready=1,
        # 그렇지 않으면 0으로 표시합니다.
        # 준비된 영상의 개수를 runtime_ready_count에 누적합니다.
        # 이 단계에서 모델 추론이나 Pose 품질을 검사하지는 않습니다.

        runtime_ready = int(
            info[
                "duration_sec"
            ]
            >= required_duration
        )

        runtime_ready_count += (
            runtime_ready
        )

        # =================================================
        # ㉜ 영상별 검사 결과를 한 행으로 기록
        # =================================================
        # Action ID·Label·상대경로·존재·FPS·Frame 수·길이·
        # 필요 Warm-up 시간·runtime_ready를 rows에 저장합니다.
        # 비교 결과는 후속 저장영상 실습의 영상 선택에 활용합니다.

        rows.append(
            {
                "action_id": action_id,
                "action_label": (
                    action_label
                ),
                "video_path": str(
                    video_path
                    .relative_to(ROOT)
                ),
                "exists": 1,
                "fps": info["fps"],
                "frames": (
                    info["frames"]
                ),
                "duration_sec": (
                    info[
                        "duration_sec"
                    ]
                ),
                "required_warmup_sec": (
                    round(
                        required_duration,
                        3,
                    )
                ),
                "runtime_ready": (
                    runtime_ready
                ),
            }
        )

    # =================================================
    # ㉝ 검사 결과 day07_input_check.csv 저장
    # =================================================
    # reports/day07/tables 폴더가 없다면 생성합니다.
    # DictWriter로 컬럼명과 영상 6개의 검사 결과를 저장합니다.
    # utf-8-sig로 저장하여 Excel에서도 한글을 읽기 쉽게 합니다.

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    # =================================================
    # ㉞ Feature·Action·S02 영상 검사 요약 출력
    # =================================================
    # Sequence Step 수, Feature 수, Action 목록을 출력합니다.
    # S02 영상 6개를 읽은 상태와 runtime_ready 개수를 출력합니다.
    # Runtime-ready는 영상 길이 기준을 충족한 수입니다.

    print(
        f"Sequence steps : "
        f"{sequence_steps}"
    )
    print(
        f"Feature count  : "
        f"{expected_feature_count}"
    )
    print(
        "Action classes :",
        sorted(
            model_classes
        ),
    )
    print(
        "S02 videos     : 6 / 6"
    )
    print(
        "Runtime-ready  : "
        f"{runtime_ready_count} / 6"
    )

    # =================================================
    # ㉟ Warm-up을 충족하지 못한 영상에 경고 출력
    # =================================================
    # runtime_ready_count가 6보다 작으면 WARNING을 출력합니다.
    # day07_input_check.csv에서 runtime_ready=1인 영상을
    # 첫 저장영상 실습에 선택하도록 안내합니다.
    # 경고만 출력하며 RuntimeError로 종료하지는 않습니다.

    if runtime_ready_count < 6:
        print()
        print(
            "[WARNING] 일부 S02 영상은 "
            "현재 Window의 Warm-up 시간보다 "
            "짧습니다."
        )
        print(
            "day07_input_check.csv에서 "
            "runtime_ready=1인 영상을 "
            "저장영상 첫 실습에 사용하세요."
        )

    # =================================================
    # ㊱ 결과 파일 경로와 READY 메시지 출력
    # =================================================
    # 저장된 CSV 경로를 출력합니다.
    # "Day07 input data: READY"를 출력하여 사전 검사를 마칩니다.
    # 주의: 일부 영상이 runtime_ready=0이어도 READY가 출력됩니다.

    print()
    print(
        f"Saved: "
        f"{OUTPUT_PATH}"
    )
    print(
        "Day07 input data: READY"
    )


# =================================================
# ㊲ Python 파일 직접 실행 시 main() 호출
# =================================================
# python scripts/41_check_day07_inputs.py로 실행하면 main()을 시작합니다.
# 다른 Python 코드에서 import할 때는 자동으로 실행되지 않습니다.

if __name__ == "__main__":
    main()