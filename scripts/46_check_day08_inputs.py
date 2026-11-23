from __future__ import annotations

# ==================================================
# ① 검사 프로그램에 필요한 라이브러리 가져오기
# ==================================================
# csv는 점검 결과를 저장하고, json은 설정 파일을 읽습니다.
# sys와 Path는 프로젝트 경로를 처리합니다.
# OpenCV는 MP4의 영상 정보를 검사하고 joblib은 학습 모델을 불러옵니다.

import csv
import json
import sys
from pathlib import Path

import cv2
import joblib


# ==================================================
# ② 프로젝트 ROOT 경로 찾기
# ==================================================
# 실행 파일이 scripts 폴더 안에 있다고 가정합니다.
# parents[1]을 이용해 scripts의 상위인 프로젝트 ROOT를 찾습니다.

ROOT = Path(__file__).resolve().parents[1]

# ==================================================
# ③ ROOT를 Python 모듈 검색 경로에 추가
# ==================================================
# src 폴더의 함수를 import할 수 있도록 ROOT를 검색 경로 맨 앞에 넣습니다.
# 이는 프로젝트 내부 파일을 정상적으로 불러오기 위한 준비입니다.

sys.path.insert(0, str(ROOT))


# ==================================================
# ④ 7일차 Runtime 검사·6일차 Feature 이름 함수 불러오기
# ==================================================
# validate_runtime_values()는 Window·Confidence·안정화 설정을 검사합니다.
# sequence_feature_names()는 Sequence Step에 따른 Feature 이름과 순서를 생성합니다.

from src.realtime_utils import (
    validate_runtime_values,
)
from src.sequence_utils import (
    sequence_feature_names,
)


# ==================================================
# ⑤ 8일차 SBU 실습 영상 폴더 지정
# ==================================================
# 8일차 다중 인물 추적 실습에 사용할 MP4 세 개가 들어 있는 폴더입니다.
# 영상은 이후 인물 추적 실험의 입력 자료이며 Random Forest 정답 데이터는 아닙니다.

VIDEO_DIR = ROOT / "data" / "day08" / "videos"

# ==================================================
# ⑥ 6일차 Random Forest 모델·Feature 목록 경로 지정
# ==================================================
# rf_action_baseline.joblib은 6일차에 학습한 Action 분류 모델입니다.
# rf_feature_columns.json은 모델이 학습할 때 사용한 Feature 이름과 순서입니다.
# 신뢰할 수 있는 출처의 joblib 파일만 불러와야 합니다.

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

# ==================================================
# ⑦ 6일차 Training Info JSON 경로 지정
# ==================================================
# 학습 당시 Action Class, Feature 개수, Sequence Step 수를 기록한 파일입니다.
# 8일차 환경과 6일차 모델의 입력 형식이 일치하는지 검사할 때 사용합니다.

TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)

# ==================================================
# ⑧ 7일차 및 8일차 설정 JSON 경로 지정
# ==================================================
# day07_realtime.json은 Window·Confidence·Stabilization 설정입니다.
# day08_multi_person.json은 다중 인물 추적의 Tracker 설정입니다.

DAY07_CONFIG_PATH = (
    ROOT
    / "configs"
    / "day07_realtime.json"
)

DAY08_CONFIG_PATH = (
    ROOT
    / "configs"
    / "day08_multi_person.json"
)

# ==================================================
# ⑨ 8일차 입력 점검 결과 CSV 저장 경로 지정
# ==================================================
# SBU 영상의 FPS, Frame 수, 해상도, 영상 길이 등을 표로 기록합니다.
# 출력 위치는 reports/day08/tables/day08_input_check.csv입니다.

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day08"
    / "tables"
    / "day08_input_check.csv"
)

# ==================================================
# ⑩ 필수 SBU 영상 3개와 실습 역할 정의
# ==================================================
# shaking_hands는 기본 추적, hugging은 가림(Occlusion),
# kicking_challenge는 빠른 상호작용 추적 점검에 사용합니다.
# 코드는 파일명과 실습 역할만 정의하며 실제 Tracking은 수행하지 않습니다.

EXPECTED_VIDEOS = {
    "01_sbu_shaking_hands.mp4": "BASIC_TRACKING",
    "02_sbu_hugging.mp4": "OCCLUSION_TRACKING",
    "03_sbu_kicking_challenge.mp4": "FAST_INTERACTION_TRACKING",
}

# ==================================================
# ⑪ 6일차 Baseline이 학습한 Action 6개 정의
# ==================================================
# 기존 Random Forest가 분류하는 bend_return 등 6개 Class입니다.
# SBU 영상의 shaking_hands·hugging·kicking은 이 분류 Class와 다릅니다.

EXPECTED_ACTIONS = {
    "bend_return",
    "leg_raise_lower",
    "walk_turn_walk",
    "drink_return",
    "sit_stand",
    "wave",
}

# ==================================================
# ⑫ Training Info에 반드시 있어야 할 항목 지정
# ==================================================
# classes는 학습 Class, feature_count는 입력 Feature 수,
# sequence_steps는 시간축 Sequence 길이입니다.
# 셋 중 하나라도 없으면 이어지는 검사가 불가능합니다.

REQUIRED_TRAIN_INFO_KEYS = {
    "classes",
    "feature_count",
    "sequence_steps",
}

# ==================================================
# ⑬ 7일차 실시간 설정의 필수 항목 정의
# ==================================================
# Window 크기·충족 비율·유효 Feature 수·예측 주기·Confidence·
# 안정화 Buffer와 최소 투표 수 등 8개 설정을 지정합니다.

REQUIRED_DAY07_KEYS = {
    "window_seconds",
    "min_window_fill_ratio",
    "min_valid_feature_ratio",
    "min_valid_features",
    "prediction_interval_sec",
    "confidence_threshold",
    "stabilization_buffer",
    "stabilization_min_votes",
}


# ==================================================
# ⑭ 공통 JSON 파일 읽기 함수
# ==================================================
# 지정된 JSON 파일을 UTF-8로 열어 Python 데이터로 변환합니다.
# 6일차 학습 정보 및 7·8일차 설정을 읽을 때 재사용합니다.

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ==================================================
# ⑮ 7일차 Runtime 설정 검증 함수 및 누락 검사
# ==================================================
# REQUIRED_DAY07_KEYS와 설정 JSON의 Key를 비교합니다.
# 누락된 항목이 있으면 이름을 표시하고 RuntimeError로 중단합니다.

def validate_day07_config(
    config: dict,
) -> None:
    missing = (
        REQUIRED_DAY07_KEYS
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

    # ==================================================
    # ⑯ Window·Confidence·안정화 설정값 공통 검증
    # ==================================================
    # validate_runtime_values()에 핵심 설정 네 가지를 전달합니다.
    # Window 시간, Confidence 임계값, Buffer 크기, 최소 투표수를 검사합니다.
    #
    validate_runtime_values(
        window_seconds=float(
            config["window_seconds"]
        ),
        confidence_threshold=float(
            config[
                "confidence_threshold"
            ]
        ),
        stabilization_buffer=int(
            config[
                "stabilization_buffer"
            ]
        ),
        stabilization_min_votes=int(
            config[
                "stabilization_min_votes"
            ]
        ),
    )

    # ==================================================
    # ⑰ Window 최소 충족 비율 검사
    # ==================================================
    # min_window_fill_ratio는 Window를 어느 정도 채워야 예측할지 정합니다.
    # 0보다 크고 1 이하여야 하며, 범위를 벗어나면 오류를 발생시킵니다.
    #
    if not (
        0
        < float(
            config[
                "min_window_fill_ratio"
            ]
        )
        <= 1
    ):
        raise RuntimeError(
            "min_window_fill_ratio는 "
            "0 초과 1 이하여야 합니다."
        )

    # ==================================================
    # ⑱ 유효 Feature 비율 검사
    # ==================================================
    # min_valid_feature_ratio는 Window 안에서 유효 Feature의 최소 비율입니다.
    # 0보다 크고 1 이하여야 합니다.
    #
    if not (
        0
        < float(
            config[
                "min_valid_feature_ratio"
            ]
        )
        <= 1
    ):
        raise RuntimeError(
            "min_valid_feature_ratio는 "
            "0 초과 1 이하여야 합니다."
        )

    # ==================================================
    # ⑲ 최소 유효 Feature 개수 검사
    # ==================================================
    # min_valid_features는 최소 몇 개의 유효 Feature가 필요한지 정합니다.
    # 2보다 작으면 실습 조건에 맞지 않으므로 중단합니다.
    #
    if int(
        config[
            "min_valid_features"
        ]
    ) < 2:
        raise RuntimeError(
            "min_valid_features는 "
            "2 이상이어야 합니다."
        )

    # ==================================================
    # ⑳ Action 예측 주기 검사
    # ==================================================
    # prediction_interval_sec는 예측을 새로 계산할 시간 간격입니다.
    # 0보다 커야 하며 0 이하라면 오류를 발생시킵니다.
    #
    if float(
        config[
            "prediction_interval_sec"
        ]
    ) <= 0:
        raise RuntimeError(
            "prediction_interval_sec는 "
            "0보다 커야 합니다."
        )


# ==================================================
# ㉑ MP4 상태 정보를 읽는 함수와 VideoCapture 생성
# ==================================================
# read_video_info()는 영상이 열리는지와 FPS·Frame 수·해상도를 점검합니다.
# OpenCV VideoCapture로 파일을 엽니다.

def read_video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(
        str(path)
    )

    # ==================================================
    # ㉒ MP4 파일을 열지 못했을 때 실패 정보 반환
    # ==================================================
    # capture.isOpened()가 False이면 open_ok=0을 반환합니다.
    # FPS·Frame 수·해상도·길이는 0으로 기록합니다.
    #
    if not capture.isOpened():
        return {
            "open_ok": 0,
            "fps": 0.0,
            "frames": 0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }

    # ==================================================
    # ㉓ FPS·Frame 수·가로·세로 해상도 읽기
    # ==================================================
    # CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_WIDTH,
    # CAP_PROP_FRAME_HEIGHT 속성으로 MP4의 기본 정보를 가져옵니다.
    #
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

    # ==================================================
    # ㉔ 첫 Frame 읽기 및 영상 객체 해제
    # ==================================================
    # 영상의 첫 Frame을 실제로 읽어 파일이 디코딩되는지 간단히 확인합니다.
    # 검사 후 capture.release()로 파일 사용 자원을 정리합니다.
    # 전체 Frame을 끝까지 읽어 검사하는 코드는 아닙니다.
    #
    success, frame = capture.read()
    capture.release()

    # ==================================================
    # ㉕ 영상 열림·첫 Frame·기본 속성 유효성 판단
    # ==================================================
    # 첫 Frame 읽기에 성공하고 Frame이 None이 아니며,
    # FPS·Frame 수·너비·높이가 모두 0보다 커야 open_ok=1입니다.
    #
    open_ok = int(
        success
        and frame is not None
        and fps > 0
        and frames > 0
        and width > 0
        and height > 0
    )

    # ==================================================
    # ㉖ 영상 길이를 FPS와 Frame 수로 계산
    # ==================================================
    # duration_sec = frames / fps로 영상의 길이를 계산합니다.
    # FPS가 0 이하이면 0초로 처리하여 0으로 나누기를 피합니다.
    #
    duration_sec = (
        frames / fps
        if fps > 0
        else 0.0
    )

    # ==================================================
    # ㉗ 개별 영상 검사 결과 딕셔너리 반환
    # ==================================================
    # open_ok, fps, frames, width, height, duration_sec를 반환합니다.
    # FPS와 영상 길이는 소수점 세 자리로 반올림해 저장합니다.
    #
    return {
        "open_ok": open_ok,
        "fps": round(
            fps,
            3,
        ),
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": round(
            duration_sec,
            3,
        ),
    }


# ==================================================
# ㉘ main 시작 및 반드시 존재해야 할 5개 파일 지정
# ==================================================
# 6일차 모델·Feature 목록·학습 정보와 7·8일차 설정의
# 총 5개 필수 파일을 목록으로 만듭니다.

def main() -> None:
    required_files = [
        MODEL_PATH,
        FEATURE_LIST_PATH,
        TRAIN_INFO_PATH,
        DAY07_CONFIG_PATH,
        DAY08_CONFIG_PATH,
    ]

    # ==================================================
    # ㉙ 필수 파일 5개 존재 여부 검사
    # ==================================================
    # 각 경로를 확인하여 파일이 없다면 FileNotFoundError로 중단합니다.
    # 존재만 확인하는 단계이며 파일 내용 검사는 다음 단계에서 수행합니다.
    #
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    # ==================================================
    # ㉚ 학습 정보·Feature 목록·설정 JSON 및 모델 불러오기
    # ==================================================
    # 세 가지 JSON과 day08 설정을 읽습니다.
    # joblib.load()로 6일차 Random Forest Baseline 모델을 불러옵니다.
    # 이 단계에서 모델 학습이나 Tracking을 새로 실행하지 않습니다.
    #
    training_info = load_json(
        TRAIN_INFO_PATH
    )
    feature_columns = load_json(
        FEATURE_LIST_PATH
    )
    day07_config = load_json(
        DAY07_CONFIG_PATH
    )
    day08_config = load_json(
        DAY08_CONFIG_PATH
    )
    model = joblib.load(
        MODEL_PATH
    )

    # ==================================================
    # ㉛ 학습 정보의 필수 Key 누락 검사
    # ==================================================
    # training_info에 classes, feature_count, sequence_steps가 있는지 검사합니다.
    # 누락된 Key를 안내하고 오류로 중단합니다.
    #
    missing_train_info = (
        REQUIRED_TRAIN_INFO_KEYS
        - set(training_info)
    )

    if missing_train_info:
        raise RuntimeError(
            "rf_training_info.json에 "
            "필수 정보가 없습니다: "
            + ", ".join(
                sorted(
                    missing_train_info
                )
            )
        )

    # ==================================================
    # ㉜ 7일차 Runtime 설정 검증 함수 실행
    # ==================================================
    # 앞에서 작성한 validate_day07_config()를 호출합니다.
    # 8일차에서도 사용하는 기존 Window·Confidence 설정의 기본 조건을 확인합니다.
    #
    validate_day07_config(
        day07_config
    )

    # ==================================================
    # ㉝ 6일차 Sequence Step 수 확인
    # ==================================================
    # training_info의 sequence_steps를 정수로 읽습니다.
    # 2 미만이면 시간 Sequence로 사용할 수 없으므로 중단합니다.
    #
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

    # ==================================================
    # ㉞ 예상 Feature 수와 학습 정보 Feature 수 비교
    # ==================================================
    # Frame당 38-D Feature이므로 예상 입력 수는 sequence_steps × 38입니다.
    # 학습 정보의 feature_count와 다른 경우 오류를 발생시킵니다.
    # 예를 들어 30-Step이면 30 × 38 = 1,140개입니다.
    #
    expected_feature_count = (
        sequence_steps
        * 38
    )

    training_feature_count = int(
        training_info[
            "feature_count"
        ]
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

    # ==================================================
    # ㉟ 현재 함수에서 만든 Feature 이름·순서와 저장 정보 비교
    # ==================================================
    # sequence_feature_names(sequence_steps)로 현재 Feature 컬럼을 생성합니다.
    # 저장된 rf_feature_columns.json과 리스트 전체를 비교하므로
    # 이름뿐 아니라 Feature 순서도 동일해야 합니다.
    #
    expected_columns = (
        sequence_feature_names(
            sequence_steps
        )
    )

    if (
        feature_columns
        != expected_columns
    ):
        raise RuntimeError(
            "6일차 Feature Column 순서와 "
            "현재 sequence_utils.py가 "
            "다릅니다."
        )

    # ==================================================
    # ㊱ 학습 Feature 컬럼 개수 확인
    # ==================================================
    # 저장한 Feature 목록 길이가 예상 Feature 수와 일치해야 합니다.
    # 30-Step × 38개라면 1,140개인지 검사합니다.
    #
    if (
        len(feature_columns)
        != expected_feature_count
    ):
        raise RuntimeError(
            "Feature Column 수가 "
            "sequence_steps × 38과 "
            "다릅니다."
        )

    # ==================================================
    # ㊲ 학습된 Random Forest의 입력 Feature 수 확인
    # ==================================================
    # model.n_features_in_로 학습된 모델의 입력 차원을 확인합니다.
    # 속성이 없으면 -1로 처리하고 예상 개수와 비교해 중단합니다.
    #
    model_feature_count = int(
        getattr(
            model,
            "n_features_in_",
            -1,
        )
    )

    if (
        model_feature_count
        != expected_feature_count
    ):
        raise RuntimeError(
            "Random Forest 입력 Feature 수가 "
            "6일차 설정과 다릅니다."
        )

    # ==================================================
    # ㊳ 모델·Training Info·기대 Action Class 6종 비교
    # ==================================================
    # model.classes_와 training_info의 classes를 문자열 집합으로 바꿉니다.
    # 둘 다 EXPECTED_ACTIONS와 일치해야 합니다.
    # 집합 비교이므로 이 단계는 Class 이름 구성을 검사하며 순서는 검사하지 않습니다.
    #
    model_classes = set(
        map(
            str,
            model.classes_,
        )
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
        != EXPECTED_ACTIONS
        or training_classes
        != EXPECTED_ACTIONS
    ):
        raise RuntimeError(
            "6일차 Action Class가 "
            "현재 A001~A006과 다릅니다."
        )

    # ==================================================
    # ㊴ 8일차 ByteTrack Tracker 설정 확인
    # ==================================================
    # day08_multi_person.json의 tracker가 bytetrack.yaml인지 검사합니다.
    # 다른 Tracker가 지정되면 RuntimeError를 발생시킵니다.
    # 이 단계에서는 ByteTrack을 실행하지 않습니다.
    #
    if (
        day08_config.get(
            "tracker"
        )
        != "bytetrack.yaml"
    ):
        raise RuntimeError(
            "8일차 기본 실습 Tracker는 "
            "bytetrack.yaml입니다."
        )

    # ==================================================
    # ㊵ 인물 추적 유실 허용 시간 검사
    # ==================================================
    # lost_timeout_seconds는 인물 추적이 끊긴 뒤의 기준 시간 설정입니다.
    # 0보다 커야 하며 누락된 경우 기본 0으로 검사되어 오류가 납니다.
    #
    if float(
        day08_config.get(
            "lost_timeout_seconds",
            0,
        )
    ) <= 0:
        raise RuntimeError(
            "lost_timeout_seconds는 "
            "0보다 커야 합니다."
        )

    # ==================================================
    # ㊶ 영상 결과 목록 생성 및 필수 SBU 파일 순회
    # ==================================================
    # rows에 영상별 검사 결과를 하나씩 추가할 예정입니다.
    # EXPECTED_VIDEOS의 3개 파일명과 실습 역할을 순서대로 확인합니다.
    #
    rows = []

    for filename, role in (
        EXPECTED_VIDEOS.items()
    ):
        path = (
            VIDEO_DIR
            / filename
        )

        # ==================================================
        # ㊷ SBU 영상 파일 존재 여부 확인
        # ==================================================
        # 필수 파일 하나라도 없으면 FileNotFoundError로 중단합니다.
        # 폴더에 다른 파일이 있어도 이 목록의 정확한 파일명을 확인합니다.
        #
        if not path.exists():
            raise FileNotFoundError(
                path
            )

        # ==================================================
        # ㊸ MP4를 열어 FPS·길이·해상도 확인
        # ==================================================
        # read_video_info()로 영상 정보와 첫 Frame 판독 성공 여부를 검사합니다.
        # open_ok가 1이 아니면 영상 경로를 포함한 오류로 중단합니다.
        #
        info = read_video_info(
            path
        )

        if info["open_ok"] != 1:
            raise RuntimeError(
                "영상을 정상적으로 "
                f"읽을 수 없습니다: {path}"
            )

        # ==================================================
        # ㊹ 영상별 결과 행 추가 및 활용 목적 구분
        # ==================================================
        # video, role, FPS, Frame 수, 너비, 높이, 길이를 기록합니다.
        # use_for_tracking=1은 추적 실습 입력이라는 뜻입니다.
        # use_as_rf_ground_truth=0은 RF 6개 Action의 정답으로 사용하지 않는다는 뜻입니다.
        #
        rows.append(
            {
                "video": filename,
                "role": role,
                "fps": info["fps"],
                "frames": (
                    info["frames"]
                ),
                "width": (
                    info["width"]
                ),
                "height": (
                    info["height"]
                ),
                "duration_sec": (
                    info[
                        "duration_sec"
                    ]
                ),
                "use_for_tracking": 1,
                "use_as_rf_ground_truth": 0,
            }
        )

    # ==================================================
    # ㊺ CSV 출력 폴더 생성
    # ==================================================
    # 출력 디렉터리가 없으면 부모 폴더까지 생성합니다.
    # exist_ok=True로 이미 있는 폴더도 안전하게 사용합니다.
    #
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # ㊻ 영상 3개 점검 결과를 UTF-8 CSV로 저장
    # ==================================================
    # DictWriter로 컬럼 이름을 첫 행에 쓰고 SBU 3개 결과를 저장합니다.
    # utf-8-sig를 사용하므로 Windows Excel에서도 열기 쉽습니다.
    # 기존 같은 파일이 있다면 덮어씁니다.
    #
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
        writer.writerows(
            rows
        )

    # ==================================================
    # ㊼ Sequence·Feature·Action·Tracker 점검 결과 요약
    # ==================================================
    # Sequence Step 수와 Feature 개수 및 학습 Class를 출력합니다.
    # SBU 영상 3개가 모두 검사되었고 Tracker 설정이 bytetrack.yaml임을 표시합니다.
    #
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
        "SBU videos     : 3 / 3"
    )
    print(
        "Tracker        : "
        "bytetrack.yaml"
    )
    print()

    # ==================================================
    # ㊽ 각 SBU 영상의 Frame 수·FPS·길이 출력
    # ==================================================
    # rows의 각 영상을 반복하며 이름·전체 Frame 수·FPS·재생 길이를 표시합니다.
    # 학생은 영상별 길이와 촬영 속도 차이를 확인할 수 있습니다.
    #
    for row in rows:
        print(
            f"{row['video']:34s} "
            f"frames={row['frames']:3d} "
            f"fps={row['fps']:.3f} "
            f"duration="
            f"{row['duration_sec']:.3f}s"
        )

    # ==================================================
    # ㊾ 검사 결과 CSV 경로와 READY 메시지 출력
    # ==================================================
    # 파일 저장 경로와 Day08 input data: READY 메시지를 출력합니다.
    # 이 메시지는 필수 파일·설정·영상 열림 검사가 통과했다는 뜻으로,
    # 인물 추적 정확도나 6-Class RF 예측 성능을 검증했다는 뜻은 아닙니다.
    #
    print()
    print(
        f"Saved: "
        f"{OUTPUT_PATH}"
    )
    print(
        "Day08 input data: READY"
    )


# ==================================================
# ㊿ 파일 직접 실행 시 main 함수 호출
# ==================================================
# python scripts/46_check_day08_inputs.py로 실행하면 main()을 호출합니다.
# 다른 코드에서 import한 경우에는 자동으로 실행되지 않습니다.

if __name__ == "__main__":
    main()