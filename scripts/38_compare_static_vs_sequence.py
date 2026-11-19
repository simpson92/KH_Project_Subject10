
from __future__ import annotations

# ==================================================
# ① 모델 비교에 필요한 라이브러리 가져오기
# ==================================================
# json: Baseline 학습 설정을 읽습니다.
# sys와 Path: 프로젝트 경로 및 내부 함수 import를 처리합니다.
# NumPy와 Pandas: Feature 배열·CSV 데이터를 처리합니다.
# RandomForestClassifier: 두 표현 방식에 사용할 분류 모델입니다.
# accuracy_score, f1_score: Validation 결과를 같은 지표로 비교합니다.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ==================================================
# ② 프로젝트 ROOT와 모듈 검색 경로 설정
# ==================================================
# 현재 파일이 scripts 폴더에 있다고 가정합니다.
# parents[1]은 프로젝트 최상위(ROOT) 폴더입니다.
# sys.path.insert()로 ROOT를 모듈 검색 경로에 추가합니다.
# 이를 통해 src 폴더의 공통 함수를 불러올 수 있습니다.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT),
)


# ==================================================
# ③ Static Feature 생성에 필요한 공통 함수 불러오기
# ==================================================
# build_frame_feature_matrix():
# Keypoint CSV의 유효 Frame을 38-D Feature로 변환합니다.
#
# 38-D 구성:
# 정규화 좌표 34개 + 관절각도 4개
#
# middle_frame_feature():
# 유효 Frame 중 가운데 한 개의 Feature를 선택합니다.
#
# 두 함수는 src/sequence_utils.py에 정의되어 있습니다.

from src.sequence_utils import (
    build_frame_feature_matrix,
    middle_frame_feature,
)


# ==================================================
# ④ Baseline 설정 JSON 경로 지정
# ==================================================
# day06_baseline.json에는 Random Forest 설정이 있습니다.
#
# n_estimators: Decision Tree 개수
# max_depth: Tree 최대 깊이
# random_state: 난수 시드
# sequence_steps: Sequence 시간 Step 수
#
# Static과 Sequence에 동일한 모델 설정을 적용합니다.

CONFIG_PATH = (
    ROOT
    / "configs"
    / "day06_baseline.json"
)


# ==================================================
# ⑤ Static 비교용 5일차 Train·Validation 경로 지정
# ==================================================
# TRAIN_SPLIT:
# S01의 승인 Clip 6개가 기록된 Manifest입니다.
#
# VAL_SPLIT:
# S02의 승인 Clip 6개가 기록된 Manifest입니다.
#
# 각 Manifest에 기록된 keypoint_csv 경로를 읽습니다.
#
# Static 실험에서는 Clip마다
# 가운데 유효 Frame 하나의 38-D Feature를 사용합니다.

TRAIN_SPLIT = (
    ROOT
    / "data"
    / "day05"
    / "splits"
    / "train.csv"
)

VAL_SPLIT = (
    ROOT
    / "data"
    / "day05"
    / "splits"
    / "val.csv"
)


# ==================================================
# ⑥ Sequence 비교용 6일차 Train·Validation 경로 지정
# ==================================================
# 34번 코드에서 생성한 Sequence Dataset을 사용합니다.
#
# SEQUENCE_TRAIN:
# S01의 6개 Clip, 기본적으로 각 1,140-D Feature
#
# SEQUENCE_VAL:
# S02의 6개 Clip, 기본적으로 각 1,140-D Feature
#
# 30 Step × 38 Feature = 1,140개 숫자입니다.

SEQUENCE_TRAIN = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "train.csv"
)

SEQUENCE_VAL = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "val.csv"
)


# ==================================================
# ⑦ Static과 Sequence 비교 결과 저장 경로 지정
# ==================================================
# 두 실험의 결과를 하나의 CSV로 저장합니다.
#
# 저장 내용:
# representation
# feature_count
# accuracy
# macro_f1
#
# 이 코드는 학습 모델을 파일로 저장하지 않고
# 실험 성능 비교 결과만 저장합니다.

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "experiments"
    / "static_vs_sequence.csv"
)


# ==================================================
# ⑧ Sequence 모델 입력에서 제외할 기본정보 컬럼 지정
# ==================================================
# 이 8개 컬럼은 Clip을 설명하는 메타데이터입니다.
#
# action_label은 모델이 맞혀야 할 정답(y)입니다.
#
# 따라서 모델 입력 X에는 포함하지 않습니다.
#
# Sequence CSV에서는 이 컬럼들을 제외한
# 나머지 숫자 Feature를 모델 입력으로 사용합니다.

META_COLUMNS = {
    "clip_id",
    "subject",
    "action_id",
    "action_label",
    "take",
    "valid_frames",
    "first_valid_frame",
    "last_valid_frame",
}


# ==================================================
# ⑨ Baseline 모델 설정을 JSON에서 읽는 함수
# ==================================================
# CONFIG_PATH의 JSON 파일을 엽니다.
#
# json.load():
# JSON 내용을 Python 딕셔너리로 변환합니다.
#
# 읽은 설정을 반환하여 두 실험에 공통 적용합니다.

def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ==================================================
# ⑩ 같은 조건의 Random Forest 모델을 만드는 함수
# ==================================================
# make_model()은 새로운 Random Forest 객체를 생성합니다.
#
# n_estimators:
# Random Forest에서 사용할 Decision Tree 개수입니다.
#
# max_depth:
# Decision Tree의 최대 깊이입니다.
#
# random_state:
# 난수를 고정하여 비교 결과의 재현성을 높입니다.
#
# class_weight="balanced":
# Class 빈도를 고려한 가중치 옵션입니다.
#
# n_jobs=-1:
# 가능한 CPU 코어를 병렬로 사용합니다.
#
# 여기서는 모델을 생성할 뿐, 아직 학습하지 않습니다.
#
# 호출할 때마다 서로 다른 새 모델 객체가 만들어집니다.

def make_model(
    config: dict,
):
    return RandomForestClassifier(
        n_estimators=int(
            config[
                "n_estimators"
            ]
        ),
        max_depth=(
            config[
                "max_depth"
            ]
        ),
        random_state=int(
            config[
                "random_state"
            ]
        ),
        class_weight="balanced",
        n_jobs=-1,
    )


# ==================================================
# ⑪ Static Dataset 생성 함수와 Split Manifest 읽기
# ==================================================
# build_static_dataset()은 5일차 Split CSV를 입력받습니다.
#
# 해당 Manifest에는 각 Clip의 ID, Action 정답,
# Keypoint CSV 경로 등이 기록되어 있습니다.
#
# 이 함수는 Train과 Validation에서 각각 호출됩니다.
#
# Clip 하나에서 가운데 유효 Frame 하나를 선택하여
# Static Feature를 만드는 것이 목적입니다.

def build_static_dataset(
    manifest_path: Path,
):
    manifest = pd.read_csv(
        manifest_path
    )

    # ==============================================
    # ⑫ Static 입력·정답 목록과 Clip별 경로 준비
    # ==============================================
    # X:
    # Clip별 38-D Static Feature를 모을 목록입니다.
    #
    # y:
    # Clip별 실제 Action 정답을 모을 목록입니다.
    #
    # manifest.iterrows():
    # 각 Clip의 정보를 한 행씩 가져옵니다.
    #
    # keypoint_csv와 ROOT를 연결하여
    # 해당 Clip의 실제 Keypoint CSV 경로를 만듭니다.

    X = []
    y = []

    for _, item in (
        manifest.iterrows()
    ):
        keypoint_path = (
            ROOT
            / str(
                item[
                    "keypoint_csv"
                ]
            )
        )

        # ==========================================
        # ⑬ Keypoint CSV에서 38-D Frame Feature 생성
        # ==========================================
        # 현재 Clip의 Keypoint CSV를 읽습니다.
        #
        # build_frame_feature_matrix():
        # 유효한 Pose Frame을 골라
        # 각 Frame을 38-D Feature로 바꿉니다.
        #
        # 반환값:
        # matrix → 유효 Frame들의 Feature 배열
        # frame_ids → 해당 Frame의 원본 번호
        #
        # 이번 Static 실험에서는 frame_ids를
        # 사용하지 않으므로 _로 받습니다.
        #
        # 예:
        # 유효 Frame이 85개라면 matrix.shape=(85,38)

        df = pd.read_csv(
            keypoint_path
        )

        matrix, _ = (
            build_frame_feature_matrix(
                df
            )
        )

        # ==========================================
        # ⑭ 사용할 수 있는 유효 Frame이 있는지 확인
        # ==========================================
        # len(matrix)==0:
        # 38-D Feature를 생성한 Frame이 없는 상태입니다.
        #
        # 가운데 Frame도 선택할 수 없으므로
        # RuntimeError를 발생시킵니다.
        #
        # 문제 Clip의 ID를 오류 메시지에 포함합니다.

        if len(matrix) == 0:
            raise RuntimeError(
                "Static Feature를 "
                "만들 수 없습니다: "
                f"{item['clip_id']}"
            )

        # ==========================================
        # ⑮ 가운데 유효 Frame의 38-D Feature 선택
        # ==========================================
        # middle_frame_feature():
        # 유효 Frame 목록에서 가운데 Feature 하나를 가져옵니다.
        #
        # 원본 영상의 정확한 시간 중앙이 아니라
        # 유효 Frame 목록의 가운데입니다.
        #
        # 선택한 38-D Feature는 X에 추가합니다.
        #
        # Clip의 실제 Action Label은 y에 추가합니다.
        #
        # 모든 Clip에 이 작업을 반복합니다.

        feature = (
            middle_frame_feature(
                matrix
            )
        )

        X.append(feature)
        y.append(
            str(
                item[
                    "action_label"
                ]
            )
        )

    # ==============================================
    # ⑯ Static Feature X와 정답 y를 배열로 반환
    # ==============================================
    # np.stack(X):
    # 여러 Clip의 38-D Feature를 2차원 배열로 합칩니다.
    #
    # np.asarray(y):
    # Action 정답 목록을 NumPy 배열로 바꿉니다.
    #
    # 기본 수업:
    # X.shape = (6, 38)
    # y의 길이 = 6
    #
    # 즉 6개 Clip이 각각 38개 Feature를 갖습니다.

    return (
        np.stack(X),
        np.asarray(y),
    )


# ==================================================
# ⑰ 기존 Sequence CSV를 읽고 Feature 컬럼 선택
# ==================================================
# sequence_xy()는 34번에서 만든 Sequence CSV를 읽습니다.
#
# 이 CSV에는 8개 메타데이터와
# 기본적으로 1,140개 Sequence Feature가 있습니다.
#
# META_COLUMNS에 있는 기본정보를 제외합니다.
#
# 남은 컬럼을 모델 입력용 숫자 Feature로 사용합니다.
#
# 현재 코드는 Feature 컬럼 개수를 별도로 검사하지 않습니다.

def sequence_xy(
    path: Path,
):
    df = pd.read_csv(path)

    columns = [
        column
        for column in df.columns
        if column
        not in META_COLUMNS
    ]

    # ==============================================
    # ⑱ Sequence 입력 X와 Action 정답 y 생성
    # ==============================================
    # X:
    # Sequence Feature 숫자 배열입니다.
    #
    # y:
    # 각 Clip의 Action 정답입니다.
    #
    # to_numpy(dtype=float):
    # 모델 입력용 숫자 배열로 바꿉니다.
    #
    # 기본 수업:
    # X.shape = (6,1140)
    # y의 길이 = 6
    #
    # 두 배열을 반환하여 학습·평가 함수에 전달합니다.

    X = (
        df[columns]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        df[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    return X, y


# ==================================================
# ⑲ 공통 평가 함수에서 Random Forest 학습
# ==================================================
# score_model()은 두 실험이 공유하는 평가 함수입니다.
#
# 입력:
# model   → Random Forest 객체
# X_train → S01 학습 Feature
# y_train → S01 Action 정답
# X_val   → S02 Validation Feature
# y_val   → S02 Action 정답
#
# model.fit()으로 실제 학습을 진행합니다.
#
# Static과 Sequence는 서로 다른 모델 객체를
# 전달받아 각각 따로 학습합니다.

def score_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):
    model.fit(
        X_train,
        y_train,
    )

    # ==============================================
    # ⑳ Validation Action 예측 및 Class 목록 준비
    # ==============================================
    # model.predict(X_val):
    # S02의 Feature를 입력해 Action을 예측합니다.
    #
    # pred:
    # 모델이 예측한 Action Label 배열입니다.
    #
    # set(y_train):
    # 학습에 사용한 중복 없는 Action Class입니다.
    #
    # sorted():
    # Class 이름을 정렬합니다.
    #
    # 이후 Macro F1 계산에 이 Class 목록을 사용합니다.

    pred = model.predict(
        X_val
    )

    labels = sorted(
        set(y_train)
    )

    # ==============================================
    # ㉑ Accuracy와 Macro F1 계산 후 반환
    # ==============================================
    # accuracy_score():
    # Validation Clip 중 정답을 맞힌 비율입니다.
    #
    # f1_score(average="macro"):
    # 각 Action의 F1을 계산한 뒤 동일 비중으로 평균합니다.
    #
    # labels:
    # Train의 Action Class를 기준으로 계산합니다.
    #
    # zero_division=0:
    # 계산이 불가능한 지표를 0으로 처리합니다.
    #
    # 두 점수를 반환합니다.

    return (
        accuracy_score(
            y_val,
            pred,
        ),
        f1_score(
            y_val,
            pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    )


# ==================================================
# ㉒ main() 실행 및 Baseline 설정 읽기
# ==================================================
# main()에서 두 가지 비교 실험을 순서대로 실행합니다.
#
# 먼저 JSON 설정을 읽습니다.
#
# Static과 Sequence 모두
# 동일한 Random Forest 설정을 사용합니다.

def main() -> None:
    config = load_config()

    # ==============================================
    # ㉓ 실험 A: Static Train·Validation 데이터 생성
    # ==============================================
    # 5일차 Split Manifest를 읽습니다.
    #
    # S01의 각 Clip에서 가운데 유효 Frame 하나를 선택합니다.
    #
    # S02도 같은 방법으로 변환합니다.
    #
    # 각 Clip을 38-D Feature로 표현합니다.
    #
    # 기본 결과:
    # Static Train      → (6,38)
    # Static Validation → (6,38)
    #
    # 각 배열과 Action 정답을 따로 저장합니다.

    (
        static_X_train,
        static_y_train,
    ) = build_static_dataset(
        TRAIN_SPLIT
    )

    (
        static_X_val,
        static_y_val,
    ) = build_static_dataset(
        VAL_SPLIT
    )

    # ==============================================
    # ㉔ 실험 A: Static Random Forest 학습·평가
    # ==============================================
    # make_model(config)로 Static 전용 모델을 만듭니다.
    #
    # S01의 38-D Feature와 정답으로 학습합니다.
    #
    # S02의 38-D Feature를 입력해 Action을 예측합니다.
    #
    # 실제 정답과 비교하여 다음 점수를 계산합니다.
    #
    # static_acc → Accuracy
    # static_f1  → Macro F1
    #
    # S03 Test는 사용하지 않습니다.

    static_acc, static_f1 = (
        score_model(
            make_model(config),
            static_X_train,
            static_y_train,
            static_X_val,
            static_y_val,
        )
    )

    # ==============================================
    # ㉕ 실험 B: Sequence Train·Validation 데이터 읽기
    # ==============================================
    # 34번에서 만든 Sequence CSV를 사용합니다.
    #
    # S01의 Sequence Train 데이터를 읽습니다.
    #
    # S02의 Sequence Validation 데이터를 읽습니다.
    #
    # 기본적으로 Clip 하나는 1,140-D Feature입니다.
    #
    # Sequence Train      → (6,1140)
    # Sequence Validation → (6,1140)
    #
    # Static과 대상자는 같지만 Feature 표현이 다릅니다.

    (
        sequence_X_train,
        sequence_y_train,
    ) = sequence_xy(
        SEQUENCE_TRAIN
    )

    (
        sequence_X_val,
        sequence_y_val,
    ) = sequence_xy(
        SEQUENCE_VAL
    )

    # ==============================================
    # ㉖ 실험 B: Sequence Random Forest 학습·평가
    # ==============================================
    # Static 실험과 같은 설정으로 새 모델을 만듭니다.
    #
    # S01의 1,140-D Feature로 모델을 학습합니다.
    #
    # S02의 1,140-D Feature를 입력해 Action을 예측합니다.
    #
    # 다음 결과를 계산합니다.
    #
    # sequence_acc → Accuracy
    # sequence_f1  → Macro F1
    #
    # Static 모델을 재사용하는 것이 아니라
    # 새로운 Random Forest 모델을 따로 학습합니다.

    sequence_acc, sequence_f1 = (
        score_model(
            make_model(config),
            sequence_X_train,
            sequence_y_train,
            sequence_X_val,
            sequence_y_val,
        )
    )

    # ==============================================
    # ㉗ 두 실험의 결과를 하나의 DataFrame으로 정리
    # ==============================================
    # Static과 Sequence 결과를 두 행으로 기록합니다.
    #
    # representation:
    # Feature 표현 방법
    #
    # feature_count:
    # 모델에 입력한 Feature 개수
    #
    # accuracy:
    # S02 Validation 정답 비율
    #
    # macro_f1:
    # Action별 F1의 동일 가중 평균
    #
    # Static은 38개 Feature를 사용합니다.
    #
    # Sequence는 기본적으로 1,140개 Feature를 사용합니다.
    #
    # 주의:
    # 시간 정보와 Feature 수가 함께 달라지므로
    # 시간 정보만의 효과를 완전히 분리한 실험은 아닙니다.

    result = pd.DataFrame(
        [
            {
                "representation": (
                    "middle_frame_static"
                ),
                "feature_count": 38,
                "accuracy": (
                    static_acc
                ),
                "macro_f1": (
                    static_f1
                ),
            },
            {
                "representation": (
                    f"{int(config['sequence_steps'])}"
                    "_step_sequence"
                ),
                "feature_count": (
                    sequence_X_train
                    .shape[1]
                ),
                "accuracy": (
                    sequence_acc
                ),
                "macro_f1": (
                    sequence_f1
                ),
            },
        ]
    )

    # ==============================================
    # ㉘ Static과 Sequence 비교 결과 CSV 저장
    # ==============================================
    # 출력 폴더가 없다면 생성합니다.
    #
    # result.to_csv()로 두 실험 결과를 CSV에 저장합니다.
    #
    # index=False:
    # Pandas의 행 인덱스를 저장하지 않습니다.
    #
    # utf-8-sig:
    # Excel에서 한글을 읽기 쉬운 형식입니다.
    #
    # 기존 파일이 존재한다면 덮어씁니다.

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ==============================================
    # ㉙ 실험 결과표와 저장 경로 출력
    # ==============================================
    # result.to_string(index=False):
    # 두 가지 표현 방식의 결과를 화면에 출력합니다.
    #
    # Feature 개수, Accuracy, Macro F1을 비교합니다.
    #
    # 마지막으로 CSV 파일의 저장 위치를 출력합니다.
    #
    # Validation Clip이 6개뿐이므로
    # 결과를 일반화하지 않도록 주의합니다.

    print(
        result.to_string(
            index=False
        )
    )
    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


# ==================================================
# ㉚ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# Static과 Sequence 실험을 차례대로 실행하고
# 비교 결과 CSV를 생성합니다.

if __name__ == "__main__":
    main()