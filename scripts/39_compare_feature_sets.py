
from __future__ import annotations

# ==================================================
# ① 실험에 사용할 라이브러리 가져오기
# ==================================================
# json은 Baseline 설정 JSON을 읽습니다.
# Path는 파일과 폴더의 위치를 관리합니다.
# pandas는 Sequence CSV를 읽고 결과표를 저장합니다.
# RandomForestClassifier는 두 비교 실험에서 같은
# 분류 모델을 만듭니다.
# accuracy_score와 f1_score는 S02 Validation 성능을
# 계산합니다.

import json
from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ==================================================
# ② 프로젝트 ROOT와 Baseline 설정 파일 위치 지정
# ==================================================
# 현재 파일이 scripts 폴더 안에 있을 때
# parents[1]은 프로젝트 ROOT입니다.
#
# day06_baseline.json에서 Tree 수, 깊이,
# 난수 시드를 읽습니다.
#
# 두 실험에 동일한 모델 설정을 적용하여
# Feature 구성만 비교합니다.

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "day06_baseline.json"
)


# ==================================================
# ③ S01 Train Sequence CSV 경로 지정
# ==================================================
# 34번 코드가 만든 Train Sequence CSV를 사용합니다.
#
# 기본 수업에서는 S01의 6개 Clip이 들어 있습니다.
#
# 각 Clip은 30 Step × 38 Feature = 1,140개 숫자로
# 표현됩니다.

TRAIN_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "train.csv"
)


# ==================================================
# ④ S02 Validation Sequence CSV 경로 지정
# ==================================================
# 학습에는 사용하지 않은 S02의 6개 Clip을
# 검증에 사용합니다.
#
# 이번 비교에서는 S03 Test를 사용하지 않습니다.

VAL_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "val.csv"
)


# ==================================================
# ⑤ 두 실험의 비교 결과 CSV 저장 경로 지정
# ==================================================
# coordinates_plus_angles와 coordinates_only의
# 점수를 한 표에 기록합니다.
#
# 출력 파일:
# reports/day06/experiments/feature_set_comparison.csv
#
# 학습된 비교 모델은 저장하지 않고
# 성능 비교 CSV만 저장합니다.

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "experiments"
    / "feature_set_comparison.csv"
)


# ==================================================
# ⑥ 학습 입력에서 제외할 8개 메타데이터 정의
# ==================================================
# clip_id, subject, action_id, action_label 등
# 8개는 Clip 기본정보입니다.
#
# 특히 action_label은 모델이 맞힐 정답(y)이므로
# 입력 Feature(X)에서 제외합니다.
#
# 이외 컬럼을 Feature 후보로 사용합니다.

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
# ⑦ 제외할 네 가지 관절각도 이름 정의
# ==================================================
# 왼쪽·오른쪽 팔꿈치 각도와
# 왼쪽·오른쪽 무릎 각도 총 4종류입니다.
#
# 30 Step이라면 각도 Feature는
# 4 × 30 = 120개입니다.
#
# 이 120개를 제외하면
# 1,140 - 120 = 1,020개 좌표 Feature가 남습니다.

ANGLE_NAMES = {
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
}


# ==================================================
# ⑧ Baseline 설정 JSON을 읽는 함수
# ==================================================
# CONFIG_PATH 파일을 UTF-8로 엽니다.
#
# json.load()로 Python 딕셔너리를 반환합니다.
#
# 기본 설정 예:
# n_estimators = 300
# max_depth = None
# random_state = 42

def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ==================================================
# ⑨ 메타데이터를 제외한 전체 Feature 컬럼 선택
# ==================================================
# df.columns를 원래 순서대로 확인합니다.
#
# META_COLUMNS에 속하지 않는 컬럼만 골라
# 리스트로 반환합니다.
#
# 기본 30-Step Sequence CSV에서는
# 1,140개가 선택됩니다.
#
# 이 함수는 컬럼 개수나 숫자 타입을
# 별도로 검증하지 않습니다.

def feature_columns(
    df: pd.DataFrame,
):
    return [
        column
        for column in df.columns
        if column
        not in META_COLUMNS
    ]


# ==================================================
# ⑩ 좌표만 사용할 Feature 선택 함수 시작
# ==================================================
# coordinate_only_columns()는
# 전체 Feature 이름 목록을 입력받습니다.
#
# selected 빈 리스트에
# 관절각도가 아닌 컬럼 이름만 담습니다.

def coordinate_only_columns(
    columns,
):
    selected = []

    # ==============================================
    # ⑪ 컬럼 이름이 관절각도인지 판별
    # ==============================================
    # 모든 Feature 컬럼을 하나씩 검사합니다.
    #
    # endswith(f"_{angle}")은
    # t00_left_elbow_angle처럼
    # 끝이 각도 이름인지 확인합니다.
    #
    # any()는 네 각도 이름 중 하나라도
    # 일치하면 True를 반환합니다.
    #
    # t00_nose_x 같은 좌표 Feature는 False입니다.

    for column in columns:
        is_angle = any(
            column.endswith(
                f"_{angle}"
            )
            for angle in (
                ANGLE_NAMES
            )
        )

        # ==========================================
        # ⑫ 관절각도가 아닌 좌표 Feature만 남기기
        # ==========================================
        # is_angle이 False인 경우에만
        # selected에 추가합니다.
        #
        # 원래 컬럼의 순서를 유지한 채
        # 좌표 Feature 목록을 반환합니다.
        #
        # 기본 30 Step에서는
        # 34개 좌표 × 30 = 1,020개입니다.

        if not is_angle:
            selected.append(
                column
            )

    return selected


# ==================================================
# ⑬ 공통 평가 함수와 Train 입력 X 준비
# ==================================================
# evaluate()는 다음을 입력받습니다.
#
# name:
# 실험 이름
#
# train:
# S01 학습 DataFrame
#
# val:
# S02 검증 DataFrame
#
# columns:
# 이번 실험에서 사용할 Feature 컬럼
#
# config:
# Random Forest 설정
#
# train[columns]를 숫자 배열로 바꾸어
# X_train을 만듭니다.
#
# 전체 Feature면 기본 크기 (6, 1140)
# 좌표만이면 기본 크기 (6, 1020)입니다.

def evaluate(
    name,
    train,
    val,
    columns,
    config,
):
    X_train = (
        train[columns]
        .to_numpy(
            dtype=float
        )
    )

    # ==============================================
    # ⑭ Train 데이터의 정답 y_train 준비
    # ==============================================
    # Train의 action_label 컬럼을
    # 문자열 배열로 변환합니다.
    #
    # y_train은 모델이 맞혀야 하는
    # 6개 Action 정답입니다.

    y_train = (
        train[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    # ==============================================
    # ⑮ Validation 입력 X_val 준비
    # ==============================================
    # 학습 때 사용한 columns와 동일한 순서로
    # S02 Feature를 선택합니다.
    #
    # 숫자 배열로 변환하며
    # 현재 실험에 맞는 Feature 수를 사용합니다.

    X_val = (
        val[columns]
        .to_numpy(
            dtype=float
        )
    )

    # ==============================================
    # ⑯ Validation의 실제 정답 y_val 준비
    # ==============================================
    # S02의 action_label을 정답 배열로 가져옵니다.
    #
    # 정답은 모델에 미리 전달하지 않고
    # 예측 후 점수 계산에만 사용합니다.

    y_val = (
        val[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    # ==============================================
    # ⑰ 공통 Random Forest 모델 생성
    # ==============================================
    # n_estimators:
    # 사용할 Decision Tree 개수입니다.
    #
    # max_depth:
    # Tree 최대 깊이입니다.
    #
    # random_state:
    # 난수 시드입니다.
    #
    # class_weight="balanced":
    # Class 빈도에 따른 가중치를 설정합니다.
    #
    # n_jobs=-1:
    # 가능한 CPU 코어를 사용합니다.
    #
    # 각 evaluate() 호출마다
    # 서로 독립된 새 모델을 생성합니다.

    model = (
        RandomForestClassifier(
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
    )

    # ==============================================
    # ⑱ S01 Train 데이터로 실제 모델 학습
    # ==============================================
    # model.fit(X_train, y_train)이
    # 실제 학습 단계입니다.
    #
    # 두 실험 모두 S01로만 학습합니다.
    #
    # S02는 학습에 사용하지 않습니다.

    model.fit(
        X_train,
        y_train,
    )

    # ==============================================
    # ⑲ S02 예측 및 6개 Action Class 목록 준비
    # ==============================================
    # model.predict(X_val)은
    # S02의 Clip별 Action을 예측합니다.
    #
    # sorted(set(y_train))으로
    # 학습에 사용한 Class 이름을 정렬합니다.
    #
    # 이 Class 목록은
    # Macro F1을 계산할 때 사용됩니다.

    pred = model.predict(
        X_val
    )

    labels = sorted(
        set(y_train)
    )

    # ==============================================
    # ⑳ Accuracy와 Macro F1을 계산하여 결과 반환
    # ==============================================
    # 반환 딕셔너리에 다음을 기록합니다.
    #
    # experiment:
    # 실험 이름
    #
    # feature_count:
    # 입력 Feature 개수
    #
    # accuracy:
    # 전체 Validation 정답 비율
    #
    # macro_f1:
    # Action별 F1의 동일 비중 평균
    #
    # zero_division=0:
    # 분모가 0인 지표를 0으로 처리합니다.

    return {
        "experiment": name,
        "feature_count": (
            len(columns)
        ),
        "accuracy": (
            accuracy_score(
                y_val,
                pred,
            )
        ),
        "macro_f1": (
            f1_score(
                y_val,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


# ==================================================
# ㉑ main() 시작 및 공통 학습 설정 읽기
# ==================================================
# 모든 비교 실험을 실행하는 main() 함수입니다.
#
# load_config()로 두 실험이 공유할 설정을 불러옵니다.

def main() -> None:
    config = load_config()

    # ==============================================
    # ㉒ Train과 Validation Sequence CSV 읽기
    # ==============================================
    # 34번에서 만든 S01 train.csv와
    # S02 val.csv를 읽습니다.
    #
    # 같은 Clip 집합을 두 실험에서 사용해
    # 비교 조건을 맞춥니다.

    train = pd.read_csv(
        TRAIN_PATH
    )
    val = pd.read_csv(
        VAL_PATH
    )

    # ==============================================
    # ㉓ 전체 좌표와 관절각도를 포함한 Feature 목록 생성
    # ==============================================
    # feature_columns(train)으로
    # 8개 메타데이터를 제외합니다.
    #
    # 기본 구성:
    # 34개 좌표 + 4개 각도 = 38개/Step
    #
    # 30 Step이면 전체 Feature는 1,140개입니다.

    all_columns = (
        feature_columns(train)
    )

    # ==============================================
    # ㉔ 관절각도를 제외한 좌표 Feature 목록 생성
    # ==============================================
    # coordinate_only_columns(all_columns)을 호출합니다.
    #
    # 기본 30 Step에서는
    # 각도 120개를 빼고 좌표 1,020개가 남습니다.
    #
    # 두 표현 모두 동일한 30-Step Sequence를 사용합니다.

    coordinate_columns = (
        coordinate_only_columns(
            all_columns
        )
    )

    # ==============================================
    # ㉕ 두 실험을 같은 Random Forest 설정으로 평가
    # ==============================================
    # 실험 A:
    # coordinates_plus_angles
    # 전체 1,140개 Feature를 사용합니다.
    #
    # 실험 B:
    # coordinates_only
    # 각도를 뺀 1,020개 Feature를 사용합니다.
    #
    # 각 evaluate()에서 모델을 따로 학습하고
    # S02 Validation으로 평가합니다.
    #
    # S03 Test를 사용하지 않으며,
    # 모델을 저장하지도 않습니다.

    rows = [
        evaluate(
            "coordinates_plus_angles",
            train,
            val,
            all_columns,
            config,
        ),
        evaluate(
            "coordinates_only",
            train,
            val,
            coordinate_columns,
            config,
        ),
    ]

    # ==============================================
    # ㉖ 두 실험 결과를 Pandas 표로 변환
    # ==============================================
    # 각 실험 결과는 다음 컬럼을 가집니다.
    #
    # experiment
    # feature_count
    # accuracy
    # macro_f1
    #
    # 두 개의 딕셔너리를
    # DataFrame의 두 행으로 만듭니다.

    result = pd.DataFrame(
        rows
    )

    # ==============================================
    # ㉗ 비교 결과 저장 폴더 생성 및 CSV 저장
    # ==============================================
    # 결과 폴더가 없으면 부모 폴더까지 생성합니다.
    #
    # feature_set_comparison.csv로 결과표를 저장합니다.
    #
    # index=False:
    # 행 번호를 제외합니다.
    #
    # utf-8-sig:
    # Excel에서 한글을 읽기 쉬운 형식입니다.
    #
    # 기존 같은 이름의 출력 파일은 덮어씁니다.

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
    # ㉘ 비교 결과를 터미널에 출력
    # ==============================================
    # result.to_string(index=False)로
    # 두 실험을 표로 출력합니다.
    #
    # 마지막으로 저장한 CSV 경로를 출력합니다.
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
# ㉙ Python 파일을 직접 실행하면 main() 호출
# ==================================================
# python scripts/39_compare_feature_sets.py로 실행하면
# main()을 시작합니다.
#
# 두 실험을 학습·평가한 뒤
# 비교 결과 CSV를 저장합니다.

if __name__ == "__main__":
    main()