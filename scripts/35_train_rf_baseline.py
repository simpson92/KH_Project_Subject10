
from __future__ import annotations

# ==================================================
# ① 모델 학습과 파일 저장에 필요한 라이브러리 가져오기
# ==================================================
# json   : 6일차 설정과 학습 정보를 JSON으로 읽고 저장합니다.
# Path   : 프로젝트 안의 파일과 폴더 경로를 관리합니다.
# joblib : 학습된 Random Forest 모델을 파일로 저장합니다.
# pandas : 34번 코드에서 생성한 Train Sequence CSV를 읽습니다.
# RandomForestClassifier:
#          여러 Decision Tree로 Action을 분류하는 모델입니다.

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)


# ==================================================
# ② 프로젝트 ROOT와 Baseline 설정 파일 경로 지정
# ==================================================
# 현재 파일을 scripts 폴더에 저장했다고 가정합니다.
# parents[1]은 scripts의 상위인 프로젝트 ROOT입니다.
# day06_baseline.json에는 Sequence Step 수, Tree 수,
# 난수 시드 등 학습에 사용할 설정이 들어 있습니다.

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "day06_baseline.json"
)

# ==================================================
# ③ Train Sequence CSV 입력 파일 경로 지정
# ==================================================
# 34번 코드에서 만든 train.csv를 사용합니다.
# 이번 기본 수업에서는 S01의 6개 Action Clip이 들어 있습니다.
# 각 Clip은 30 Step × 38 Feature = 1,140개 숫자로
# 표현되며 CSV에서 Clip 하나가 한 행입니다.
# Validation(S02)과 Test(S03)는 여기서 사용하지 않습니다.

TRAIN_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "train.csv"
)

# ==================================================
# ④ 학습 모델·Feature 순서·학습 정보의 저장 경로 지정
# ==================================================
# MODEL_PATH:
#   학습이 끝난 Random Forest 모델 파일입니다.
# FEATURE_LIST_PATH:
#   모델에 입력한 Feature 이름과 순서를 기록합니다.
# TRAIN_INFO_PATH:
#   사용한 Subject, Class, 학습 설정 등을 기록합니다.
# 모델을 나중에 불러올 때 Feature 순서도 동일해야 합니다.

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

# ==================================================
# ⑤ 모델 입력에서 제외할 Clip 기본정보 컬럼 정의
# ==================================================
# 이 8개 컬럼은 Clip을 구분하거나 데이터 상태를 기록합니다.
# 예: action_label은 모델이 맞혀야 하는 정답(y)입니다.
# 이 컬럼들을 Feature(X)에 넣으면 안 됩니다.
# 나머지 컬럼을 모델의 숫자 Feature로 선택합니다.

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
# ⑥ Baseline 학습 설정 JSON을 읽는 함수
# ==================================================
# JSON 파일을 열어 Python 딕셔너리로 변환합니다.
# 예: sequence_steps=30, n_estimators=300,
# max_depth=null, random_state=42 등입니다.
# 실제 설정값은 day06_baseline.json 내용을 따릅니다.

def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ==================================================
# ⑦ Sequence Feature 컬럼 이름만 선택하는 함수
# ==================================================
# df.columns에서 ⑤의 META_COLUMNS에 없는 이름만
# 가져와 리스트로 반환합니다.
# 34번 코드의 기본 결과라면 1,140개가 선택됩니다.
# 컬럼 순서를 그대로 보존하므로 모델 저장 시에도
# 동일한 순서를 따로 기록해야 합니다.
# 주의: 예상하지 못한 추가 컬럼도 포함될 수 있으므로
# 뒤에서 Feature 개수를 검사합니다.

def feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in META_COLUMNS
    ]


# ==================================================
# ⑧ main() 시작: Train CSV가 존재하는지 확인
# ==================================================
# 모델 학습은 main()에서 수행합니다.
# 34번 코드의 결과 파일이 없다면 학습할 수 없으므로
# FileNotFoundError를 발생시킵니다.

def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    # ==============================================
    # ⑨ 설정 파일과 Train Sequence CSV 읽기
    # ==============================================
    # ⑥의 load_config()로 Baseline 설정을 읽습니다.
    # pandas로 train.csv를 읽어 DataFrame으로 만듭니다.
    # 이 파일은 S01의 6개 Clip을 담은 학습용 데이터입니다.

    config = load_config()

    train = pd.read_csv(
        TRAIN_PATH
    )

    # ==============================================
    # ⑩ Train에 정확히 6개 Clip이 있는지 확인
    # ==============================================
    # len(train)은 CSV의 데이터 행 수를 뜻합니다.
    # 이번 기본 수업은 S01의 6개 Action, 총 6개 Clip로
    # 학습하므로 행 수가 6이 아니면 중단합니다.
    # 주의: 행 수만 확인하며 Subject가 S01인지 또는
    # Clip ID가 중복되었는지 여기서 다시 검사하지 않습니다.

    if len(train) != 6:
        raise RuntimeError(
            "현재 Baseline Train은 "
            "6개 Clip이어야 합니다."
        )

    # ==============================================
    # ⑪ Train에 6종류의 Action Class가 있는지 확인
    # ==============================================
    # action_label은 모델이 맞힐 정답 이름입니다.
    # unique()는 중복된 Action 이름을 제거합니다.
    # sorted()로 이름을 일정한 순서로 정렬합니다.
    # 서로 다른 Class가 6개가 아니면 오류를 발생시킵니다.
    # 단, 어떤 특정 Class가 있는지까지 검사하지는 않습니다.

    classes = sorted(
        train[
            "action_label"
        ]
        .astype(str)
        .unique()
    )

    if len(classes) != 6:
        raise RuntimeError(
            "Train에 6개 Action Class가 "
            "모두 필요합니다."
        )

    # ==============================================
    # ⑫ 기본정보를 제외한 Feature 컬럼 목록 생성
    # ==============================================
    # ⑦의 feature_columns()를 호출합니다.
    # Clip ID, Subject, 정답 Label 등을 제외하고
    # t00_nose_x, ..., t29_right_knee_angle과 같은
    # Sequence Feature 이름만 가져옵니다.

    columns = feature_columns(
        train
    )

    # ==============================================
    # ⑬ 예상 Feature 수를 계산하고 실제 개수와 비교
    # ==============================================
    # 각 시간 Step에는 38-D Frame Feature가 있습니다.
    # 예상 Feature 수 = sequence_steps × 38
    # 예: 30 × 38 = 1,140개입니다.
    # 실제 선택한 컬럼 수와 다르면 학습을 중단합니다.

    expected_features = (
        int(
            config[
                "sequence_steps"
            ]
        )
        * 38
    )

    if len(columns) != (
        expected_features
    ):
        raise RuntimeError(
            "Sequence Feature 수가 "
            f"{expected_features}개가 "
            f"아닙니다: {len(columns)}"
        )

    # ==============================================
    # ⑭ 학습 입력 X_train을 숫자 배열로 만들기
    # ==============================================
    # X_train은 모델이 보고 학습하는 숫자 데이터입니다.
    # train[columns]에서 1,140개 Feature만 선택합니다.
    # to_numpy(dtype=float)로 NumPy 숫자 배열을 만듭니다.
    # 기본 수업의 입력 형태: (6, 1140)
    # = 6개 Clip 각각에 1,140개 Sequence Feature

    X_train = (
        train[columns]
        .to_numpy(
            dtype=float
        )
    )

    # ==============================================
    # ⑮ 정답 y_train을 Action Label 배열로 만들기
    # ==============================================
    # y_train은 모델이 맞혀야 하는 정답입니다.
    # 각 Clip의 action_label을 읽어 문자열 배열로 만듭니다.
    # 예: bend_return, ..., wave
    # 기본 수업의 정답 배열 길이는 6입니다.
    # X_train의 각 행과 y_train의 각 값은 같은 Clip입니다.

    y_train = (
        train[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    # ==============================================
    # ⑯ 설정값으로 Random Forest 분류 모델 생성
    # ==============================================
    # n_estimators: 사용할 Decision Tree 개수입니다.
    # max_depth   : 각 Tree의 최대 깊이입니다.
    # random_state: 반복 실행 시 난수를 고정합니다.
    # class_weight="balanced": Class 빈도를 고려한 가중치입니다.
    # n_jobs=-1: 가능한 CPU 코어를 병렬 작업에 사용합니다.
    # 이 단계는 모델의 구조·설정만 준비한 것이며
    # 아직 실제 학습이 완료된 상태는 아닙니다.

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
            class_weight=(
                "balanced"
            ),
            n_jobs=-1,
        )
    )

    # ==============================================
    # ⑰ Train 데이터만 사용하여 모델 학습 실행
    # ==============================================
    # fit(X_train, y_train)이 실제 학습 명령입니다.
    # S01의 6개 Sequence와 정답 Action을 연결해
    # Random Forest가 분류 규칙을 학습합니다.
    # 이 단계에 S02 Validation과 S03 Test는 사용하지 않습니다.

    model.fit(
        X_train,
        y_train,
    )

    # ==============================================
    # ⑱ 학습한 모델을 joblib 파일로 저장
    # ==============================================
    # artifacts/day06 폴더가 없으면 생성합니다.
    # joblib.dump()로 학습된 모델을 저장합니다.
    # 저장된 모델은 다음 Validation 단계에서 불러옵니다.
    # joblib 파일은 신뢰할 수 있는 출처의 것만 불러와야 합니다.

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # ==============================================
    # ⑲ 모델의 Feature 이름과 순서를 JSON으로 저장
    # ==============================================
    # 같은 모델을 추론에 사용할 때도 학습 당시의
    # Feature 이름과 순서를 정확히 유지해야 합니다.
    # 1,140개 컬럼 이름을 JSON 배열로 저장합니다.
    # ensure_ascii=False는 한글 등의 문자를 보존합니다.
    # indent=2는 사람이 읽기 좋게 들여쓰기합니다.

    with FEATURE_LIST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            columns,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ==============================================
    # ⑳ 학습 데이터와 Baseline 설정을 딕셔너리로 정리
    # ==============================================
    # 나중에 어떤 데이터·설정으로 학습했는지 확인하려고
    # Clip 수, Subject, Action Class, Feature 수와
    # Sequence 길이 및 Random Forest 설정을 기록합니다.
    # 이 정보는 실제 모델 파라미터가 아니라 학습 기록입니다.

    training_info = {
        "train_clips": len(train),
        "subjects": sorted(
            train[
                "subject"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
        "classes": classes,
        "feature_count": (
            len(columns)
        ),
        "sequence_steps": int(
            config[
                "sequence_steps"
            ]
        ),
        "min_valid_frames": int(
            config[
                "min_valid_frames"
            ]
        ),
        "n_estimators": int(
            config[
                "n_estimators"
            ]
        ),
        "max_depth": (
            config[
                "max_depth"
            ]
        ),
        "random_state": int(
            config[
                "random_state"
            ]
        ),
    }

    # ==============================================
    # ㉑ 학습 정보를 JSON 파일로 저장
    # ==============================================
    # training_info 딕셔너리를 사람이 읽을 수 있는
    # rf_training_info.json으로 저장합니다.
    # ⑱에서 artifacts/day06 폴더를 이미 생성했습니다.
    # 모델 파일, Feature 목록, 학습 기록을 함께 보관합니다.

    with TRAIN_INFO_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ==============================================
    # ㉒ 학습 결과와 저장 경로 출력
    # ==============================================
    # 학습에 사용한 Clip 수와 Feature 개수,
    # 6개 Action Class를 터미널에서 확인합니다.
    # Training completed.는 fit()과 파일 저장이
    # 오류 없이 끝났다는 의미입니다.
    # 모델의 Validation 성능이 좋다는 의미는 아닙니다.
    # 학습 정보 JSON은 저장하지만 여기서 경로를
    # 별도로 출력하지는 않습니다(원본 코드 그대로).

    print(
        f"Train clips   : "
        f"{len(train)}"
    )
    print(
        f"Feature count : "
        f"{len(columns)}"
    )
    print(
        f"Classes       : "
        f"{classes}"
    )
    print()
    print(
        "Training completed."
    )
    print(
        f"Model: {MODEL_PATH}"
    )
    print(
        f"Features: "
        f"{FEATURE_LIST_PATH}"
    )


# ==================================================
# ㉓ 파일을 직접 실행했을 때 main() 호출
# ==================================================
# python scripts/35_train_rf_baseline.py로 실행하면
# main()이 실행되어 학습·저장까지 진행합니다.
# 다른 Python 파일에서 import할 때는 자동 실행되지 않습니다.

if __name__ == "__main__":
    main()