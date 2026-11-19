
from __future__ import annotations

# ==================================================
# ① Test 모델 평가에 필요한 라이브러리 가져오기
# ==================================================
# json은 35번 학습 때 저장한 Feature 컬럼 목록을 읽습니다.
# Path는 입력·출력 파일 경로를 관리합니다.
# joblib은 학습된 Random Forest 모델을 불러옵니다.
# pandas는 Test CSV를 읽고 예측 결과를 저장합니다.
# matplotlib과 sklearn.metrics는 성능 지표와
# Confusion Matrix를 생성합니다.

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ==================================================
# ② 프로젝트 ROOT와 S03 Test 입력 경로 지정
# ==================================================
# 현재 파일이 scripts 폴더에 있다고 가정합니다.
# parents[1]로 프로젝트 ROOT를 찾습니다.
#
# TEST_PATH:
# 34번 코드에서 만든 Test Sequence CSV입니다.
#
# 기본 수업에서는 S03의 6개 Action Clip이 들어 있습니다.
# S03은 학습과 Validation에 사용하지 않은 사람입니다.

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "test.csv"
)


# ==================================================
# ③ 학습된 Baseline 모델과 Feature 목록 경로 지정
# ==================================================
# MODEL_PATH:
# 35번 코드에서 학습해 저장한 Random Forest입니다.
#
# FEATURE_LIST_PATH:
# 모델 학습에 사용한 Feature 이름과 순서를 기록한 JSON입니다.
#
# Test 입력도 학습 당시와 같은 Feature 순서를 사용해야 합니다.
#
# 중요:
# 이 코드는 39번 좌표 전용 비교 실험 모델이 아니라
# 35번의 기본 Baseline 모델을 평가합니다.
#
# joblib 파일은 신뢰할 수 있는 출처의 것만 불러옵니다.

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
# ④ Test 평가 결과 세 파일의 저장 경로 지정
# ==================================================
# METRIC_PATH:
# Accuracy, Macro F1, Classification Report를 저장합니다.
#
# PREDICTION_PATH:
# Clip별 실제 정답, 모델 예측, Confidence를 저장합니다.
#
# CONFUSION_PATH:
# 실제와 예측 Action을 비교한 혼동행렬 PNG입니다.
#
# 이 코드는 모델을 다시 학습하거나 변경하지 않습니다.

METRIC_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "test_metrics.txt"
)

PREDICTION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "test_predictions.csv"
)

CONFUSION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "confusion"
    / "test_confusion_matrix.png"
)


# ==================================================
# ⑤ 학습 당시 Feature 이름과 순서를 읽는 함수
# ==================================================
# rf_feature_columns.json을 UTF-8로 엽니다.
#
# json.load():
# JSON 파일을 Python 리스트로 읽습니다.
#
# 35번 코드에서 저장한 Feature 순서를 그대로 반환합니다.
#
# 예:
# 30 Step × 38 Feature = 1,140개 Feature 이름

def load_feature_columns():
    with FEATURE_LIST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ==================================================
# ⑥ main() 시작: 학습된 Random Forest 모델 불러오기
# ==================================================
# main()은 Test 평가 전체를 수행하는 함수입니다.
#
# joblib.load():
# 35번에서 저장한 Random Forest 모델을 불러옵니다.
#
# model.fit()을 실행하지 않으므로 재학습하지 않습니다.
#
# Test는 모델과 Feature 선택을 마친 뒤
# 최종 성능을 확인하는 용도로 사용합니다.

def main() -> None:
    model = joblib.load(
        MODEL_PATH
    )

    # ==============================================
    # ⑦ S03 Test CSV와 학습 Feature 목록 읽기
    # ==============================================
    # Test Sequence CSV를 Pandas로 읽습니다.
    #
    # load_feature_columns()를 호출하여
    # 학습 당시의 Feature 컬럼 목록도 가져옵니다.
    #
    # 기본 수업:
    # S03의 6개 Clip
    # Clip당 1,140개 Sequence Feature

    test = pd.read_csv(
        TEST_PATH
    )

    columns = (
        load_feature_columns()
    )

    # ==============================================
    # ⑧ Test CSV에 필요한 Feature가 모두 있는지 확인
    # ==============================================
    # columns:
    # 학습에 사용한 Feature 이름입니다.
    #
    # test.columns:
    # Test CSV에 실제로 있는 컬럼 이름입니다.
    #
    # 학습 Feature 중 Test에 없는 컬럼을 찾습니다.
    #
    # 하나라도 누락되면 RuntimeError를 발생시킵니다.
    #
    # 주의:
    # 현재 코드는 컬럼 존재 여부를 검사하며
    # NaN, 무한대, Test Clip 수 등은 별도로 검사하지 않습니다.

    missing = [
        column
        for column in columns
        if column not in test.columns
    ]

    if missing:
        raise RuntimeError(
            "Test에 학습 Feature가 "
            "누락되었습니다."
        )

    # ==============================================
    # ⑨ Test 입력 X_test를 숫자 배열로 생성
    # ==============================================
    # test[columns]:
    # 모델이 학습한 Feature를 같은 순서로 선택합니다.
    #
    # to_numpy(dtype=float):
    # 데이터를 숫자 배열로 변환합니다.
    #
    # 기본 결과:
    # X_test.shape = (6, 1140)
    #
    # 6개 Clip × 1,140개 Sequence Feature
    #
    # 실제 정답과 Clip 기본정보는 X_test에 포함하지 않습니다.

    X_test = (
        test[columns]
        .to_numpy(
            dtype=float
        )
    )

    # ==============================================
    # ⑩ Test의 실제 Action 정답 y_true 생성
    # ==============================================
    # action_label은 각 Clip의 실제 동작 이름입니다.
    #
    # 문자열로 변환한 뒤 NumPy 배열로 만듭니다.
    #
    # y_true는 예측 결과와 비교하는 정답입니다.
    #
    # model.predict()에 정답을 입력하지 않습니다.

    y_true = (
        test[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    # ==============================================
    # ⑪ 학습된 모델로 Test Action 예측
    # ==============================================
    # model.predict(X_test):
    # S03의 각 Clip이 어떤 Action인지 예측합니다.
    #
    # y_pred:
    # 모델이 예측한 Action Label 배열입니다.
    #
    # 아직 성능 점수를 계산한 것은 아닙니다.
    # 실제 정답과 비교하는 과정이 필요합니다.

    y_pred = model.predict(
        X_test
    )

    # ==============================================
    # ⑫ Clip별 Action 예측확률 계산
    # ==============================================
    # predict_proba():
    # 각 Clip의 Action Class별 예측확률을 반환합니다.
    #
    # 기본 수업:
    # 6개 Clip × 6개 Class = (6, 6)
    #
    # 반환 배열의 Class 순서는 model.classes_와 같습니다.

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    # ==============================================
    # ⑬ 가장 높은 예측확률과 Class 목록 가져오기
    # ==============================================
    # probabilities.max(axis=1):
    # Clip마다 가장 높은 예측확률을 가져옵니다.
    #
    # 이 값을 confidence로 기록합니다.
    #
    # model.classes_:
    # 모델이 학습한 Action Class 목록입니다.
    #
    # 주의:
    # Confidence는 모델의 예측확률값이며
    # 실제로 정답일 확률이 보정되어 있다는 뜻은 아닙니다.

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    labels = list(
        model.classes_
    )

    # ==============================================
    # ⑭ Test Accuracy 계산
    # ==============================================
    # accuracy_score():
    # 실제 정답 y_true와 예측 y_pred를 비교합니다.
    #
    # Accuracy = 맞힌 Clip 수 / 전체 Clip 수
    #
    # 예:
    # 6개 중 4개를 맞히면 4/6 = 약 0.6667입니다.
    #
    # Test 데이터가 6개뿐이라
    # Clip 하나의 결과가 점수에 큰 영향을 줍니다.

    accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    # ==============================================
    # ⑮ Test Macro F1 계산
    # ==============================================
    # f1_score():
    # Precision과 Recall을 반영한 F1을 계산합니다.
    #
    # average="macro":
    # 각 Action Class의 F1을 같은 비중으로 평균합니다.
    #
    # labels=labels:
    # 모델이 학습한 Class 전체를 기준으로 계산합니다.
    #
    # zero_division=0:
    # 계산이 불가능한 지표를 0으로 처리합니다.

    macro_f1 = (
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    # ==============================================
    # ⑯ Classification Report 작성
    # ==============================================
    # classification_report():
    # 각 Action별 세부 성능을 계산합니다.
    #
    # Precision : 해당 Action으로 예측한 것의 정확성
    # Recall    : 실제 해당 Action을 찾아낸 비율
    # F1-score  : Precision과 Recall의 조화평균
    # Support   : 실제 정답에 포함된 데이터 개수
    #
    # 결과를 문자열 형태의 report로 저장합니다.

    report = (
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    # ==============================================
    # ⑰ Test 성능 결과를 터미널에 출력
    # ==============================================
    # Accuracy와 Macro F1을 소수점 넷째 자리로 출력합니다.
    #
    # 이어서 Classification Report를 출력합니다.
    #
    # 실제 점수는 Test CSV와 저장된 모델에 따라 달라집니다.

    print(
        f"Test Accuracy: "
        f"{accuracy:.4f}"
    )
    print(
        f"Test Macro F1: "
        f"{macro_f1:.4f}"
    )
    print()
    print(report)

    # ==============================================
    # ⑱ Test 전체 성능을 텍스트 파일로 저장
    # ==============================================
    # reports/day06/metrics 폴더를 생성합니다.
    #
    # test_metrics.txt에 다음 결과를 기록합니다.
    #
    # Test Accuracy
    # Test Macro F1
    # Classification Report
    #
    # 파일에서는 점수를 소수점 여섯 자리로 기록합니다.
    #
    # write_text()는 기존 동일 이름의 파일을 덮어씁니다.

    METRIC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_PATH.write_text(
        (
            f"Test Accuracy: "
            f"{accuracy:.6f}\n"
            f"Test Macro F1: "
            f"{macro_f1:.6f}\n\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    # ==============================================
    # ⑲ Clip별 Test 예측 결과표 생성
    # ==============================================
    # 각 Clip의 예측 결과를 하나의 행으로 만듭니다.
    #
    # clip_id:
    # 개별 영상 식별자
    #
    # subject:
    # 촬영한 사람, 기본 수업에서는 S03
    #
    # true_label:
    # 실제 Action
    #
    # pred_label:
    # 모델이 예측한 Action
    #
    # confidence:
    # 가장 높은 예측확률
    #
    # correct:
    # 정답과 예측이 같으면 True, 다르면 False입니다.

    prediction_df = (
        pd.DataFrame(
            {
                "clip_id": (
                    test["clip_id"]
                ),
                "subject": (
                    test["subject"]
                ),
                "true_label": (
                    y_true
                ),
                "pred_label": (
                    y_pred
                ),
                "confidence": (
                    confidence
                ),
                "correct": (
                    y_true == y_pred
                ),
            }
        )
    )

    # ==============================================
    # ⑳ Clip별 예측 결과 CSV 저장
    # ==============================================
    # test_predictions.csv에 결과를 저장합니다.
    #
    # index=False:
    # Pandas 행 인덱스는 저장하지 않습니다.
    #
    # utf-8-sig:
    # Excel에서 한글을 읽기 쉬운 형식입니다.
    #
    # Clip별 정답·오분류 결과를 확인할 수 있습니다.

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ==============================================
    # ㉑ Confusion Matrix 숫자 행렬 생성
    # ==============================================
    # confusion_matrix():
    # 실제 Action과 예측 Action의 관계를 계산합니다.
    #
    # 행: 실제 Action
    # 열: 예측 Action
    #
    # 대각선 값:
    # 정답으로 예측한 Clip 개수
    #
    # 대각선 밖의 값:
    # 다른 Action으로 잘못 분류한 Clip 개수
    #
    # labels로 Action 순서를 지정합니다.

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    # ==============================================
    # ㉒ Confusion Matrix에 Action 이름 연결
    # ==============================================
    # ConfusionMatrixDisplay():
    # 숫자 행렬을 시각화할 수 있도록 준비합니다.
    #
    # display_labels=labels:
    # x축과 y축에 Action 이름을 표시합니다.

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels,
        )
    )

    # ==============================================
    # ㉓ Confusion Matrix 그래프 그리기
    # ==============================================
    # display.plot():
    # 실제 혼동행렬을 그림으로 만듭니다.
    #
    # xticks_rotation=45:
    # 긴 Action 이름을 보기 쉽게 45도 회전합니다.
    #
    # plt.title():
    # 그림 제목을 설정합니다.
    #
    # plt.tight_layout():
    # 글자가 겹치지 않도록 여백을 조정합니다.

    display.plot(
        xticks_rotation=45,
    )

    plt.title(
        "Test Confusion Matrix"
    )
    plt.tight_layout()

    # ==============================================
    # ㉔ Confusion Matrix PNG 저장 및 그림 닫기
    # ==============================================
    # reports/day06/confusion 폴더를 생성합니다.
    #
    # plt.savefig():
    # test_confusion_matrix.png로 저장합니다.
    #
    # dpi=150:
    # 이미지 해상도를 지정합니다.
    #
    # plt.close():
    # 그림 객체를 닫아 자원을 정리합니다.

    CONFUSION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CONFUSION_PATH,
        dpi=150,
    )
    plt.close()

    # ==============================================
    # ㉕ 생성한 Test 평가 결과 파일 경로 출력
    # ==============================================
    # 전체 성능 보고서 경로를 출력합니다.
    #
    # Clip별 예측 결과 CSV 경로를 출력합니다.
    #
    # Confusion Matrix PNG 경로를 출력합니다.
    #
    # 학생은 세 파일을 이용하여
    # 최종 Test 결과를 확인할 수 있습니다.

    print(
        f"Metrics: "
        f"{METRIC_PATH}"
    )
    print(
        f"Predictions: "
        f"{PREDICTION_PATH}"
    )
    print(
        f"Confusion Matrix: "
        f"{CONFUSION_PATH}"
    )


# ==================================================
# ㉖ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# 예:
# python scripts/40_evaluate_test.py
#
# 다른 Python 파일에서 import할 때는 자동 실행되지 않습니다.

if __name__ == "__main__":
    main()