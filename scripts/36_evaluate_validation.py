
from __future__ import annotations

# ==================================================
# ① 모델 평가에 필요한 라이브러리 가져오기
# ==================================================
# json: 학습할 때 저장한 Feature 이름과 순서를 읽습니다.
# Path: 입력·출력 파일 경로를 관리합니다.
# joblib: 35번 코드에서 저장한 Random Forest 모델을 불러옵니다.
# matplotlib: Confusion Matrix 그림을 저장합니다.
# pandas: Validation CSV와 예측 결과 CSV를 처리합니다.
# sklearn.metrics: Accuracy, Macro F1, Class별 성능과 혼동행렬을 계산합니다.

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
# ② 프로젝트 ROOT와 S02 Validation 입력 경로 지정
# ==================================================
# scripts 폴더의 상위인 프로젝트 ROOT를 찾습니다.
# VAL_PATH는 34번 코드가 만든 6일차 val.csv 경로입니다.
# 기본 수업에서는 S02의 6개 Action Clip이 저장되어 있습니다.
# S01은 Train, S02는 Validation, S03은 Test입니다.

ROOT = Path(__file__).resolve().parents[1]

VAL_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "val.csv"
)


# ==================================================
# ③ 학습된 모델과 Feature 목록 파일 경로 지정
# ==================================================
# MODEL_PATH: 35번에서 학습해 저장한 Random Forest입니다.
# FEATURE_LIST_PATH: 학습할 때 사용한 1,140개 Feature 이름과 순서입니다.
# 새로운 데이터도 학습 때와 동일한 Feature 순서로 전달해야 합니다.
# 주의: joblib 파일은 신뢰할 수 있는 출처의 것만 불러옵니다.

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
# ④ Validation 평가 결과를 저장할 세 가지 경로 설정
# ==================================================
# METRIC_PATH: Accuracy, Macro F1, Classification Report 텍스트입니다.
# PREDICTION_PATH: Clip별 정답·예측·Confidence·정오답 CSV입니다.
# CONFUSION_PATH: 실제와 예측 Action을 비교한 혼동행렬 그림입니다.
# 이 코드는 Test 파일을 읽거나 모델을 다시 학습하지 않습니다.

METRIC_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "validation_metrics.txt"
)

PREDICTION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "validation_predictions.csv"
)

CONFUSION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "confusion"
    / "validation_confusion_matrix.png"
)


# ==================================================
# ⑤ 학습 당시 Feature 이름과 순서를 JSON에서 읽기
# ==================================================
# rf_feature_columns.json을 UTF-8로 열고 json.load()로 읽습니다.
# JSON 배열의 순서를 그대로 보존하여 반환합니다.
# 모델 입력 X_val의 컬럼 순서를 맞추기 위해 사용합니다.

def load_feature_columns():
    with FEATURE_LIST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ==================================================
# ⑥ main 시작: 저장된 Random Forest 모델 불러오기
# ==================================================
# main()은 Validation 평가 전체를 수행합니다.
# joblib.load()로 학습된 모델을 메모리에 복원합니다.
# model.fit()을 호출하지 않으므로 추가 학습은 하지 않습니다.

def main() -> None:
    model = joblib.load(
        MODEL_PATH
    )

    # ==============================================
    # ⑦ S02 Validation CSV와 학습 Feature 목록 읽기
    # ==============================================
    # val.csv를 Pandas DataFrame으로 읽습니다.
    # load_feature_columns()에서 학습 당시의 Feature 순서를 가져옵니다.
    # 이번 기본 데이터는 S02의 6개 Clip과 1,140-D Sequence Feature입니다.

    val = pd.read_csv(
        VAL_PATH
    )

    columns = (
        load_feature_columns()
    )

    # ==============================================
    # ⑧ Validation에 필요한 Feature 누락 여부 확인
    # ==============================================
    # 학습 시 사용한 columns 중 val.csv에 없는 컬럼을 찾습니다.
    # 누락된 컬럼이 하나라도 있으면 RuntimeError를 발생시킵니다.
    # 이 검사를 통과해야 모델에 같은 이름·순서의 Feature를 줄 수 있습니다.
    # 주의: 이 코드는 Feature 값의 NaN/Inf나 데이터 타입을 별도 검증하지는 않습니다.

    missing = [
        column
        for column in columns
        if column not in val.columns
    ]

    if missing:
        raise RuntimeError(
            "Validation에 학습 Feature가 "
            "누락되었습니다."
        )

    # ==============================================
    # ⑨ Validation 입력 X_val을 숫자 배열로 만들기
    # ==============================================
    # val[columns]로 학습 당시와 같은 Feature를 같은 순서로 선택합니다.
    # to_numpy(dtype=float)로 모델이 받을 숫자 배열을 만듭니다.
    # 기본 데이터의 형태: (6, 1140) = 6개 Clip × 1,140개 Feature.
    # clip_id·subject·action_label 같은 메타데이터는 X_val에서 제외됩니다.

    X_val = (
        val[columns]
        .to_numpy(
            dtype=float
        )
    )

    # ==============================================
    # ⑩ Validation의 실제 정답 y_true 가져오기
    # ==============================================
    # 각 Clip의 action_label을 실제 정답으로 사용합니다.
    # y_true는 성능을 계산할 때 예측값과 비교하는 용도입니다.
    # 예측 시 model.predict()에 y_true를 전달하지 않습니다.

    y_true = (
        val[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    # ==============================================
    # ⑪ 학습된 모델로 Action 예측하기
    # ==============================================
    # model.predict(X_val)로 S02의 각 Clip에 대한 Action을 예측합니다.
    # 반환한 y_pred는 Clip 순서대로 예측한 Action 이름 배열입니다.
    # 아직 Accuracy를 계산한 것은 아니며, 정답과의 비교가 필요합니다.

    y_pred = model.predict(
        X_val
    )

    # ==============================================
    # ⑫ 여섯 Action에 대한 모델 예측확률 계산
    # ==============================================
    # predict_proba()는 각 Clip의 Action Class별 예측확률을 반환합니다.
    # 열의 순서는 model.classes_와 일치합니다.
    # 기본 데이터에서 배열 형태는 (6, 6)입니다.
    # Random Forest의 확률은 실제 정답 확률로 보정되었다고 볼 수 없습니다.

    probabilities = (
        model.predict_proba(
            X_val
        )
    )

    # ==============================================
    # ⑬ 선택된 예측의 Confidence와 Class 목록 가져오기
    # ==============================================
    # probabilities.max(axis=1)로 Clip마다 가장 큰 예측확률을 뽑습니다.
    # 이것을 해당 Clip의 confidence로 기록합니다.
    # model.classes_에서 모델이 학습한 Class 목록을 가져옵니다.
    # 이 labels는 성능 지표와 혼동행렬의 표시 순서에 사용됩니다.

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    labels = list(
        model.classes_
    )

    # ==============================================
    # ⑭ Accuracy: 전체 Clip 중 맞힌 비율 계산
    # ==============================================
    # accuracy_score()는 정답 y_true와 예측 y_pred를 비교합니다.
    # 예: 6개 중 4개를 맞히면 4/6 = 약 0.6667입니다.
    # Validation이 6개뿐이라 Clip 1개 차이가 점수에 크게 반영됩니다.

    accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    # ==============================================
    # ⑮ Macro F1: Action별 F1의 평균 계산
    # ==============================================
    # f1_score(average="macro")는 Class별 F1을 같은 비중으로 평균합니다.
    # labels=labels로 모델의 6개 Action을 기준으로 계산합니다.
    # zero_division=0은 분모가 0인 경우 해당 값이 0이 되도록 합니다.
    # Accuracy와 함께 작은 데이터의 성능을 해석하는 참고 지표입니다.

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
    # ⑯ Classification Report: Action별 상세 성능 작성
    # ==============================================
    # classification_report()로 Precision·Recall·F1·Support를 작성합니다.
    # labels=labels로 모델의 Class 순서를 사용합니다.
    # 이 결과는 사람이 읽을 수 있는 문자열로 반환됩니다.
    # 어떤 Action을 특히 잘 맞히거나 놓치는지 확인합니다.

    report = (
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    # ==============================================
    # ⑰ Accuracy·Macro F1·Classification Report 출력
    # ==============================================
    # 소수점 넷째 자리까지 Accuracy와 Macro F1을 출력합니다.
    # 이어서 Class별 Classification Report를 표시합니다.
    # 여기 표시되는 점수는 실제 S02 CSV와 모델에 따라 달라집니다.

    print(
        f"Validation Accuracy: "
        f"{accuracy:.4f}"
    )
    print(
        f"Validation Macro F1: "
        f"{macro_f1:.4f}"
    )
    print()
    print(report)

    # ==============================================
    # ⑱ 전체 Validation 성능을 텍스트 파일로 저장
    # ==============================================
    # reports/day06/metrics 폴더가 없다면 생성합니다.
    # validation_metrics.txt에 Accuracy·Macro F1과 Report를 저장합니다.
    # 파일에 기록할 점수는 소수점 여섯 자리로 작성합니다.
    # write_text()는 기존 동일 이름의 파일을 덮어씁니다.

    METRIC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_PATH.write_text(
        (
            f"Validation Accuracy: "
            f"{accuracy:.6f}\n"
            f"Validation Macro F1: "
            f"{macro_f1:.6f}\n\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    # ==============================================
    # ⑲ Clip별 정답·예측·Confidence·정오답 표 생성
    # ==============================================
    # validation_predictions.csv에 넣을 DataFrame을 만듭니다.
    # clip_id: 영상 식별자 / subject: 사람(S02)
    # true_label: 실제 Action / pred_label: 모델의 예측 Action
    # confidence: 예측한 Class의 가장 높은 확률값
    # correct: 실제 정답과 예측이 같으면 True, 다르면 False입니다.

    prediction_df = (
        pd.DataFrame(
            {
                "clip_id": (
                    val["clip_id"]
                ),
                "subject": (
                    val["subject"]
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
    # ⑳ Clip별 예측 결과를 CSV로 저장
    # ==============================================
    # validation_predictions.csv로 예측 결과를 저장합니다.
    # index=False로 Pandas 인덱스는 저장하지 않습니다.
    # utf-8-sig는 Excel에서 한글을 읽기 쉽게 하는 인코딩입니다.
    # 저장한 CSV는 37번 오분류 분석 실습의 입력으로 사용됩니다.

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ==============================================
    # ㉑ Confusion Matrix에 사용할 숫자 행렬 생성
    # ==============================================
    # confusion_matrix()는 실제 정답과 모델의 예측을 비교합니다.
    # 행은 실제 Action, 열은 예측 Action입니다.
    # 대각선에 있는 수는 정답, 대각선 밖은 오분류입니다.
    # labels=labels로 두 축에 같은 Class 순서를 적용합니다.

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    # ==============================================
    # ㉒ Confusion Matrix를 사람이 읽을 수 있는 그림으로 표현
    # ==============================================
    # ConfusionMatrixDisplay로 숫자 행렬과 Class 이름을 연결합니다.
    # plot(xticks_rotation=45)로 x축 Action 이름을 45도 회전합니다.
    # 제목을 넣고 tight_layout()으로 글자 겹침을 줄입니다.
    # 이 단계는 그림을 준비하며 아직 PNG로 저장하지는 않습니다.

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels,
        )
    )

    display.plot(
        xticks_rotation=45,
    )

    plt.title(
        "Validation Confusion Matrix"
    )
    plt.tight_layout()

    # ==============================================
    # ㉓ Confusion Matrix PNG 저장 및 그림 닫기
    # ==============================================
    # reports/day06/confusion 폴더가 없다면 생성합니다.
    # savefig(dpi=150)로 혼동행렬 PNG를 저장합니다.
    # plt.close()로 사용한 그림을 닫아 자원을 정리합니다.

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
    # ㉔ 평가 결과 세 파일의 저장 경로 출력
    # ==============================================
    # 텍스트 성능 보고서, Clip별 예측 CSV,
    # Confusion Matrix 그림의 실제 저장 경로를 출력합니다.
    # 학생은 이 파일들을 열어 오분류와 모델 성능을 확인합니다.

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
# ㉕ Python 파일을 직접 실행하면 main() 시작
# ==================================================
# python scripts/36_evaluate_validation.py로 직접 실행했을 때만
# main()을 호출합니다. 다른 코드에서 import하면 자동 실행되지 않습니다.

if __name__ == "__main__":
    main()