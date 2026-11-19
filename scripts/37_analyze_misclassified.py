
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# Path는 파일과 폴더 경로를 관리합니다.
# pandas는 CSV를 DataFrame으로 읽고, 두 표를 연결하며,
# 오분류 결과를 다시 CSV로 저장할 때 사용합니다.

from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT 폴더 찾기
# ==================================================
# 현재 파일이 scripts 폴더에 있다고 가정합니다.
# parents[1]은 scripts 폴더의 상위인 프로젝트 ROOT입니다.
# 이후 모든 입력·출력 경로는 ROOT를 기준으로 작성합니다.

ROOT = Path(__file__).resolve().parents[1]


# ==================================================
# ③ 36번 Validation 예측 결과 CSV 경로 설정
# ==================================================
# 36번 코드에서 저장한 validation_predictions.csv를
# 입력으로 사용합니다.
#
# 이 CSV에는 clip_id, true_label, pred_label,
# confidence, correct 등의 예측 결과가 들어 있습니다.
#
# 어떤 Clip이 틀렸는지 판단할 기본 자료입니다.

PREDICTION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "validation_predictions.csv"
)


# ==================================================
# ④ 5일차 Validation Split 경로 설정
# ==================================================
# 5일차 30번 코드에서 생성한 val.csv를 읽습니다.
#
# 이 파일의 clip_id, video_path, keypoint_csv를 활용하여
# 예측 결과를 원본 MP4와 Keypoint CSV 경로에 연결합니다.

VAL_SPLIT_PATH = (
    ROOT
    / "data"
    / "day05"
    / "splits"
    / "val.csv"
)


# ==================================================
# ⑤ 오분류 분석 결과 CSV 저장 경로 설정
# ==================================================
# 틀린 Clip만 모아 validation_misclassified.csv에 저장합니다.
#
# 원본 영상을 자동 분석하거나 재학습하는 것이 아니라,
# 사람이 오분류 원인을 확인할 수 있도록 목록을 만듭니다.

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "misclassified"
    / "validation_misclassified.csv"
)


# ==================================================
# ⑥ main() 시작 및 Validation 예측 결과 읽기
# ==================================================
# main()은 오분류 목록을 만드는 전체 과정을 실행합니다.
#
# pd.read_csv()로 36번의 예측 결과를 DataFrame으로 읽습니다.
#
# 예측 결과에는 Clip별 실제 정답과 모델 예측이 포함됩니다.

def main() -> None:
    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    # ==============================================
    # ⑦ 5일차 Validation Manifest 읽기
    # ==============================================
    # 5일차 val.csv를 DataFrame으로 읽습니다.
    #
    # 이 표에는 Clip을 식별하는 clip_id와
    # 원본 영상·Keypoint 파일 경로가 들어 있어
    # 예측 결과와 연결할 수 있습니다.

    val_manifest = pd.read_csv(
        VAL_SPLIT_PATH
    )

    # ==============================================
    # ⑧ clip_id를 기준으로 두 데이터표 연결하기
    # ==============================================
    # merge()는 두 표에서 clip_id가 같은 행을 연결합니다.
    #
    # on="clip_id":
    # Clip ID를 공통 연결 키로 사용합니다.
    #
    # how="left":
    # predictions의 모든 행을 기준으로 남깁니다.
    #
    # val_manifest에서는 video_path, keypoint_csv,
    # action_label 컬럼을 가져옵니다.
    #
    # suffixes:
    # 이름이 겹치는 컬럼에 붙일 접미사입니다.
    #
    # 주의:
    # 일치하는 Clip이 없으면 영상 경로가 NaN이 될 수 있고,
    # 동일 clip_id가 val_manifest에 중복되면
    # 결과 행이 늘어날 수 있습니다.
    #
    # 이 코드는 이러한 경우를 별도로 검사하지 않습니다.

    merged = predictions.merge(
        val_manifest[
            [
                "clip_id",
                "video_path",
                "keypoint_csv",
                "action_label",
            ]
        ],
        on="clip_id",
        how="left",
        suffixes=(
            "",
            "_manifest",
        ),
    )

    # ==============================================
    # ⑨ 실제 정답과 예측이 다른 오분류 Clip만 선택
    # ==============================================
    # true_label:
    # 실제 Action 정답입니다.
    #
    # pred_label:
    # 모델이 예측한 Action입니다.
    #
    # 두 값이 다른 행만 선택하여
    # wrong DataFrame을 만듭니다.
    #
    # .copy():
    # 필터링한 결과를 별도의 표로 복사합니다.
    #
    # 주의:
    # correct 컬럼을 사용하지 않고
    # 실제 정답과 예측 Label을 직접 비교합니다.

    wrong = merged[
        merged[
            "true_label"
        ]
        != merged[
            "pred_label"
        ]
    ].copy()

    # ==============================================
    # ⑩ 오분류 결과 저장 폴더 생성
    # ==============================================
    # OUTPUT_PATH.parent:
    # 저장할 CSV의 상위 폴더입니다.
    #
    # parents=True:
    # 필요한 상위 폴더도 함께 생성합니다.
    #
    # exist_ok=True:
    # 폴더가 이미 있어도 오류가 나지 않게 합니다.

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # ⑪ 오분류 Clip 목록을 CSV 파일로 저장
    # ==============================================
    # wrong의 모든 컬럼을
    # validation_misclassified.csv로 저장합니다.
    #
    # index=False:
    # Pandas의 행 인덱스를 제외합니다.
    #
    # utf-8-sig:
    # Excel에서 한글을 읽기 쉽게 하는 인코딩입니다.
    #
    # 오분류가 없으면 헤더만 있는 빈 결과 CSV가
    # 저장될 수 있습니다.
    #
    # 동일 경로의 기존 파일은 덮어씁니다.

    wrong.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ==============================================
    # ⑫ 전체 Validation Clip 수와 오분류 개수 출력
    # ==============================================
    # len(predictions):
    # 평가한 전체 Validation Clip 수입니다.
    #
    # len(wrong):
    # 두 Label이 달랐던 결과 행의 개수입니다.
    #
    # 정상적인 1:1 연결인 경우
    # Clip별 오분류 개수를 뜻합니다.
    #
    # 예:
    # Validation clips : 6
    # Misclassified    : 2

    print(
        f"Validation clips : "
        f"{len(predictions)}"
    )
    print(
        f"Misclassified    : "
        f"{len(wrong)}"
    )

    # ==============================================
    # ⑬ 오분류가 없는지 확인하고 안내 메시지 출력
    # ==============================================
    # wrong.empty가 True이면
    # 오분류가 한 건도 없는 상태입니다.
    #
    # 이 경우:
    # "Validation 오분류가 없습니다."를 출력합니다.
    #
    # 주의:
    # 오분류 0개가 다른 Subject에서도
    # 성능을 보장한다는 의미는 아닙니다.

    if wrong.empty:
        print()
        print(
            "Validation 오분류가 "
            "없습니다."
        )

    # ==============================================
    # ⑭ 오분류가 있다면 핵심 정보만 표로 출력
    # ==============================================
    # 오분류가 존재하면 다음 정보를 출력합니다.
    #
    # clip_id:
    # 오분류된 영상의 ID
    #
    # true_label:
    # 실제 Action
    #
    # pred_label:
    # 모델이 예측한 Action
    #
    # confidence:
    # 모델의 예측 Confidence
    #
    # video_path:
    # 원본 MP4 경로
    #
    # to_string(index=False):
    # 행 인덱스 없이 표를 출력합니다.
    #
    # 저장된 CSV에는 keypoint_csv를 포함한
    # 나머지 컬럼도 남습니다.
    #
    # 여기서는 MP4를 직접 재생하거나
    # 오류 원인을 확정하지 않습니다.

    else:
        print()
        print(
            wrong[
                [
                    "clip_id",
                    "true_label",
                    "pred_label",
                    "confidence",
                    "video_path",
                ]
            ].to_string(
                index=False
            )
        )

    # ==============================================
    # ⑮ 오분류 결과 CSV 저장 위치 출력
    # ==============================================
    # 학생이 결과 파일을 바로 열어볼 수 있도록
    # 저장한 경로를 출력합니다.
    #
    # 이 파일에서 잘못 분류된 Clip의 원본 MP4와
    # Keypoint CSV 경로를 확인하고
    # 수동으로 원인을 분석합니다.

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


# ==================================================
# ⑯ Python 파일 직접 실행 시 main() 호출
# ==================================================
# 아래 명령으로 직접 실행하면 main()을 호출합니다.
#
# python scripts/37_analyze_misclassified.py
#
# 다른 파일에서 import할 때는 자동 실행되지 않습니다.

if __name__ == "__main__":
    main()