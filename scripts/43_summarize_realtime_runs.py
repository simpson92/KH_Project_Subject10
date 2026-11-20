
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse는 터미널 실행 옵션을 읽습니다.
# Path는 입력·출력 파일 경로를 관리합니다.
# pandas는 CSV 로그를 읽고 요약표를 만듭니다.

import argparse
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT 찾기
# ==================================================
# 이 파일을 scripts 폴더에 저장했다고 가정합니다.
# parents[1]은 scripts의 상위인 프로젝트 ROOT입니다.

ROOT = Path(__file__).resolve().parents[1]


# ==================================================
# ③ 기본 입력 폴더와 요약 결과 파일 경로 지정
# ==================================================
# DEFAULT_INPUT_DIR:
# 42번에서 만든 Frame별 CSV 로그가 저장된 폴더입니다.
#
# DEFAULT_OUTPUT_PATH:
# 여러 실험 결과를 하나의 표로 요약할 CSV 경로입니다.

DEFAULT_INPUT_DIR = (
    ROOT
    / "reports"
    / "day07"
    / "saved_video"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day07"
    / "realtime_run_summary.csv"
)


# ==================================================
# ④ 터미널 실행 옵션을 처리하는 함수 정의
# ==================================================
# argparse를 사용하면 실행할 때
# 입력 폴더와 출력 파일을 변경할 수 있습니다.
#
# 옵션을 생략하면 위에서 정한 기본 경로를 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    # ==============================================
    # ⑤ CSV 입력 폴더 옵션 설정
    # ==============================================
    # --input-dir로 분석할 CSV 폴더를 지정합니다.
    #
    # 예:
    # --input-dir reports/day07/saved_video

    parser.add_argument(
        "--input-dir",
        default=str(
            DEFAULT_INPUT_DIR
        ),
    )

    # ==============================================
    # ⑥ 결과 CSV 저장 경로 옵션 설정
    # ==============================================
    # --output으로 요약 CSV의 이름이나
    # 저장 위치를 변경할 수 있습니다.
    #
    # parse_args()는 입력한 옵션을 반환합니다.

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


# ==================================================
# ⑦ Label 변화 횟수 계산 함수 정의
# ==================================================
# label_changes()는 Action Label이
# 이전 예측과 비교해 몇 번 바뀌었는지 계산합니다.
#
# 예:
# wave → wave → sit_stand → wave
#
# Label 변화 횟수는 2회입니다.
#
# 이 함수에는 실제 예측이 갱신된 행의
# Raw 또는 Stable Label을 전달합니다.

def label_changes(
    series: pd.Series,
) -> int:

    # ==============================================
    # ⑧ 결측값을 정리하고 문자열로 변환
    # ==============================================
    # fillna("")는 결측값을 빈 문자열로 바꿉니다.
    #
    # astype(str)은 Label 값을 문자열로 통일합니다.
    #
    # 이후 이전 Label과 현재 Label을 비교합니다.

    clean = (
        series
        .fillna("")
        .astype(str)
    )

    # ==============================================
    # ⑨ Label 목록이 비었으면 0 반환
    # ==============================================
    # 비교할 예측 결과가 없다면
    # 변화 횟수는 0으로 처리합니다.

    if clean.empty:
        return 0

    # ==============================================
    # ⑩ 이전 Label과 비교해 변경 횟수 계산
    # ==============================================
    # shift(1)은 이전 행의 Label을 가져옵니다.
    #
    # 현재 Label != 이전 Label이면 True입니다.
    #
    # sum()은 True 개수를 셉니다.
    #
    # 첫 행에는 이전 Label이 없으므로
    # 첫 행에서 발생한 차이 1개를 제외합니다.
    #
    # 예:
    # wave → wave → sit_stand → wave
    # 결과: 2회

    return int(
        (
            clean
            != clean.shift(1)
        ).sum()
        - 1
    )


# ==================================================
# ⑪ main() 시작 및 터미널 옵션 읽기
# ==================================================
# main()은 여러 실험 CSV를 읽고
# 실행별 비교표를 만드는 전체 함수입니다.
#
# 먼저 사용자가 입력한 실행 옵션을 가져옵니다.

def main() -> None:
    args = parse_args()

    # ==============================================
    # ⑫ 입력 폴더와 출력 파일을 Path로 변환
    # ==============================================
    # argparse에서 받은 문자열 경로를
    # Path 객체로 변환합니다.
    #
    # Path를 사용하면 경로 연결이나
    # 절대경로 검사를 쉽게 할 수 있습니다.

    input_dir = Path(
        args.input_dir
    )

    output_path = Path(
        args.output
    )

    # ==============================================
    # ⑬ 상대경로를 프로젝트 ROOT 기준으로 변환
    # ==============================================
    # 절대경로라면 그대로 사용합니다.
    #
    # 상대경로라면 ROOT를 앞에 연결합니다.
    #
    # 입력 폴더와 출력 파일 모두 처리합니다.

    if not input_dir.is_absolute():
        input_dir = (
            ROOT
            / input_dir
        )

    if not output_path.is_absolute():
        output_path = (
            ROOT
            / output_path
        )

    # ==============================================
    # ⑭ 입력 폴더에서 CSV 파일 검색
    # ==============================================
    # glob("*.csv")로 입력 폴더의 CSV를 찾습니다.
    #
    # sorted()로 파일 이름순으로 정렬합니다.
    #
    # 현재 코드는 하위 폴더까지 검색하지 않습니다.

    files = sorted(
        input_dir.glob(
            "*.csv"
        )
    )

    # ==============================================
    # ⑮ CSV 파일이 없으면 오류 발생
    # ==============================================
    # CSV가 없다면 분석할 실행 로그가 없으므로
    # RuntimeError를 발생시키고 중단합니다.

    if not files:
        raise RuntimeError(
            "Day07 저장영상 실행 CSV가 "
            "없습니다."
        )

    # ==============================================
    # ⑯ 실행별 결과 목록 준비 및 CSV 반복 읽기
    # ==============================================
    # rows는 실행별 요약 결과를 저장할 목록입니다.
    #
    # for 반복문으로 CSV 파일을 하나씩 읽습니다.
    #
    # 각 파일은 42번 저장영상 실습의
    # Frame별 로그라고 가정합니다.

    rows = []

    for path in files:
        df = pd.read_csv(
            path
        )

        # ==========================================
        # ⑰ prediction_updated 컬럼 존재 여부 확인
        # ==========================================
        # prediction_updated는 실제로 새로운
        # Action 예측이 수행되었는지 나타냅니다.
        #
        # 1: 새 예측이 수행됨
        # 0: 새 예측이 수행되지 않음
        #
        # 컬럼이 없으면 해당 CSV를 건너뜁니다.

        if (
            "prediction_updated"
            not in df.columns
        ):
            continue

        # ==========================================
        # ⑱ 실제 예측이 갱신된 행만 선택
        # ==========================================
        # 42번 CSV에는 모든 Frame의 정보가 있습니다.
        #
        # 하지만 모델은 설정된 예측 간격에 따라
        # 일부 Frame에서만 새 예측을 수행합니다.
        #
        # 따라서 prediction_updated == 1인 행만
        # 따로 추출합니다.
        #
        # 이후 UNKNOWN 비율과 Label 변경 횟수는
        # 이 예측 갱신 행들을 기준으로 계산합니다.

        prediction_rows = df[
            df[
                "prediction_updated"
            ]
            == 1
        ].copy()

        # ==========================================
        # ⑲ 예측이 한 번도 갱신되지 않았다면 건너뛰기
        # ==========================================
        # Window가 준비되지 않아 예측하지 못했거나
        # 영상이 너무 짧은 경우일 수 있습니다.
        #
        # 예측 갱신 행이 없으면
        # 해당 CSV는 비교표에 추가하지 않습니다.

        if prediction_rows.empty:
            continue

        # ==========================================
        # ⑳ 원본 영상의 기대 Action Label 확인
        # ==========================================
        # expected_label은 42번 코드가
        # MP4 파일명에서 추론한 참조 정답입니다.
        #
        # 예:
        # S02_A006_T01.mp4
        # → wave
        #
        # 예측 갱신 행의 첫 번째 값을 가져옵니다.
        #
        # 값이 없으면 빈 문자열로 처리합니다.

        expected_label = str(
            prediction_rows[
                "expected_label"
            ]
            .fillna("")
            .iloc[0]
        )

        # ==========================================
        # ㉑ UNKNOWN 예측 비율 계산
        # ==========================================
        # threshold_label은 Confidence 기준을
        # 적용한 뒤의 예측 결과입니다.
        #
        # 기준값을 만족하지 못하면
        # UNKNOWN으로 표시될 수 있습니다.
        #
        # UNKNOWN 비율은 다음과 같습니다.
        #
        # UNKNOWN 예측 수 / 전체 예측 갱신 수
        #
        # 예:
        # 예측 10회 중 UNKNOWN 3회
        # → unknown_ratio = 0.3

        unknown_ratio = float(
            (
                prediction_rows[
                    "threshold_label"
                ]
                == "UNKNOWN"
            ).mean()
        )

        # ==========================================
        # ㉒ Raw Label 변경 횟수 계산
        # ==========================================
        # Raw Label은 Confidence Threshold와
        # 안정화 처리를 하기 전의 예측 결과입니다.
        #
        # label_changes()를 이용해
        # 예측이 몇 번 바뀌었는지 계산합니다.
        #
        # 변경 횟수가 많다면
        # 예측이 자주 흔들리고 있을 수 있습니다.

        raw_changes = (
            label_changes(
                prediction_rows[
                    "raw_label"
                ]
            )
        )

        # ==========================================
        # ㉓ Stable Label 변경 횟수 계산
        # ==========================================
        # Stable Label은 최근 예측을 종합하여
        # 안정화한 결과입니다.
        #
        # Raw Label과 같은 방법으로
        # 변경 횟수를 계산합니다.
        #
        # Raw와 Stable의 변화 횟수를 비교하면
        # 안정화 처리의 영향을 확인할 수 있습니다.

        stable_changes = (
            label_changes(
                prediction_rows[
                    "stable_label"
                ]
            )
        )

        # ==========================================
        # ㉔ Stable Label과 기대 Action의 일치율 계산
        # ==========================================
        # expected_label이 있으면
        # Stable Label과 비교합니다.
        #
        # 계산:
        # 참조 정답과 일치한 예측 수 / 전체 예측 수
        #
        # expected_label이 없다면
        # 일치율을 계산할 수 없으므로 NaN을 사용합니다.
        #
        # 결과 CSV에서는 NaN을 공란으로 처리합니다.

        if expected_label:
            stable_match_ratio = float(
                (
                    prediction_rows[
                        "stable_label"
                    ]
                    == expected_label
                ).mean()
            )
        else:
            stable_match_ratio = float(
                "nan"
            )

        # ==========================================
        # ㉕ 평균 Processing FPS 계산에 사용할 값 선택
        # ==========================================
        # processing_fps는 프로그램의
        # Frame 처리 속도를 나타냅니다.
        #
        # 0보다 큰 FPS 값만 선택합니다.
        #
        # 이 값은 prediction_rows가 아닌
        # 전체 Frame 로그 df에서 가져옵니다.

        fps_values = df.loc[
            df[
                "processing_fps"
            ]
            > 0,
            "processing_fps",
        ]

        # ==========================================
        # ㉖ 실행별 요약 결과를 딕셔너리로 만들기
        # ==========================================
        # 하나의 CSV 실험 결과를 한 행으로 정리합니다.
        #
        # run:
        # 파일 이름에서 확장자를 제거한 실행 이름
        #
        # window_seconds:
        # Action을 판단할 시간 Window
        #
        # confidence_threshold:
        # UNKNOWN 판단에 사용하는 Confidence 기준
        #
        # stabilization_buffer:
        # 안정화에 사용하는 최근 예측 개수
        #
        # prediction_updates:
        # 실제 Action 예측이 갱신된 횟수
        #
        # unknown_ratio:
        # UNKNOWN 결과 비율
        #
        # raw_label_changes / stable_label_changes:
        # 안정화 전후 Label 변경 횟수
        #
        # stable_match_ratio:
        # Stable Label과 참조 정답의 일치율
        #
        # mean_confidence:
        # 갱신된 예측의 평균 Confidence
        #
        # mean_processing_fps:
        # 유효한 Frame 처리 FPS의 평균

        rows.append(
            {
                "run": path.stem,
                "expected_label": (
                    expected_label
                ),
                "window_seconds": (
                    prediction_rows[
                        "window_seconds"
                    ].iloc[0]
                    if "window_seconds"
                    in prediction_rows.columns
                    else ""
                ),
                "confidence_threshold": (
                    prediction_rows[
                        "confidence_threshold"
                    ].iloc[0]
                    if "confidence_threshold"
                    in prediction_rows.columns
                    else ""
                ),
                "stabilization_buffer": (
                    prediction_rows[
                        "stabilization_buffer"
                    ].iloc[0]
                    if "stabilization_buffer"
                    in prediction_rows.columns
                    else ""
                ),
                "stabilization_min_votes": (
                    prediction_rows[
                        "stabilization_min_votes"
                    ].iloc[0]
                    if "stabilization_min_votes"
                    in prediction_rows.columns
                    else ""
                ),
                "prediction_updates": (
                    len(
                        prediction_rows
                    )
                ),
                "unknown_ratio": round(
                    unknown_ratio,
                    4,
                ),
                "raw_label_changes": (
                    raw_changes
                ),
                "stable_label_changes": (
                    stable_changes
                ),
                "stable_match_ratio": (
                    round(
                        stable_match_ratio,
                        4,
                    )
                    if pd.notna(
                        stable_match_ratio
                    )
                    else ""
                ),
                "mean_confidence": round(
                    float(
                        prediction_rows[
                            "confidence"
                        ].mean()
                    ),
                    4,
                ),
                "mean_processing_fps": round(
                    float(
                        fps_values.mean()
                    ),
                    3,
                )
                if not fps_values.empty
                else 0.0,
            }
        )

    # ==============================================
    # ㉗ 요약할 실행 결과가 없다면 오류 발생
    # ==============================================
    # CSV가 있어도 모든 실험에서
    # prediction_updated=1인 행이 없다면
    # rows는 비어 있게 됩니다.
    #
    # 요약할 데이터가 없으면 RuntimeError를 냅니다.

    if not rows:
        raise RuntimeError(
            "요약할 Prediction 결과가 "
            "없습니다."
        )

    # ==============================================
    # ㉘ 실행별 요약 목록을 DataFrame으로 변환
    # ==============================================
    # rows에 저장된 여러 실행 결과를
    # 하나의 DataFrame으로 만듭니다.
    #
    # CSV 한 개당 하나의 결과 행이 만들어집니다.

    result = pd.DataFrame(
        rows
    )

    # ==============================================
    # ㉙ 출력 폴더 생성 및 요약 CSV 저장
    # ==============================================
    # 저장할 폴더가 없다면 생성합니다.
    #
    # index=False:
    # Pandas 인덱스를 CSV에 저장하지 않습니다.
    #
    # utf-8-sig:
    # Excel에서 한글을 읽기 쉽게 저장합니다.

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ==============================================
    # ㉚ 실행별 비교표와 저장 경로 출력
    # ==============================================
    # 여러 실험의 요약 결과를 터미널에 출력합니다.
    #
    # 마지막으로 결과 CSV 저장 위치를 출력합니다.

    print(
        result.to_string(
            index=False
        )
    )
    print()
    print(
        f"Saved: {output_path}"
    )


# ==================================================
# ㉛ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# 예:
# python scripts/43_summarize_saved_video_logs.py

if __name__ == "__main__":
    main()