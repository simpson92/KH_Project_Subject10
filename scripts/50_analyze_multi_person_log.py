# ====================================================
# ① 실습에 필요한 라이브러리 가져오기
# ====================================================
# argparse는 명령행 실행 옵션을 읽습니다.
# Path는 입력·출력 파일 경로를 처리합니다.
# pandas는 49번에서 저장한 CSV 로그를 분석합니다.
# 이 코드는 영상 추론을 다시 실행하지 않습니다.
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ====================================================
# ② 프로젝트 ROOT 경로 찾기
# ====================================================
# 이 파일을 scripts 폴더에 두는 구조를 기준으로 합니다.
# parents[1]은 scripts의 상위인 프로젝트 루트입니다.
ROOT = Path(__file__).resolve().parents[1]

# ====================================================
# ③ 49번 Runtime 로그 입력 폴더 지정
# ====================================================
# 49번 프로그램이 만든 ip_multi_*.csv가 모이는 폴더입니다.
# 기본 위치는 reports/day08/ip_webcam입니다.
DEFAULT_INPUT_DIR = (
    ROOT / "reports" / "day08" / "ip_webcam"
)

# ====================================================
# ④ Track ID별 요약 CSV 출력 경로 지정
# ====================================================
# 입력 폴더 안에 ip_multi_runtime_summary.csv를 저장합니다.
# 이 파일은 개별 Frame 로그가 아니라 Track ID별 요약표입니다.
DEFAULT_OUTPUT_PATH = (
    DEFAULT_INPUT_DIR
    / "ip_multi_runtime_summary.csv"
)


# ====================================================
# ⑤ 실행 옵션을 읽는 parse_args 함수 시작
# ====================================================
# argparse.ArgumentParser()를 생성하여 터미널 옵션을 관리합니다.
# 사용자는 특정 CSV, 입력 폴더, 출력 경로를 선택할 수 있습니다.
def parse_args():
    parser = argparse.ArgumentParser()

    # ====================================================
    # ⑥ 특정 Runtime 로그를 직접 지정하는 --log 옵션
    # ====================================================
    # --log를 지정하면 그 파일 하나만 분석합니다.
    # 생략하면 입력 폴더의 ip_multi_*.csv 중 이름순 마지막 파일을 선택합니다.
    # 파일명이 일정한 날짜·시각 형식일 때 최신 실행 로그와 일치합니다.
    parser.add_argument(
        "--log",
        default=None,
        help=(
            "생략하면 입력 폴더의 가장 최근 "
            "ip_multi_*.csv 사용"
        ),
    )

    # ====================================================
    # ⑦ 자동 검색할 로그 폴더를 지정하는 --input-dir 옵션
    # ====================================================
    # 자동 선택 모드에서 검색할 CSV 폴더를 설정합니다.
    # 생략하면 DEFAULT_INPUT_DIR을 사용합니다.
    parser.add_argument(
        "--input-dir",
        default=str(
            DEFAULT_INPUT_DIR
        ),
    )

    # ====================================================
    # ⑧ 결과 파일 위치를 지정하는 --output 옵션
    # ====================================================
    # Track ID별 요약 CSV를 어디에 저장할지 지정합니다.
    # 생략하면 DEFAULT_OUTPUT_PATH를 사용합니다.
    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    # ====================================================
    # ⑨ 사용자가 입력한 명령행 옵션 반환
    # ====================================================
    # parse_args()가 해석한 옵션을 main()으로 전달합니다.
    # 결과에는 log, input_dir, output 속성이 들어 있습니다.
    return parser.parse_args()


# ====================================================
# ⑩ 분석할 로그를 선택하는 choose_log 함수 정의
# ====================================================
# --log를 직접 받았는지 먼저 확인합니다.
# 직접 지정하면 자동 검색을 건너뛰고 해당 경로를 사용합니다.
def choose_log(
    value,
    input_dir: Path,
    output_path: Path,
) -> Path:
    # ====================================================
    # ⑪ 직접 지정한 로그 경로를 Path로 변환
    # ====================================================
    # --log에 값이 있으면 문자열을 Path 객체로 바꿉니다.
    # 이때 자동 검색할 파일 목록은 만들지 않습니다.
    if value:
        path = Path(value)

        # ====================================================
        # ⑫ 직접 지정 경로가 상대경로라면 ROOT와 연결
        # ====================================================
        # 상대경로를 프로젝트 ROOT 기준의 전체 경로로 만듭니다.
        # 이미 절대경로라면 그대로 유지합니다.
        if not path.is_absolute():
            path = ROOT / path

        # ====================================================
        # ⑬ 직접 지정한 파일 경로를 선택 결과로 반환
        # ====================================================
        # 자동 검색 대신 직접 지정한 경로를 즉시 반환합니다.
        # 파일 존재 여부는 main()에서 검사합니다.
        return path

    # ====================================================
    # ⑭ 입력 폴더에서 ip_multi_*.csv 이름순 검색
    # ====================================================
    # glob()는 입력 폴더 바로 아래의 일치하는 CSV를 찾습니다.
    # sorted()는 파일 이름·경로 순으로 정렬합니다.
    # 수정 시각 순으로 정렬하는 코드는 아니라는 점에 주의하세요.
    files = sorted(
        input_dir.glob("ip_multi_*.csv")
    )

    # ====================================================
    # ⑮ 요약 CSV 자체를 검색 후보에서 제외
    # ====================================================
    # 출력 예정인 ip_multi_runtime_summary.csv를 다시 입력으로
    # 읽지 않도록 절대 경로(resolve)로 비교하여 제외합니다.
    files = [
        path
        for path in files
        if path.resolve()
        != output_path.resolve()
    ]

    # ====================================================
    # ⑯ 후보가 없으면 오류, 있으면 마지막 파일 선택
    # ====================================================
    # 후보 목록이 비었다면 RuntimeError를 발생시킵니다.
    # 후보가 있다면 이름순 마지막 files[-1]을 선택합니다.
    # 이 방식은 파일 수정 시각 기준의 최신 선택과는 다릅니다.
    if not files:
        raise RuntimeError(
            "Multi-Person IP Webcam Log가 없습니다."
        )

    return files[-1]


# ====================================================
# ⑰ 가장 자주 나온 Stable Action을 찾는 함수 정의
# ====================================================
# 인자로 전달받은 예측 갱신 행에서 stable_label을 읽습니다.
# 결측값을 빈 문자열로 채우고 문자형으로 통일합니다.
def most_common_stable_label(
    group: pd.DataFrame,
) -> str:
    labels = (
        group["stable_label"]
        .fillna("")
        .astype(str)
    )

    # ====================================================
    # ⑱ WARMING_UP·UNKNOWN·빈 Label 제외
    # ====================================================
    # 아직 준비 중인 WARMING_UP, 확신이 낮은 UNKNOWN,
    # 비어 있는 Label은 대표 Action 후보에서 제외합니다.
    # 물결표(~)는 isin() 조건을 반전시키는 역할을 합니다.
    labels = labels[
        ~labels.isin(
            ["", "WARMING_UP", "UNKNOWN"]
        )
    ]

    # ====================================================
    # ⑲ 유효 Action이 없다면 빈 문자열 반환, 있으면 최빈값 선택
    # ====================================================
    # 유효 Action이 없으면 빈 문자열을 반환합니다.
    # 그렇지 않으면 value_counts()로 빈도를 계산하고
    # 가장 자주 등장한 Action 이름을 반환합니다.
    if labels.empty:
        return ""

    return str(
        labels.value_counts().index[0]
    )


# ====================================================
# ⑳ main 시작 및 터미널 입력값 읽기
# ====================================================
# main()은 CSV를 읽고 Track ID별 지표를 요약해 저장합니다.
# 먼저 parse_args()에서 사용자가 지정한 옵션을 가져옵니다.
def main() -> None:
    args = parse_args()

    # ====================================================
    # ㉑ 입력 폴더와 출력 CSV 경로를 Path로 변환
    # ====================================================
    # 입력 폴더와 출력 파일을 각각 Path로 변환합니다.
    # 이후 경로 결합 및 폴더 생성에 사용합니다.
    input_dir = Path(
        args.input_dir
    )

    output_path = Path(
        args.output
    )

    # ====================================================
    # ㉒ 상대경로를 프로젝트 ROOT 기준으로 변환
    # ====================================================
    # 입력 폴더와 출력 파일 각각 절대경로인지 확인합니다.
    # 상대경로라면 ROOT를 앞에 붙여 실행 위치에 따른 혼동을 줄입니다.
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    # ====================================================
    # ㉓ 직접 지정 또는 자동 검색으로 분석 로그 결정
    # ====================================================
    # choose_log()에서 사용할 CSV 하나를 정합니다.
    # 입력 폴더에 로그가 여럿 있어도 한 파일만 선택합니다.
    log_path = choose_log(
        args.log,
        input_dir,
        output_path,
    )

    # ====================================================
    # ㉔ 선택한 Runtime CSV가 실제로 존재하는지 확인
    # ====================================================
    # 지정된 경로에 파일이 없으면 FileNotFoundError를 냅니다.
    # 파일을 읽기 전에 먼저 존재 여부를 검사합니다.
    if not log_path.exists():
        raise FileNotFoundError(
            log_path
        )

    # ====================================================
    # ㉕ 49번 Track ID별 Runtime CSV를 Pandas로 읽기
    # ====================================================
    # 49번의 로그에는 Frame마다 보인 사람의 Track ID와
    # Raw·Threshold·Stable Label 및 Confidence·FPS 등이 있습니다.
    # pd.read_csv()로 DataFrame 형태로 불러옵니다.
    df = pd.read_csv(
        log_path
    )

    # ====================================================
    # ㉖ 입력 로그에 행이 없다면 실행 중단
    # ====================================================
    # 컬럼만 있고 데이터 행이 없으면 분석할 정보가 없습니다.
    # 따라서 RuntimeError로 명확히 알려줍니다.
    if df.empty:
        raise RuntimeError(
            "Runtime Log가 비어 있습니다."
        )

    # ====================================================
    # ㉗ 요약 결과 목록을 만들고 Track ID별로 그룹화
    # ====================================================
    # rows는 Track ID 하나의 요약 결과를 한 행씩 모으는 목록입니다.
    # groupby("track_id")는 같은 Track ID의 모든 기록을 묶습니다.
    rows = []

    for track_id, group in df.groupby(
        "track_id"
    ):
        # ====================================================
        # ㉘ 실제로 새 Action 예측이 발생한 행만 추출
        # ====================================================
        # prediction_updated == 1인 행만 updates로 복사합니다.
        # Window 예측이 갱신되지 않은 Frame의 반복 Label을
        # UNKNOWN 비율·Confidence 계산에 중복 반영하지 않습니다.
        updates = group[
            group["prediction_updated"] == 1
        ].copy()

        # ====================================================
        # ㉙ 예측 갱신이 없으면 Action 관련 지표를 공란 처리
        # ====================================================
        # 추적 ID는 보였지만 Window가 준비되지 않아
        # 새 예측이 한 번도 없을 수 있습니다.
        # 이 경우 UNKNOWN 비율·평균 Confidence·대표 Label을
        # 0이 아닌 빈 문자열로 두어 '측정 불가'를 나타냅니다.
        if updates.empty:
            unknown_ratio = ""
            mean_confidence = ""
            common_label = ""
        else:
            # ====================================================
            # ㉚ 예측 갱신 행 중 UNKNOWN이 나온 비율 계산
            # ====================================================
            # Threshold를 적용한 Label이 UNKNOWN인 횟수 비율입니다.
            # 예측 갱신 10회 중 UNKNOWN이 3회면 0.3이 됩니다.
            # mean()은 True를 1, False를 0으로 계산합니다.
            unknown_ratio = round(
                float(
                    (
                        updates["threshold_label"]
                        == "UNKNOWN"
                    ).mean()
                ),
                4,
            )

            # ====================================================
            # ㉛ 현재 Track ID의 평균 예측 Confidence 계산
            # ====================================================
            # 새 예측이 수행된 행들의 confidence 평균입니다.
            # UNKNOWN 판정도 원래 Confidence 값은 평균에 포함됩니다.
            # 계산 결과는 소수 넷째 자리로 반올림합니다.
            mean_confidence = round(
                float(
                    updates["confidence"].mean()
                ),
                4,
            )

            # ====================================================
            # ㉜ 예측 갱신 중 가장 많이 나온 Stable Action 찾기
            # ====================================================
            # most_common_stable_label()에 updates를 전달합니다.
            # WARMING_UP·UNKNOWN을 제외한 실제 Action 가운데
            # 가장 자주 나온 Stable Label을 대표 Action으로 사용합니다.
            common_label = (
                most_common_stable_label(
                    updates
                )
            )

        # ====================================================
        # ㉝ 현재 Track ID의 양수 Processing FPS 값 선택
        # ====================================================
        # FPS 평균은 예측 갱신 행만이 아닌 전체 group을 사용합니다.
        # FPS가 0보다 큰 행만 가져옵니다.
        # processing_fps는 프로그램 처리 속도이지 모델 정확도가 아닙니다.
        fps_values = group.loc[
            group["processing_fps"] > 0,
            "processing_fps",
        ]

        # ====================================================
        # ㉞ Track ID별 시간·Frame·예측 지표를 한 행으로 정리
        # ====================================================
        # 한 사람 ID의 추적 시작·종료 경과 시간(min/max),
        # 관찰된 서로 다른 Frame 수(nunique), 예측 갱신 횟수,
        # UNKNOWN 비율·대표 Stable Action·평균 Confidence와 FPS를
        # 딕셔너리로 묶어 rows에 추가합니다.
        # first_seen_sec·last_seen_sec은 세션 시작 후 경과 시간이며
        # 실제 시계 시각이 아닙니다.
        rows.append(
            {
                "track_id": int(
                    track_id
                ),
                "first_seen_sec": round(
                    float(
                        group["elapsed_sec"].min()
                    ),
                    3,
                ),
                "last_seen_sec": round(
                    float(
                        group["elapsed_sec"].max()
                    ),
                    3,
                ),
                "observed_frames": int(
                    group["frame_id"].nunique()
                ),
                "prediction_updates": len(
                    updates
                ),
                "unknown_ratio": (
                    unknown_ratio
                ),
                "most_common_stable_label": (
                    common_label
                ),
                "mean_confidence": (
                    mean_confidence
                ),
                "mean_processing_fps": (
                    round(
                        float(
                            fps_values.mean()
                        ),
                        3,
                    )
                    if not fps_values.empty
                    else 0.0
                ),
            }
        )

    # ====================================================
    # ㉟ 모든 Track ID의 요약표를 DataFrame으로 만들고 ID순 정렬
    # ====================================================
    # rows에 누적된 각 Track ID의 분석 결과를 표로 만듭니다.
    # sort_values("track_id")로 ID 번호가 작은 순서대로 정렬합니다.
    result = (
        pd.DataFrame(rows)
        .sort_values("track_id")
    )

    # ====================================================
    # ㊱ 결과 폴더를 준비하고 UTF-8 CSV로 저장
    # ====================================================
    # 저장 경로의 부모 폴더가 없으면 생성합니다.
    # index=False는 Pandas 인덱스를 제외합니다.
    # utf-8-sig는 Excel에서 한글이 깨지는 일을 줄입니다.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ====================================================
    # ㊲ 선택한 로그·요약표·해석 주의사항·저장 경로 출력
    # ====================================================
    # 분석에 사용된 입력 CSV 경로를 출력합니다.
    # Track ID별 요약표와 저장 위치를 터미널에 보여줍니다.
    # 이 요약값은 실시간 관찰 지표이며 정식 Action Accuracy는 아닙니다.
    print(
        f"Log: {log_path}"
    )

    print()

    print(
        result.to_string(
            index=False
        )
    )

    print()

    print(
        "주의: 이 표는 Runtime 관찰용이며 "
        "정식 Action Accuracy가 아닙니다."
    )

    print(
        f"Saved: {output_path}"
    )


# ====================================================
# ㊳ 파일을 직접 실행할 때 main() 호출
# ====================================================
# 이 스크립트를 python 명령어로 직접 실행하면
# main()을 호출해 분석을 시작합니다.
if __name__ == "__main__":
    main()