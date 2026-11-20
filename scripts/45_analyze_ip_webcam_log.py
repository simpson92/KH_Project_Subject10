from __future__ import annotations

# ========================================================
# ① 필요한 라이브러리 가져오기
# ========================================================
# argparse는 터미널 옵션, Path는 파일 경로, pandas는 CSV 데이터 처리를 맡습니다.
# 44번 IP Webcam 프로그램이 이미 저장한 로그를 분석하므로 YOLO와 OpenCV는 필요하지 않습니다.

import argparse
from pathlib import Path

import pandas as pd


# ========================================================
# ② 프로젝트 최상위 ROOT 폴더 찾기
# ========================================================
# 현재 Python 파일이 scripts 폴더에 있다고 가정합니다.
# parents[1]은 scripts의 상위인 프로젝트 ROOT를 가리킵니다.

ROOT = Path(__file__).resolve().parents[1]

# ========================================================
# ③ 입력 로그 폴더와 출력 요약 CSV 위치 지정
# ========================================================
# 입력: reports/day07/ip_webcam 폴더의 44번 Runtime 로그입니다.
# 출력: 같은 폴더의 ip_webcam_runtime_summary.csv입니다.
# 요약 CSV가 다음 실행에서 원본 로그로 오인되지 않도록 뒤에서 제외합니다.

DEFAULT_INPUT_DIR = (
    ROOT
    / "reports"
    / "day07"
    / "ip_webcam"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day07"
    / "ip_webcam"
    / "ip_webcam_runtime_summary.csv"
)

# ========================================================
# ④ Runtime 로그에 필요한 필수 컬럼 정의
# ========================================================
# expected_label은 키보드로 정한 Target, prediction_updated는 새 예측 여부입니다.
# Raw·Threshold·Stable Label, Confidence, Processing FPS도 필요합니다.
# 한 컬럼이라도 없다면 이 스크립트는 해당 로그를 분석할 수 없습니다.

REQUIRED_COLUMNS = {
    "expected_label",
    "prediction_updated",
    "threshold_label",
    "stable_label",
    "raw_label",
    "confidence",
    "processing_fps",
}


# ========================================================
# ⑤ 실행 옵션을 읽는 parse_args 함수 시작
# ========================================================
# 터미널에서 특정 CSV나 입력 폴더·출력 경로를 바꿀 수 있도록 합니다.
# 아무 옵션도 입력하지 않았을 때 사용할 기본값도 함께 준비합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    # ========================================================
    # ⑥ 특정 로그를 직접 지정하는 --log 옵션
    # ========================================================
    # --log 뒤에 CSV 경로를 적으면 그 파일만 분석합니다.
    # 입력하지 않으면 가장 최근의 분석 가능한 로그를 자동으로 찾습니다.
    #
    parser.add_argument(
        "--log",
        default=None,
        help=(
            "특정 Runtime Log를 직접 지정합니다. "
            "생략하면 입력 폴더에서 가장 최근의 "
            "분석 가능한 Log를 자동으로 찾습니다."
        ),
    )

    # ========================================================
    # ⑦ 로그 탐색 폴더를 바꾸는 --input-dir 옵션
    # ========================================================
    # --input-dir를 생략하면 reports/day07/ip_webcam을 검색합니다.
    # 사용자가 다른 폴더를 지정하면 그 폴더에서 Runtime CSV를 찾습니다.
    #
    parser.add_argument(
        "--input-dir",
        default=str(
            DEFAULT_INPUT_DIR
        ),
        help="IP Webcam Runtime Log가 저장된 폴더",
    )

    # ========================================================
    # ⑧ 요약 CSV 저장 위치를 바꾸는 --output 옵션
    # ========================================================
    # --output은 요약 CSV의 출력 경로를 바꿉니다.
    # parse_args()는 세 옵션을 실제 입력값으로 해석해서 반환합니다.
    #
    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help="Action별 Runtime 요약 CSV 저장 경로",
    )

    return parser.parse_args()


# ========================================================
# ⑨ 절대경로와 상대경로를 통일하는 함수
# ========================================================
# resolve_path()는 받은 문자열 또는 Path를 Path 객체로 만듭니다.
# 상대경로라면 프로젝트 ROOT와 연결하고, 절대경로라면 그대로 사용합니다.

def resolve_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path


# ========================================================
# ⑩ Runtime CSV를 읽는 공통 함수 시작
# ========================================================
# read_runtime_log()는 CSV 전체, 실제 분석 가능한 행, 진단 정보를 반환합니다.
# 분석 가능한 행은 Target이 정해졌고 새 예측이 발생한 경우입니다.

def read_runtime_log(
    path: Path,
):
    df = pd.read_csv(
        path
    )

    # ========================================================
    # ⑪ 필수 컬럼 누락 여부 검사
    # ========================================================
    # REQUIRED_COLUMNS와 CSV 컬럼의 차집합으로 누락된 항목을 찾습니다.
    # 누락이 있으면 ValueError를 발생시켜 잘못된 파일을 분석하지 않습니다.
    #
    missing = sorted(
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "필수 Column이 없습니다: "
            + ", ".join(missing)
        )

    # ========================================================
    # ⑫ 키보드로 설정한 Target 문자열 정리
    # ========================================================
    # expected_label 결측값은 빈 문자열로 바꾸고 공백을 제거합니다.
    # Target이 없는 행을 판별할 때 wave 같은 실제 Action 이름만 남도록 합니다.
    #
    expected = (
        df[
            "expected_label"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # ⑬ 새 Prediction 발생 행의 조건 만들기
    # ========================================================
    # prediction_updated를 숫자로 변환한 뒤 값이 1인 행만 True로 표시합니다.
    # 잘못된 숫자는 NaN→0으로 처리하며, 갱신되지 않은 Frame은 제외됩니다.
    #
    prediction_mask = (
        pd.to_numeric(
            df[
                "prediction_updated"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        == 1
    )

    # ========================================================
    # ⑭ Target이 지정된 행만 가리는 조건 만들기
    # ========================================================
    # expected_label이 빈 문자열이 아닌 행을 True로 표시합니다.
    # Target을 누르지 않은 상태의 예측은 정답 일치율 계산에서 제외해야 합니다.
    #
    target_mask = (
        expected
        != ""
    )

    # ========================================================
    # ⑮ Target과 실제 Prediction을 모두 만족한 행 추출
    # ========================================================
    # 두 조건을 &로 결합해 usable DataFrame을 만듭니다.
    # 즉 Target 지정과 prediction_updated=1이 같은 행에서 성립해야 합니다.
    #
    usable = df[
        prediction_mask
        & target_mask
    ].copy()

    # ========================================================
    # ⑯ 분석 대상 행의 Target 표기를 다시 통일
    # ========================================================
    # usable이 비어 있지 않을 때 expected_label을 문자열로 정리합니다.
    # 그룹화 시 앞뒤 공백 때문에 같은 Action이 다른 그룹으로 나뉘지 않게 합니다.
    #
    if not usable.empty:
        usable[
            "expected_label"
        ] = (
            usable[
                "expected_label"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # ========================================================
    # ⑰ 입력 로그 진단 정보 계산 및 반환
    # ========================================================
    # rows는 전체 Frame 로그 수, target_rows는 Target 지정 행 수입니다.
    # prediction_rows는 예측 갱신 행 수, usable_rows는 두 조건을 모두 만족한 행 수입니다.
    # 세 결과(df, usable, diagnostics)를 반환합니다.
    #
    diagnostics = {
        "rows": int(
            len(df)
        ),
        "target_rows": int(
            target_mask.sum()
        ),
        "prediction_rows": int(
            prediction_mask.sum()
        ),
        "usable_rows": int(
            len(usable)
        ),
    }

    return (
        df,
        usable,
        diagnostics,
    )


# ========================================================
# ⑱ 자동 선택할 Runtime 로그 후보 찾기
# ========================================================
# runtime_log_candidates()는 44번 코드의 파일명 패턴과 맞는 CSV를 찾습니다.
# 우선 ip_webcam_w*_c*_s*_*.csv 형식의 실행 로그를 탐색합니다.

def runtime_log_candidates(
    input_dir: Path,
    output_path: Path,
):
    files = list(
        input_dir.glob(
            "ip_webcam_w*_c*_s*_*.csv"
        )
    )

    # ========================================================
    # ⑲ 기본 패턴의 로그가 없다면 넓게 검색
    # ========================================================
    # 첫 검색 결과가 없을 때만 ip_webcam_*.csv 전체를 검색합니다.
    # 첫 검색 결과가 있으면 분석 불가 파일만 있더라도 넓은 검색으로 넘어가지는 않습니다.
    #
    if not files:
        files = list(
            input_dir.glob(
                "ip_webcam_*.csv"
            )
        )

    # ========================================================
    # ⑳ 요약·비교 파일을 원본 후보에서 제외
    # ========================================================
    # 현재 출력 예정 CSV와 이름에 summary 또는 comparison이 있는 파일은 제외합니다.
    # 요약표를 원본 실시간 Frame 로그로 읽는 실수를 줄이기 위한 조건입니다.
    #
    cleaned = []

    for path in files:
        if (
            path.resolve()
            == output_path.resolve()
        ):
            continue

        lower_name = (
            path.name.lower()
        )

        if (
            "summary" in lower_name
            or "comparison" in lower_name
        ):
            continue

        cleaned.append(
            path
        )

    # ========================================================
    # ㉑ 최근에 수정한 로그부터 검사하도록 정렬
    # ========================================================
    # stat().st_mtime은 파일의 마지막 수정 시각입니다.
    # 수정 시각을 내림차순으로 정렬하고 같은 시각에는 파일명을 비교합니다.
    #
    return sorted(
        cleaned,
        key=lambda p: (
            p.stat().st_mtime,
            p.name,
        ),
        reverse=True,
    )


# ========================================================
# ㉒ 로그별 진단 행 수를 출력하는 함수
# ========================================================
# print_log_diagnostic()은 파일명과 4종류 행 수를 한 줄로 출력합니다.
# 자동 선택에서 건너뛴 파일이나 직접 지정한 실패 로그의 원인을 파악하는 데 사용합니다.

def print_log_diagnostic(
    path: Path,
    diagnostic: dict,
    prefix: str = "",
) -> None:
    print(
        f"{prefix}{path.name} | "
        f"rows={diagnostic['rows']} | "
        f"target_rows={diagnostic['target_rows']} | "
        f"prediction_rows={diagnostic['prediction_rows']} | "
        f"usable_rows={diagnostic['usable_rows']}"
    )


# ========================================================
# ㉓ 분석할 CSV를 결정하는 choose_log 함수
# ========================================================
# --log 지정 여부에 따라 직접 지정 모드와 자동 선택 모드를 분기합니다.
# 선택된 파일 경로, 전체 데이터, usable 데이터를 반환하는 것이 목적입니다.

def choose_log(
    value,
    input_dir: Path,
    output_path: Path,
):
    # ========================================================
    # ㉔ 직접 지정한 로그 파일이 존재하는지 검사
    # ========================================================
    # --log를 입력했다면 resolve_path()로 실제 경로를 확정합니다.
    # 파일이 없으면 FileNotFoundError를 발생시킵니다.
    #
    if value:
        path = resolve_path(
            value
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        # ========================================================
        # ㉕ 직접 지정 로그를 안전하게 읽고 오류 안내
        # ========================================================
        # read_runtime_log()를 실행하며 오류가 나면 파일명과 원인을 출력합니다.
        # SystemExit(1)로 분석을 중단합니다.
        #
        try:
            (
                df,
                usable,
                diagnostic,
            ) = read_runtime_log(
                path
            )
        except Exception as exc:
            print(
                f"[ERROR] Log를 읽을 수 없습니다: "
                f"{path}"
            )
            print(
                f"원인: {exc}"
            )
            raise SystemExit(1)

        # ========================================================
        # ㉖ 직접 지정 로그에 usable 행이 없으면 중단
        # ========================================================
        # Target과 실제 Prediction이 동시에 존재하지 않는다면 요약할 수 없습니다.
        # 44번 실습에서 창 클릭→1~6 Target 지정→Warm-up 이후 예측→Q 종료를 안내합니다.
        #
        if usable.empty:
            print()
            print(
                "[STOP] 지정한 Log에는 "
                "Target이 표시된 Prediction 결과가 없습니다."
            )
            print_log_diagnostic(
                path,
                diagnostic,
                prefix="  ",
            )
            print()
            print(
                "44번 실시간 실행에서 다음을 확인하세요."
            )
            print(
                "1) 카메라 창을 클릭하여 키 입력이 들어가게 합니다."
            )
            print(
                "2) 1~6 키를 눌러 Target을 선택합니다."
            )
            print(
                "3) 화면의 Target이 '-'가 아니라 Action 이름인지 확인합니다."
            )
            print(
                "4) WARMING_UP이 끝날 때까지 동작을 충분히 수행합니다."
            )
            print(
                "5) Prediction이 나온 뒤 Q로 종료합니다."
            )
            raise SystemExit(1)

        return (
            path,
            df,
            usable,
        )

    # ========================================================
    # ㉗ --log가 없으면 최신 Runtime 로그 자동 검색
    # ========================================================
    # runtime_log_candidates()로 정렬된 파일 목록을 가져옵니다.
    # 이후 분석 가능한 로그를 찾을 때까지 순서대로 검사합니다.
    #
    files = runtime_log_candidates(
        input_dir,
        output_path,
    )

    # ========================================================
    # ㉘ 자동 검색에서 후보 CSV 자체가 없으면 중단
    # ========================================================
    # 입력 폴더에 적절한 패턴의 파일이 없으면 폴더를 안내합니다.
    # SystemExit(1)로 종료하여 잘못된 빈 요약표를 만들지 않습니다.
    #
    if not files:
        print(
            "[STOP] IP Webcam Runtime Log가 없습니다."
        )
        print(
            f"확인 폴더: {input_dir}"
        )
        raise SystemExit(1)

    # ========================================================
    # ㉙ 건너뛴 후보 목록을 준비해 순회
    # ========================================================
    # skipped에는 분석 불가 파일과 그 진단 정보를 기록합니다.
    # for 반복문에서 최신 파일부터 하나씩 읽으며 조건을 확인합니다.
    #
    skipped = []

    # ========================================================
    # ㉚ 각 후보 로그에서 usable 데이터 읽기 시도
    # ========================================================
    # 각 파일을 read_runtime_log()로 읽습니다.
    # 읽기 성공 시 전체 df, usable, diagnostic을 받습니다.
    #
    for path in files:
        try:
            (
                df,
                usable,
                diagnostic,
            ) = read_runtime_log(
                path
            )
        # ========================================================
        # ㉛ 읽기 실패 후보는 건너뛰고 이유 기록
        # ========================================================
        # CSV 오류 또는 필수 컬럼 누락 등 예외가 발생할 수 있습니다.
        # 그 파일을 skipped에 기록하고 다음 파일로 넘어갑니다.
        #
        except Exception as exc:
            skipped.append(
                (
                    path,
                    {
                        "rows": 0,
                        "target_rows": 0,
                        "prediction_rows": 0,
                        "usable_rows": 0,
                    },
                    str(exc),
                )
            )
            continue

        # ========================================================
        # ㉜ 분석 가능한 최신 로그를 찾았는지 확인
        # ========================================================
        # usable이 비어 있지 않다면 이 로그를 최종 분석 대상으로 결정합니다.
        # 그보다 최신이지만 건너뛴 로그가 있으면 먼저 그 목록을 출력합니다.
        #
        if not usable.empty:
            if skipped:
                print(
                    "[INFO] 가장 최근 Log 중 일부는 "
                    "분석 조건을 만족하지 않아 건너뛰었습니다."
                )

                for (
                    skipped_path,
                    skipped_diag,
                    reason,
                ) in skipped:
                    print_log_diagnostic(
                        skipped_path,
                        skipped_diag,
                        prefix="  SKIP: ",
                    )

                    if reason:
                        print(
                            f"        reason={reason}"
                        )

                print()

            print(
                "[SELECTED] 가장 최근의 분석 가능한 Log"
            )
            print(
                f"  {path}"
            )

            # ========================================================
            # ㉝ 선택된 로그 반환 또는 현재 후보 건너뛰기
            # ========================================================
            # 분석 가능한 파일이라면 선택 경로, df, usable을 반환합니다.
            # 반대로 usable이 없는 후보는 skipped에 기록하고 다음 파일을 확인합니다.
            #
            return (
                path,
                df,
                usable,
            )

        skipped.append(
            (
                path,
                diagnostic,
                "",
            )
        )

    # ========================================================
    # ㉞ 모든 후보가 분석 불가할 때 안내하고 종료
    # ========================================================
    # 어느 CSV에도 Target이 있는 실제 Prediction이 없다면 STOP을 출력합니다.
    # 각 후보의 진단 수치와 44번 재실행 방법을 알려주고 종료합니다.
    #
    print()
    print(
        "[STOP] 분석 가능한 IP Webcam Runtime Log가 없습니다."
    )
    print(
        "현재 Log들을 확인했지만 "
        "'Target이 표시된 실제 Prediction'이 한 번도 없습니다."
    )
    print()

    for (
        skipped_path,
        skipped_diag,
        reason,
    ) in skipped:
        print_log_diagnostic(
            skipped_path,
            skipped_diag,
            prefix="  ",
        )

        if reason:
            print(
                f"      reason={reason}"
            )

    print()
    print(
        "다시 44번 실시간 실행을 진행하세요."
    )
    print(
        "1) 카메라 창을 클릭합니다."
    )
    print(
        "2) 1~6 키 중 하나를 눌러 Target을 선택합니다."
    )
    print(
        "3) 화면에서 Target이 Action 이름으로 바뀌었는지 확인합니다."
    )
    print(
        "4) WARMING_UP 이후 Prediction이 나오도록 "
        "동작을 충분히 수행합니다."
    )
    print(
        "5) Q로 종료하여 새 CSV를 저장합니다."
    )

    raise SystemExit(1)


# ========================================================
# ㉟ Action Label 변화 횟수를 계산하는 함수
# ========================================================
# label_changes()는 연속 예측에서 Label이 바뀐 횟수를 반환합니다.
# 이 함수는 Action별 segment_id 그룹 안에서 따로 호출됩니다.

def label_changes(
    series: pd.Series,
) -> int:
    # ========================================================
    # ㊱ Label의 결측값을 정리하고 문자형으로 통일
    # ========================================================
    # 결측값을 빈 문자열로 채우고 모두 문자열로 변환합니다.
    # 비교 방식이 달라지지 않도록 Label 타입을 통일합니다.
    #
    clean = (
        series
        .fillna("")
        .astype(str)
    )

    # ========================================================
    # ㊲ 이전 Label과 현재 Label 비교하기
    # ========================================================
    # 비어 있으면 0을 반환합니다.
    # shift(1)로 이전 예측을 가져온 뒤 달라진 횟수를 세고 첫 행 비교 1회를 뺍니다.
    #
    if clean.empty:
        return 0

    return int(
        (
            clean
            != clean.shift(1)
        ).sum()
        - 1
    )


# ========================================================
# ㊳ main 시작: 옵션과 경로 확정
# ========================================================
# parse_args()로 터미널 옵션을 읽습니다.
# 입력 폴더와 출력 CSV 경로를 resolve_path()로 변환합니다.

def main() -> None:
    args = parse_args()

    input_dir = resolve_path(
        args.input_dir
    )

    output_path = resolve_path(
        args.output
    )

    # ========================================================
    # ㊴ 최종 분석할 Runtime CSV 선택하기
    # ========================================================
    # choose_log()로 특정 로그 또는 최신 분석 가능 로그를 선택합니다.
    # 전체 df는 필요하지 않아 _로 받고, usable 데이터만 이후 분석에 사용합니다.
    #
    (
        log_path,
        _,
        usable,
    ) = choose_log(
        args.log,
        input_dir,
        output_path,
    )

    # ========================================================
    # ㊵ segment_id가 없는 이전 로그도 처리
    # ========================================================
    # 선택한 로그에 segment_id 컬럼이 없으면 모두 0으로 설정합니다.
    # 새 로그에서는 Action을 다시 선택할 때마다 구간 번호가 달라질 수 있습니다.
    #
    if "segment_id" not in usable.columns:
        usable[
            "segment_id"
        ] = 0

    # ========================================================
    # ㊶ Target Action별로 예측 결과 그룹화
    # ========================================================
    # expected_label을 기준으로 같은 Target의 usable 행을 묶습니다.
    # 여섯 Action 중 실제로 Target을 지정해 예측한 Action만 결과표에 나타납니다.
    #
    rows = []

    for label, group in (
        usable.groupby(
            "expected_label"
        )
    ):
        # ========================================================
        # ㊷ 같은 Action 안에서 수행 Segment 구분하기
        # ========================================================
        # 하나의 Action을 여러 차례 선택했을 수 있으므로 segment_id로 다시 그룹화합니다.
        # 다른 수행 구간의 마지막 Label과 첫 Label을 연결해 비교하지 않도록 합니다.
        #
        segment_groups = list(
            group.groupby(
                "segment_id"
            )
        )

        # ========================================================
        # ㊸ 구간별 Raw Label 변화 횟수 합산
        # ========================================================
        # 각 Segment의 raw_label 변경 횟수를 계산하고 sum()으로 합칩니다.
        # 안정화 전 예측이 얼마나 자주 바뀌었는지 나타냅니다.
        #
        raw_changes = sum(
            label_changes(
                segment[
                    "raw_label"
                ]
            )
            for _, segment
            in segment_groups
        )

        # ========================================================
        # ㊹ 구간별 Stable Label 변화 횟수 합산
        # ========================================================
        # 각 Segment의 stable_label 변경 횟수를 합칩니다.
        # Raw 변화 횟수와 비교해 안정화 효과를 관찰할 수 있습니다.
        #
        stable_changes = sum(
            label_changes(
                segment[
                    "stable_label"
                ]
            )
            for _, segment
            in segment_groups
        )

        # ========================================================
        # ㊺ 유효한 처리 FPS 값만 선택하기
        # ========================================================
        # processing_fps를 숫자로 변환하고 0보다 큰 값만 남깁니다.
        # 평균 FPS는 전체 Frame이 아니라 usable 예측 행에서 가져온 값이라는 점에 주의합니다.
        #
        fps_numeric = pd.to_numeric(
            group[
                "processing_fps"
            ],
            errors="coerce",
        )

        fps_values = fps_numeric[
            fps_numeric
            > 0
        ]

        # ========================================================
        # ㊻ Confidence도 숫자형으로 변환하기
        # ========================================================
        # confidence 값에 문자열이나 결측이 있어도 숫자로 변환하도록 합니다.
        # 숫자로 바꿀 수 없는 값은 NaN으로 처리되어 평균에서 제외됩니다.
        #
        confidence_numeric = (
            pd.to_numeric(
                group[
                    "confidence"
                ],
                errors="coerce",
            )
        )

        # ========================================================
        # ㊼ Action별 성능·안정화 지표 한 행 만들기
        # ========================================================
        # Action명, Segment 수, 예측 갱신 수, UNKNOWN 비율을 기록합니다.
        # Stable 정답 일치율과 Raw·Stable 변경 횟수, 평균 Confidence·FPS, 실행 옵션을 기록합니다.
        # stable_match_ratio는 키로 정한 Target과 Sliding Window 예측의 일치율이지 정식 Test Accuracy가 아닙니다.
        #
        rows.append(
            {
                "expected_label": (
                    label
                ),
                "segments": (
                    len(
                        segment_groups
                    )
                ),
                "prediction_updates": (
                    len(group)
                ),
                "unknown_ratio": round(
                    float(
                        (
                            group[
                                "threshold_label"
                            ]
                            == "UNKNOWN"
                        ).mean()
                    ),
                    4,
                ),
                "stable_match_ratio": round(
                    float(
                        (
                            group[
                                "stable_label"
                            ]
                            == label
                        ).mean()
                    ),
                    4,
                ),
                "raw_label_changes": (
                    raw_changes
                ),
                "stable_label_changes": (
                    stable_changes
                ),
                "mean_confidence": round(
                    float(
                        confidence_numeric.mean()
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
                "window_seconds": (
                    group[
                        "window_seconds"
                    ].iloc[0]
                    if "window_seconds"
                    in group.columns
                    else ""
                ),
                "confidence_threshold": (
                    group[
                        "confidence_threshold"
                    ].iloc[0]
                    if "confidence_threshold"
                    in group.columns
                    else ""
                ),
                "stabilization_buffer": (
                    group[
                        "stabilization_buffer"
                    ].iloc[0]
                    if "stabilization_buffer"
                    in group.columns
                    else ""
                ),
                "stabilization_min_votes": (
                    group[
                        "stabilization_min_votes"
                    ].iloc[0]
                    if "stabilization_min_votes"
                    in group.columns
                    else ""
                ),
            }
        )

    # ========================================================
    # ㊽ Action별 요약 행을 DataFrame으로 변환
    # ========================================================
    # rows 목록에는 실제 분석 가능한 Target Action만 들어 있습니다.
    # 각 Action이 한 행인 결과표를 생성합니다.
    #
    result = pd.DataFrame(
        rows
    )

    # ========================================================
    # ㊾ 출력 폴더 생성 및 요약 CSV 저장
    # ========================================================
    # 폴더가 없으면 상위 폴더까지 생성합니다.
    # index=False로 행 번호를 제외하고 utf-8-sig로 저장합니다.
    #
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # ㊿ 선택한 로그·결과표·주의사항 출력 및 실행
    # ========================================================
    # 선택된 CSV 경로와 Action별 요약표를 출력합니다.
    # Stable 일치율은 정식 Test Accuracy가 아니라는 점과 Segment별 변화 횟수 계산을 설명합니다.
    # 파일을 직접 실행하면 맨 아래에서 main()이 호출됩니다.
    #
    print()
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
        "주의: stable_match_ratio는 "
        "수동 Target 표시와 겹치는 "
        "Sliding Window를 이용한 "
        "Runtime 확인값이며 "
        "정식 Test Accuracy가 아닙니다."
    )
    print(
        "같은 Action을 여러 번 수행한 경우 "
        "segment_id별 Label 변화량을 먼저 "
        "계산한 뒤 합산합니다."
    )
    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()