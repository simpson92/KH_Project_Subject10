
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse : 터미널에서 입력한 실행 옵션 처리
# csv      : Action별 요약 결과를 CSV 파일로 저장
# sys      : 프로젝트 내부 모듈을 찾도록 경로 추가
# Path     : 입력·출력 파일 및 폴더 경로 관리
# NumPy    : 인접 Frame의 좌표 변화량 계산
# Pandas   : Sequence CSV 읽기와 통계 계산

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# ② 프로젝트 ROOT 설정 및 COCO 관절 이름 가져오기
# ==================================================
# 현재 스크립트의 상위 프로젝트 폴더를 ROOT로 지정합니다.
# ROOT를 Python 모듈 검색 경로에 추가한 뒤
# src.pose_utils에서 COCO 17개 관절 이름을 가져옵니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# ③ Sequence 입력 폴더와 요약 CSV 출력 경로 지정
# ==================================================
# DEFAULT_SEQUENCE_DIR:
# 22번 코드에서 만든 Sequence CSV 폴더입니다.
#
# DEFAULT_OUTPUT_PATH:
# 여러 Action을 한 표로 요약할 CSV 파일입니다.

DEFAULT_SEQUENCE_DIR = (
    ROOT
    / "reports"
    / "day04"
    / "sequences"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day04"
    / "tables"
    / "sequence_summary.csv"
)


# ==================================================
# ④ 터미널 실행 옵션 설정
# ==================================================
# --sequence-dir:
# Sequence CSV가 저장된 입력 폴더입니다.
#
# --output:
# 요약 CSV를 저장할 파일 경로입니다.
#
# 옵션을 입력하지 않으면 기본 경로를 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-dir",
        default=str(
            DEFAULT_SEQUENCE_DIR
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


# ==================================================
# ⑤ 34개 정규화 좌표 컬럼 이름 생성
# ==================================================
# COCO 관절 17개에 각각 _x, _y를 붙입니다.
#
# 17개 관절 × 2개 좌표 = 34개 컬럼입니다.
#
# 이번 움직임 변화량 계산에는 이 34개 좌표만
# 사용하고 나머지 4개 관절각도는 포함하지 않습니다.

def coordinate_columns() -> list[str]:
    columns = []

    for name in COCO_KEYPOINT_NAMES:
        columns.append(
            f"{name}_x"
        )

        columns.append(
            f"{name}_y"
        )

    return columns


# ==================================================
# ⑥ 인접 Row의 Feature 변화량 계산 준비
# ==================================================
# df:
# Action 하나의 Sequence 데이터입니다.
#
# columns:
# 34개 정규화 좌표 컬럼명입니다.
#
# Row가 2개 미만이면 비교할 두 시점이 없으므로
# 0.0을 반환합니다.
#
# 좌표 데이터를 숫자로 변환하며,
# 숫자로 바꿀 수 없는 값은 NaN으로 처리합니다.

def mean_adjacent_feature_change(
    df: pd.DataFrame,
    columns: list[str],
) -> float:
    if len(df) < 2:
        return 0.0

    numeric = (
        df[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # ==============================================
    # ⑦ 연속된 두 Row가 모두 정상 Feature인지 확인
    # ==============================================
    # feature_ok=1인 Row를 정상으로 판단합니다.
    #
    # shift(1):
    # 바로 이전 Row의 성공 여부를 가져옵니다.
    #
    # 현재 Row와 이전 Row가 모두 정상일 때
    # pair_ok=True가 됩니다.
    #
    # 실패한 Row를 건너뛰어 비교하지 않습니다.
    #
    # 여기서 인접 Row는 CSV에서 연속된
    # 두 개의 처리 Row를 의미합니다.

    feature_ok = (
        df["feature_ok"]
        == 1
    )

    pair_ok = (
        feature_ok
        & feature_ok.shift(
            1,
            fill_value=False,
        )
    )

    # ==============================================
    # ⑧ 두 Row의 좌표 차이 중 유효한 값만 선택
    # ==============================================
    # diff():
    # 현재 Row에서 바로 이전 Row의 값을 뺍니다.
    #
    # 34개 좌표별 차이를 계산합니다.
    #
    # pair_ok=True인 Row만 남깁니다.
    #
    # dropna():
    # 좌표 차이에 NaN이 포함된 Row를 제외합니다.
    #
    # 비교할 정상 Row 쌍이 없다면
    # 0.0을 반환합니다.

    differences = (
        numeric.diff()
    )

    valid_differences = (
        differences.loc[
            pair_ok
        ]
        .dropna()
    )

    if valid_differences.empty:
        return 0.0

    # ==============================================
    # ⑨ 좌표 변화의 거리 계산 후 전체 평균 반환
    # ==============================================
    # np.linalg.norm(..., axis=1):
    # 각 인접 Row의 34개 좌표 차이를
    # 하나의 거리값으로 계산합니다.
    #
    # np.mean():
    # 인접 Row마다 계산한 거리의 평균을 구합니다.
    #
    # 이 값은 자세 좌표 변화량을 살펴보는
    # 참고 지표이며 Action 인식 정확도가 아닙니다.

    distances = np.linalg.norm(
        valid_differences.to_numpy(
            dtype=float
        ),
        axis=1,
    )

    return float(
        np.mean(distances)
    )


def main() -> None:

    # ==============================================
    # ⑩ 실행 옵션 및 Sequence 입력 폴더 확인
    # ==============================================
    # 터미널 옵션을 읽습니다.
    #
    # Sequence CSV 입력 폴더를 지정합니다.
    # 요약 CSV 출력 경로를 지정합니다.
    #
    # 입력 폴더가 존재하지 않으면
    # FileNotFoundError를 발생시킵니다.

    args = parse_args()

    sequence_dir = Path(
        args.sequence_dir
    )

    output_path = Path(
        args.output
    )

    if not sequence_dir.exists():
        raise FileNotFoundError(
            sequence_dir
        )

    # ==============================================
    # ⑪ Sequence CSV 목록과 필수 컬럼 준비
    # ==============================================
    # *_sequence.csv 형식의 파일을 모두 찾습니다.
    #
    # sorted():
    # 파일명을 기준으로 정렬합니다.
    #
    # 입력 파일이 하나도 없다면 오류를 발생시킵니다.
    #
    # 34개 정규화 좌표 컬럼 이름을 가져옵니다.
    #
    # rows:
    # Action별 요약 결과를 저장할 목록입니다.
    #
    # required_columns:
    # 입력 CSV에 반드시 있어야 하는 컬럼 목록입니다.

    sequence_files = sorted(
        sequence_dir.glob(
            "*_sequence.csv"
        )
    )

    if not sequence_files:
        raise RuntimeError(
            "Sequence CSV가 없습니다."
        )

    coord_cols = (
        coordinate_columns()
    )

    rows = []

    required_columns = {
        "action_id",
        "action_label",
        "time_sec",
        "pose_ok",
        "feature_ok",
        "mean_keypoint_conf",
        "low_keypoint_count",
        *coord_cols,
    }

    # ==============================================
    # ⑫ Sequence CSV를 하나씩 읽고 형식 검사
    # ==============================================
    # Sequence CSV를 하나씩 읽습니다.
    #
    # 파일이 비어 있다면 [SKIP]을 출력하고
    # 다음 파일로 이동합니다.
    #
    # 필수 컬럼이 누락되어 있다면
    # RuntimeError를 발생시킵니다.
    #
    # 정상적인 CSV라면 통계 계산을 시작합니다.

    for path in sequence_files:
        df = pd.read_csv(path)

        if df.empty:
            print(
                f"[SKIP] 빈 CSV: "
                f"{path.name}"
            )
            continue

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise RuntimeError(
                f"{path.name}에 "
                "필수 열이 없습니다: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        # ==========================================
        # ⑬ Pose와 Feature 성공 수 및 성공률 계산
        # ==========================================
        # pose_ok=1인 Row를 선택합니다.
        #
        # feature_ok=1인 Row를 선택합니다.
        #
        # 전체 처리 Row 수를 계산합니다.
        # Pose 성공 Row 수를 계산합니다.
        # Feature 성공 Row 수를 계산합니다.
        #
        # 성공률 = 성공 Row 수 / 전체 Row 수
        #
        # 예:
        # 전체 100 Row 중 Pose 성공 95 Row
        # → pose_ok_ratio = 0.95

        valid_pose = df[
            df["pose_ok"] == 1
        ].copy()

        valid_feature = df[
            df["feature_ok"] == 1
        ].copy()

        total_rows = len(df)
        pose_rows = len(
            valid_pose
        )
        feature_rows = len(
            valid_feature
        )

        pose_ratio = (
            pose_rows / total_rows
            if total_rows > 0
            else 0.0
        )

        feature_ratio = (
            feature_rows / total_rows
            if total_rows > 0
            else 0.0
        )

        # ==========================================
        # ⑭ Keypoint Confidence와 저신뢰도 관절 수 평균
        # ==========================================
        # Pose 검출에 성공한 Row를 대상으로 계산합니다.
        #
        # mean_keypoint_conf:
        # Frame별 평균 관절 Confidence를
        # Sequence 전체에서 다시 평균낸 값입니다.
        #
        # mean_low_keypoint_count:
        # Frame별 저신뢰도 관절 개수를
        # Sequence 전체에서 평균낸 값입니다.
        #
        # Pose 성공 Row가 없다면 원본 코드 기준으로
        # 각각 0.0과 17.0을 사용합니다.
        #
        # 신뢰도 기준값은 앞의 Sequence 생성 단계에서
        # 정한 값을 사용하며 여기서는 다시 정하지 않습니다.

        mean_conf = (
            float(
                valid_pose[
                    "mean_keypoint_conf"
                ].mean()
            )
            if pose_rows > 0
            else 0.0
        )

        mean_low = (
            float(
                valid_pose[
                    "low_keypoint_count"
                ].mean()
            )
            if pose_rows > 0
            else 17.0
        )

        # ==========================================
        # ⑮ 인접 Row 변화량과 마지막 기록 시간 계산
        # ==========================================
        # 앞에서 만든 함수를 호출하여
        # 평균 좌표 변화량을 계산합니다.
        #
        # 34개의 정규화 좌표를 사용합니다.
        #
        # last_time:
        # CSV에 저장된 time_sec 중 가장 큰 값입니다.
        #
        # 이 값은 영상의 실제 전체 재생시간과
        # 반드시 일치하는 것은 아닙니다.

        movement = (
            mean_adjacent_feature_change(
                df,
                coord_cols,
            )
        )

        last_time = float(
            df["time_sec"].max()
        )

        # ==========================================
        # ⑯ Action 하나의 요약 결과를 한 행으로 저장
        # ==========================================
        # 지금까지 계산한 값을 하나의 딕셔너리로 만듭니다.
        #
        # 저장 항목:
        # Action ID, Action Label
        # 전체 Row 수
        # Pose 성공 수와 성공률
        # Feature 성공 수와 성공률
        # 평균 Confidence
        # 평균 저신뢰도 관절 수
        # 평균 좌표 변화량
        # 마지막 기록 시각
        #
        # Action 하나당 rows에 한 행이 추가됩니다.

        rows.append(
            {
                "action_id": (
                    df[
                        "action_id"
                    ].iloc[0]
                ),
                "action_label": (
                    df[
                        "action_label"
                    ].iloc[0]
                ),
                "rows": total_rows,
                "pose_ok_rows": pose_rows,
                "pose_ok_ratio": round(
                    pose_ratio,
                    4,
                ),
                "feature_ok_rows":
                    feature_rows,
                "feature_ok_ratio":
                    round(
                        feature_ratio,
                        4,
                    ),
                "mean_keypoint_conf":
                    round(
                        mean_conf,
                        4,
                    ),
                "mean_low_keypoint_count":
                    round(
                        mean_low,
                        3,
                    ),
                "mean_adjacent_feature_change":
                    round(
                        movement,
                        6,
                    ),
                "last_time_sec": round(
                    last_time,
                    3,
                ),
            }
        )

    # ==============================================
    # ⑰ Action별 요약 데이터를 CSV 파일로 저장
    # ==============================================
    # 모든 Sequence CSV의 처리가 끝나면 실행합니다.
    #
    # 요약할 데이터가 없다면 오류를 발생시킵니다.
    #
    # 출력 폴더가 없으면 생성합니다.
    #
    # sequence_summary.csv 파일을 만들고
    # 첫 번째 행에 컬럼명을 작성합니다.
    #
    # 각 Action의 요약 결과를 한 행씩 저장합니다.

    if not rows:
        raise RuntimeError(
            "요약할 수 있는 "
            "Sequence 데이터가 없습니다."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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
        writer.writerows(rows)

    # ==============================================
    # ⑱ CSV 저장 경로와 Action별 요약표 출력
    # ==============================================
    # 저장한 CSV의 경로를 터미널에 출력합니다.
    #
    # 같은 데이터를 Pandas DataFrame으로 변환합니다.
    #
    # to_string(index=False):
    # 행 번호 없이 전체 요약표를 출력합니다.
    #
    # 이를 통해 여러 Action을 한눈에 비교합니다.

    print(
        f"Saved: {output_path}"
    )

    print()

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False
        )
    )


# ==================================================
# ⑲ Python 파일 직접 실행 시 main() 시작
# ==================================================
# 이 파일을 직접 실행했을 때 main()을 호출합니다.

if __name__ == "__main__":
    main()