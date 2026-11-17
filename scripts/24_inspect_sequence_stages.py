
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse : 터미널에서 실행 옵션을 입력받습니다.
# Path     : Sequence CSV 파일의 경로를 관리합니다.
# NumPy    : 대표 시점을 일정한 간격으로 선택합니다.
# Pandas   : CSV를 읽고 필요한 Feature를 출력합니다.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# ② 터미널 실행 옵션 설정
# ==================================================
# --source:
# 분석할 Sequence CSV의 경로를 입력받습니다.
#
# required=True:
# 반드시 입력해야 하는 필수 옵션입니다.
#
# --samples:
# Sequence에서 선택할 대표 Row 개수입니다.
#
# 기본값은 5개입니다.
#
# 예:
# --samples 5
# → 정상 Feature Row 중 5개를 선택합니다.
#
# --samples 9
# → 정상 Feature Row 중 9개를 선택합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "예: reports/day04/"
            "sequences/wave_sequence.csv"
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> None:

    # ==============================================
    # ③ 실행 옵션 읽기 및 입력 파일 경로 설정
    # ==============================================
    # parse_args()를 실행하여 터미널 옵션을 읽습니다.
    #
    # args.source:
    # 분석할 Sequence CSV 파일의 경로입니다.
    #
    # Path()를 이용하여 문자열 경로를
    # 파일 경로 객체로 변환합니다.

    args = parse_args()

    source = Path(
        args.source
    )

    # ==============================================
    # ④ 입력 파일 존재 여부와 Sample 개수 검사
    # ==============================================
    # Sequence CSV 파일이 실제로 존재하는지
    # 확인합니다.
    #
    # 파일이 없다면 FileNotFoundError를 발생시킵니다.
    #
    # 비교할 대표 Row는 최소 2개가 필요합니다.
    #
    # --samples 값이 2보다 작다면
    # ValueError를 발생시킵니다.
    #
    # 정상적인 입력이라면 CSV를 읽는 단계로 이동합니다.

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    if args.samples < 2:
        raise ValueError(
            "--samples는 "
            "2 이상이어야 합니다."
        )

    # ==============================================
    # ⑤ Sequence CSV 읽기 및 필수 컬럼 정의
    # ==============================================
    # Pandas의 read_csv()로 Sequence CSV를 읽습니다.
    #
    # 필수 컬럼은 다음과 같습니다.
    #
    # action_label:
    # 어떤 Action인지 나타내는 이름입니다.
    #
    # frame_id / time_sec:
    # Frame 번호와 영상에서의 시간입니다.
    #
    # feature_ok:
    # 38-D Feature 생성 성공 여부입니다.
    #
    # hip_center_x / y:
    # 화면에서 골반 중심의 위치입니다.
    #
    # left/right_wrist_x/y:
    # 왼쪽·오른쪽 손목의 정규화 좌표입니다.
    #
    # left/right_ankle_x/y:
    # 왼쪽·오른쪽 발목의 정규화 좌표입니다.
    #
    # elbow_angle:
    # 왼쪽·오른쪽 팔꿈치 각도입니다.
    #
    # knee_angle:
    # 왼쪽·오른쪽 무릎 각도입니다.
    #
    # 필요한 컬럼을 집합(set)으로 정의합니다.

    df = pd.read_csv(source)

    required_columns = {
        "action_label",
        "frame_id",
        "time_sec",
        "feature_ok",
        "hip_center_x",
        "hip_center_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "left_ankle_x",
        "left_ankle_y",
        "right_ankle_x",
        "right_ankle_y",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
    }

    # ==============================================
    # ⑥ CSV의 필수 컬럼 누락 여부 확인
    # ==============================================
    # df.columns:
    # CSV에 실제로 존재하는 컬럼 이름입니다.
    #
    # required_columns - set(df.columns):
    # 필요한 컬럼 중 실제 CSV에 없는 항목을 찾습니다.
    #
    # 누락된 컬럼이 있다면
    # RuntimeError를 발생시킵니다.
    #
    # 모든 필수 컬럼이 있다면 다음 단계로 이동합니다.

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Sequence CSV에 "
            "필수 열이 없습니다: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    # ==============================================
    # ⑦ 정상적으로 생성된 Feature Row만 선택
    # ==============================================
    # feature_ok=1은 해당 Frame에서
    # 38-D Feature 생성이 성공했다는 의미입니다.
    #
    # feature_ok=0인 Row는 비교 대상에서 제외합니다.
    #
    # .copy():
    # 선택한 데이터를 별도의 DataFrame으로 만듭니다.
    #
    # 정상 Feature Row가 2개 미만이면
    # 시간에 따른 비교가 어려우므로 오류를 발생시킵니다.
    #
    # 주의:
    # 실패한 Row를 원본 CSV에서 삭제하는 것은 아닙니다.
    # 이번 비교에서만 제외하는 것입니다.

    valid = df[
        df["feature_ok"] == 1
    ].copy()

    if len(valid) < 2:
        raise RuntimeError(
            "비교할 수 있는 정상 "
            "Feature Row가 2개 미만입니다."
        )

    # ==============================================
    # ⑧ 정상 Feature를 Frame 순서대로 정렬
    # ==============================================
    # sort_values("frame_id"):
    # Frame 번호를 기준으로 오름차순 정렬합니다.
    #
    # Sequence는 시간 순서가 중요하므로
    # Frame 순서대로 정렬한 후 비교합니다.
    #
    # reset_index(drop=True):
    # 정렬된 DataFrame의 행 인덱스를
    # 0부터 다시 부여합니다.
    #
    # 예:
    # Frame 0 → DataFrame 인덱스 0
    # Frame 1 → DataFrame 인덱스 1
    #
    # 단, 중간에 실패 Frame이 있다면
    # Frame 번호와 DataFrame 인덱스는
    # 서로 다를 수 있습니다.

    valid = valid.sort_values(
        "frame_id"
    ).reset_index(
        drop=True
    )

    # ==============================================
    # ⑨ 실제로 선택할 Sample 개수 계산
    # ==============================================
    # 사용자가 요청한 Sample 수와
    # 실제 정상 Feature Row 수를 비교합니다.
    #
    # min():
    # 두 값 중 작은 값을 선택합니다.
    #
    # 예:
    # 정상 Row 126개, 요청 Sample 5개
    # → sample_count = 5
    #
    # 정상 Row 3개, 요청 Sample 5개
    # → sample_count = 3
    #
    # 실제 데이터보다 많은 Row를 선택하지 않도록
    # 처리하는 코드입니다.

    sample_count = min(
        args.samples,
        len(valid),
    )

    # ==============================================
    # ⑩ 일정한 간격으로 대표 Row 위치 계산
    # ==============================================
    # np.linspace():
    # 시작 위치부터 마지막 위치까지
    # 일정한 간격의 숫자를 생성합니다.
    #
    # 시작 위치 = 0
    # 마지막 위치 = len(valid) - 1
    #
    # sample_count만큼의 위치를 계산합니다.
    #
    # round():
    # 계산한 위치를 가장 가까운 정수로 반올림합니다.
    #
    # astype(int):
    # 정수형 인덱스로 변환합니다.
    #
    # 예:
    # 정상 Feature Row 5개를 선택한다면
    #
    # 초반 → 초중반 → 중간 → 중후반 → 후반
    #
    # 위치를 일정한 간격으로 선택합니다.
    #
    # 주의:
    # 동작이 가장 크게 일어난 순간을
    # 자동으로 찾아주는 코드는 아닙니다.

    indices = np.linspace(
        0,
        len(valid) - 1,
        sample_count,
    ).round().astype(int)

    # ==============================================
    # ⑪ 계산한 위치에 해당하는 대표 Row 선택
    # ==============================================
    # iloc:
    # DataFrame의 행 위치 번호로 데이터를 선택합니다.
    #
    # 위에서 계산한 indices에 해당하는
    # 정상 Feature Row를 가져옵니다.
    #
    # selected에는 비교에 사용할 대표 Row만 남습니다.
    #
    # 예:
    # 정상 Row가 126개이고 samples=5라면
    # 대략 0, 31, 62, 94, 125 위치를 선택합니다.
    #
    # 여기서 indices는 정렬된 정상 Row의 위치이며
    # 원본 영상의 Frame 번호와 다를 수 있습니다.

    selected = valid.iloc[
        indices
    ].copy()

    # ==============================================
    # ⑫ 화면에 출력할 주요 Feature 컬럼 선택
    # ==============================================
    # Sequence CSV에는 38-D Feature 전체와
    # 다양한 Metadata가 저장되어 있습니다.
    #
    # 이 코드에서는 그중 움직임을 이해하는 데
    # 필요한 주요 값만 출력합니다.
    #
    # frame_id / time_sec:
    # 어느 시점의 데이터인지 확인합니다.
    #
    # hip_center:
    # 화면에서 골반 중심의 이동을 확인합니다.
    #
    # wrist:
    # 팔과 손의 위치 변화를 확인합니다.
    #
    # ankle:
    # 다리와 발의 위치 변화를 확인합니다.
    #
    # elbow_angle / knee_angle:
    # 팔꿈치와 무릎이 굽혀지는 정도를 확인합니다.
    #
    # 전체 Feature를 다시 계산하는 것이 아니라
    # 이미 저장된 값을 선택하여 보여줍니다.

    columns = [
        "frame_id",
        "time_sec",
        "hip_center_x",
        "hip_center_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "left_ankle_x",
        "left_ankle_y",
        "right_ankle_x",
        "right_ankle_y",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
    ]

    # ==============================================
    # ⑬ 입력 Sequence와 선택된 Sample 정보 출력
    # ==============================================
    # Source:
    # 분석에 사용한 CSV 파일명입니다.
    #
    # Action:
    # 해당 Sequence의 Action Label입니다.
    #
    # Valid feature rows:
    # feature_ok=1인 전체 Row 개수입니다.
    #
    # Selected samples:
    # 실제로 선택한 대표 Row 개수입니다.
    #
    # 이를 통해 어떤 Action에서 몇 개의
    # 정상 Feature를 비교하는지 확인합니다.

    print(
        f"Source: {source.name}"
    )

    print(
        f"Action: "
        f"{valid['action_label'].iloc[0]}"
    )

    print(
        f"Valid feature rows: "
        f"{len(valid)}"
    )

    print(
        f"Selected samples: "
        f"{sample_count}"
    )

    # ==============================================
    # ⑭ 대표 시점의 Feature를 표로 출력
    # ==============================================
    # selected[columns]:
    # 대표 Row에서 필요한 컬럼만 선택합니다.
    #
    # to_string(index=False):
    # DataFrame의 행 인덱스를 제외하고
    # 모든 선택 컬럼을 문자열 표로 출력합니다.
    #
    # 출력된 값을 통해 Action의 초반·중간·후반에서
    # 관절 위치와 각도가 어떻게 변했는지 비교합니다.
    #
    # 이 코드는 Action을 자동으로 분류하지 않고,
    # 사람이 Feature 변화를 해석하도록 돕습니다.

    print()

    print(
        selected[
            columns
        ].to_string(
            index=False
        )
    )


# ==================================================
# 프로그램 실행 시작
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.

if __name__ == "__main__":
    main()