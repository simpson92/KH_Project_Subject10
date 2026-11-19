
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# csv   : 검사 결과를 CSV 파일로 저장합니다.
# Path  : 프로젝트의 파일·폴더 경로를 관리합니다.
# pandas: Manifest와 Keypoint CSV를 읽고 검사합니다.

import csv
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 입력·출력 경로 설정
# ==================================================
# ROOT:
# scripts 폴더의 상위인 프로젝트 루트입니다.
#
# APPROVED_PATH:
# 29번 코드에서 만든 승인 Clip 목록입니다.
#
# SPLIT_DIR:
# 30번 코드에서 만든 train.csv, val.csv,
# test.csv가 저장된 폴더입니다.
#
# OUTPUT_PATH:
# 6일차 입력 데이터 검사 결과를 저장할 CSV입니다.

ROOT = Path(__file__).resolve().parents[1]

APPROVED_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "approved_clip_manifest.csv"
)

SPLIT_DIR = (
    ROOT
    / "data"
    / "day05"
    / "splits"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "tables"
    / "day06_input_check.csv"
)


# ==================================================
# ③ Split별 예상 Subject와 Clip 개수 정의
# ==================================================
# 5일차에서 사람을 기준으로 데이터를 분리했습니다.
#
# Train      → S01 / 6 Clips
# Validation → S02 / 6 Clips
# Test       → S03 / 6 Clips
#
# EXPECTED에는 각 Split에서 반드시 만족해야 할
# Subject와 Clip 수를 기록합니다.
#
# 이후 실제 CSV 내용과 이 기준을 비교합니다.

EXPECTED = {
    "train": {
        "subject": "S01",
        "count": 6,
    },
    "val": {
        "subject": "S02",
        "count": 6,
    },
    "test": {
        "subject": "S03",
        "count": 6,
    },
}


# ==================================================
# ④ 각 Split에 필요한 6개 Action ID 정의
# ==================================================
# 이번 수업의 Action은 A001부터 A006까지입니다.
#
# 각 Subject는 6개 Action을 하나씩 수행했습니다.
#
# 따라서 Train, Validation, Test에는 각각
# A001~A006이 모두 포함되어 있어야 합니다.
#
# set 자료형을 사용하여 Action 목록을 정의합니다.

EXPECTED_ACTIONS = {
    "A001",
    "A002",
    "A003",
    "A004",
    "A005",
    "A006",
}


# ==================================================
# ⑤ main() 시작 및 Approved Manifest 존재 여부 확인
# ==================================================
# main()은 6일차 입력 데이터 전체를 검사합니다.
#
# 먼저 5일차 Approved Manifest가 있는지 확인합니다.
#
# 파일이 없다면 29번 결과가 준비되지 않은 것이므로
# FileNotFoundError를 발생시키고 중단합니다.

def main() -> None:
    if not APPROVED_PATH.exists():
        raise FileNotFoundError(
            APPROVED_PATH
        )

    # ==============================================
    # ⑥ Approved Manifest 읽기 및 18개 Clip 확인
    # ==============================================
    # pd.read_csv():
    # 승인된 Clip 목록을 DataFrame으로 읽습니다.
    #
    # len(approved):
    # 승인된 Clip 목록의 전체 Row 수입니다.
    #
    # 이번 6일차 기본 수업에서는
    # 정확히 18개 승인 Clip이 필요합니다.
    #
    # 18개가 아니라면 RuntimeError가 발생합니다.
    #
    # 주의:
    # 현재 코드는 승인 목록의 행 수를 확인하며,
    # 각 행의 Human QA 판정을 다시 검사하지 않습니다.

    approved = pd.read_csv(
        APPROVED_PATH
    )

    if len(approved) != 18:
        raise RuntimeError(
            "현재 6일차 기본 실습은 "
            "18개 PASS Clip을 사용합니다. "
            f"Approved Clip: {len(approved)}"
        )

    # ==============================================
    # ⑦ Split 데이터·검사 결과·오류 목록 준비
    # ==============================================
    # split_frames:
    # Train, Validation, Test의 DataFrame을 저장합니다.
    #
    # rows:
    # Clip별 입력 데이터 검사 결과를 모읍니다.
    #
    # problems:
    # 검사 과정에서 발견된 오류를 기록합니다.
    #
    # 오류를 발견해도 가능한 검사를 계속 수행한 후,
    # 마지막에 오류 목록을 출력합니다.

    split_frames = {}
    rows = []
    problems = []

    # ==============================================
    # ⑧ Train·Validation·Test CSV를 차례로 검사
    # ==============================================
    # EXPECTED.items()를 이용하여
    # train, val, test를 하나씩 가져옵니다.
    #
    # split_name:
    # 현재 검사할 Split 이름입니다.
    #
    # rule:
    # 해당 Split의 예상 Subject와 Clip 수입니다.
    #
    # 예:
    # split_name = "train"
    # rule = {"subject": "S01", "count": 6}
    #
    # Split CSV가 없다면 FileNotFoundError가 발생합니다.

    for split_name, rule in (
        EXPECTED.items()
    ):
        split_path = (
            SPLIT_DIR
            / f"{split_name}.csv"
        )

        if not split_path.exists():
            raise FileNotFoundError(
                split_path
            )

        # ==========================================
        # ⑨ 현재 Split CSV 읽기 및 DataFrame 보관
        # ==========================================
        # 현재 Split의 CSV를 Pandas로 읽습니다.
        #
        # 읽은 DataFrame을 split_frames에 저장합니다.
        #
        # 예:
        # split_frames["train"] = Train DataFrame
        #
        # 이 데이터는 나중에
        # Clip Leakage 검사에도 사용합니다.

        df = pd.read_csv(
            split_path
        )

        split_frames[
            split_name
        ] = df

        # ==========================================
        # ⑩ 현재 Split의 Clip 개수 확인
        # ==========================================
        # len(df)는 현재 Split의 전체 Clip 수입니다.
        #
        # EXPECTED에 지정한 count와 비교합니다.
        #
        # 예:
        # Train에 6개가 있어야 하는데
        # 실제로 5개라면 오류입니다.
        #
        # 문제를 problems에 기록합니다.

        if len(df) != rule["count"]:
            problems.append(
                f"{split_name}: "
                f"Clip 수 {len(df)}"
            )

        # ==========================================
        # ⑪ Split의 Subject가 올바른지 확인
        # ==========================================
        # subject 컬럼에서 Subject ID를 가져옵니다.
        #
        # astype(str):
        # 모든 값을 문자열로 변환합니다.
        #
        # set():
        # 중복 Subject를 제거합니다.
        #
        # 예:
        # train.csv에 S01만 있다면 {"S01"}입니다.
        #
        # Train에는 S01만,
        # Validation에는 S02만,
        # Test에는 S03만 있어야 합니다.
        #
        # 다르면 problems에 오류를 기록합니다.

        subjects = set(
            df["subject"]
            .astype(str)
        )

        if subjects != {
            rule["subject"]
        }:
            problems.append(
                f"{split_name}: "
                f"Subject {sorted(subjects)}"
            )

        # ==========================================
        # ⑫ 각 Split의 Action Coverage 확인
        # ==========================================
        # action_id 컬럼을 가져옵니다.
        #
        # 문자열로 변환하고 대문자로 통일합니다.
        #
        # set을 이용하여 중복을 제거합니다.
        #
        # EXPECTED_ACTIONS와 비교합니다.
        #
        # 예:
        # A001~A006 중 A004가 빠졌다면
        # Action Coverage 오류입니다.
        #
        # 이번 코드는 6개 Clip과 6개 Action을
        # 함께 검사하므로 정상이라면
        # 각 Split에 Action이 하나씩 존재합니다.

        actions = set(
            df["action_id"]
            .astype(str)
            .str.upper()
        )

        if actions != EXPECTED_ACTIONS:
            problems.append(
                f"{split_name}: "
                "Action Coverage 오류"
            )

        # ==========================================
        # ⑬ Clip별 Keypoint CSV 경로 확인
        # ==========================================
        # 현재 Split의 Clip을 하나씩 가져옵니다.
        #
        # df.iterrows():
        # DataFrame의 각 행을 반복하여 읽습니다.
        #
        # keypoint_csv:
        # 26번 코드에서 생성한 Keypoint CSV 경로입니다.
        #
        # ROOT와 CSV의 상대 경로를 연결하여
        # 실제 파일 경로를 만듭니다.

        for _, item in (
            df.iterrows()
        ):
            keypoint_path = (
                ROOT
                / str(
                    item[
                        "keypoint_csv"
                    ]
                )
            )

            # ======================================
            # ⑭ Keypoint CSV 존재 여부와 행 수 확인
            # ======================================
            # exists():
            # Keypoint CSV가 실제로 존재하는지 검사합니다.
            #
            # 존재한다면:
            # Pandas로 CSV를 읽고 Row 수를 계산합니다.
            #
            # 존재하지 않는다면:
            # row_count는 0으로 유지됩니다.
            #
            # 한 Row는 일반적으로 한 Frame의
            # Keypoint 정보를 의미합니다.
            #
            # 주의:
            # 이 단계는 행 수만 확인합니다.
            # 관절 좌표와 Confidence의 품질까지
            # 다시 계산하지는 않습니다.

            exists = (
                keypoint_path.exists()
            )

            row_count = 0

            if exists:
                keypoint_df = (
                    pd.read_csv(
                        keypoint_path
                    )
                )

                row_count = len(
                    keypoint_df
                )

            # ======================================
            # ⑮ 누락되거나 데이터가 없는 CSV 기록
            # ======================================
            # 파일이 존재하지 않거나
            # 데이터 Row가 0개라면 오류입니다.
            #
            # 어떤 Clip에서 문제가 발견되었는지
            # clip_id와 함께 problems에 기록합니다.
            #
            # 문제가 있어도 다음 Clip 검사를 계속합니다.

            if (
                not exists
                or row_count == 0
            ):
                problems.append(
                    "Keypoint CSV 오류: "
                    f"{item['clip_id']}"
                )

            # ======================================
            # ⑯ Clip별 검사 결과를 rows에 추가
            # ======================================
            # 각 Clip의 검사 결과를 딕셔너리로 만듭니다.
            #
            # split:
            # Train / Validation / Test 구분
            #
            # clip_id:
            # 개별 Clip ID
            #
            # subject / action_id / action_label:
            # 해당 Clip의 사람과 동작 정보
            #
            # keypoint_csv:
            # Keypoint CSV 경로
            #
            # exists:
            # 파일 존재 1 / 미존재 0
            #
            # rows:
            # 실제 Keypoint CSV 데이터 행 수
            #
            # 이 결과는 최종 CSV에 저장됩니다.

            rows.append(
                {
                    "split": (
                        split_name
                    ),
                    "clip_id": (
                        item["clip_id"]
                    ),
                    "subject": (
                        item["subject"]
                    ),
                    "action_id": (
                        item["action_id"]
                    ),
                    "action_label": (
                        item[
                            "action_label"
                        ]
                    ),
                    "keypoint_csv": (
                        item[
                            "keypoint_csv"
                        ]
                    ),
                    "exists": int(
                        exists
                    ),
                    "rows": (
                        row_count
                    ),
                }
            )

    # ==============================================
    # ⑰ Split별 Clip ID를 집합으로 생성
    # ==============================================
    # 세 Split의 검사가 끝나면
    # 각 Split의 clip_id를 집합으로 만듭니다.
    #
    # 예:
    # clip_sets["train"]
    # → S01의 6개 Clip ID
    #
    # clip_sets["val"]
    # → S02의 6개 Clip ID
    #
    # clip_sets["test"]
    # → S03의 6개 Clip ID
    #
    # 이후 서로 겹치는 Clip이 있는지 검사합니다.

    clip_sets = {
        name: set(
            df["clip_id"]
            .astype(str)
        )
        for name, df in (
            split_frames.items()
        )
    }

    # ==============================================
    # ⑱ Clip Leakage 검사에 사용할 Split 쌍 정의
    # ==============================================
    # 서로 다른 Split에 같은 Clip이 있으면
    # Data Leakage가 발생할 수 있습니다.
    #
    # 세 가지 조합을 검사합니다.
    #
    # Train / Validation
    # Train / Test
    # Validation / Test

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    # ==============================================
    # ⑲ Clip ID 교집합으로 Leakage 검사
    # ==============================================
    # 각 Split 쌍의 clip_id 집합을 비교합니다.
    #
    # &:
    # 두 집합에 공통으로 존재하는 값을 찾습니다.
    #
    # 예:
    # Train과 Test에 같은 Clip ID가 있다면
    # Clip Leakage에 해당합니다.
    #
    # 공통 Clip ID가 있으면 problems에 기록합니다.
    #
    # 주의:
    # 현재 코드는 서로 다른 Split 사이의 중복만
    # 검사하며, 한 Split 내부의 중복 행은
    # 별도로 검사하지 않습니다.

    for left, right in pairs:
        overlap = (
            clip_sets[left]
            & clip_sets[right]
        )

        if overlap:
            problems.append(
                f"{left}/{right} "
                "Clip Leakage: "
                f"{sorted(overlap)}"
            )

    # ==============================================
    # ⑳ 6일차 검사 결과를 저장할 폴더 생성
    # ==============================================
    # reports/day06/tables/ 폴더를 준비합니다.
    #
    # 폴더가 없다면 생성합니다.
    #
    # parents=True:
    # 필요한 상위 폴더도 함께 생성합니다.
    #
    # exist_ok=True:
    # 이미 폴더가 있어도 오류가 발생하지 않습니다.

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # ㉑ day06_input_check.csv 저장
    # ==============================================
    # 지금까지 모은 rows를 CSV로 저장합니다.
    #
    # csv.DictWriter:
    # 딕셔너리 목록을 CSV로 작성합니다.
    #
    # fieldnames:
    # 첫 번째 결과 행의 Key를 컬럼명으로 사용합니다.
    #
    # writeheader():
    # 컬럼명을 첫 번째 행에 작성합니다.
    #
    # writerows():
    # Clip별 검사 결과를 저장합니다.
    #
    # 중요:
    # 현재 코드는 오류를 최종 판정하기 전에
    # CSV 파일을 먼저 저장합니다.
    #
    # 따라서 CSV가 생성되었다고 해서
    # 모든 검사를 통과한 것은 아닙니다.

    with OUTPUT_PATH.open(
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
    # ㉒ 누적된 오류 출력 및 실패 처리
    # ==============================================
    # problems 목록에 오류가 있는지 확인합니다.
    #
    # 오류가 하나라도 있다면
    # [ERROR] 메시지로 모두 출력합니다.
    #
    # 가능한 문제:
    # Clip 개수 부족 또는 초과
    # 잘못된 Subject
    # Action Coverage 오류
    # Keypoint CSV 누락 또는 빈 데이터
    # Clip Leakage
    #
    # 오류를 출력한 후 RuntimeError로 중단합니다.
    #
    # 오류가 없다면 마지막 READY 단계로 이동합니다.

    if problems:
        for problem in problems:
            print(
                "[ERROR]",
                problem,
            )

        raise RuntimeError(
            "6일차 입력 데이터에 "
            "문제가 있습니다."
        )

    # ==============================================
    # ㉓ 모든 검사 통과 시 READY 메시지 출력
    # ==============================================
    # 오류 없이 모든 검사를 통과하면
    # 입력 데이터 구성을 출력합니다.
    #
    # Train      → S01 / 6 Clips
    # Validation → S02 / 6 Clips
    # Test       → S03 / 6 Clips
    #
    # 또한 다음 검사 결과를 표시합니다.
    #
    # Keypoint CSV 존재와 데이터 행 수 확인
    # Action Coverage 확인
    # Clip Leakage 없음
    #
    # Day06 input data: READY가 출력되면
    # 다음 34번 Sequence Dataset 생성 실습으로
    # 진행할 수 있습니다.

    print(
        "Train      : S01 / 6 Clips"
    )
    print(
        "Validation : S02 / 6 Clips"
    )
    print(
        "Test       : S03 / 6 Clips"
    )
    print()
    print(
        "All Keypoint CSV: OK"
    )
    print(
        "Action Coverage  : OK"
    )
    print(
        "Clip Leakage     : NONE"
    )
    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print(
        "Day06 input data: READY"
    )


# ==================================================
# ㉔ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# 실제 데이터 검사 작업은 main()에서 시작됩니다.

if __name__ == "__main__":
    main()