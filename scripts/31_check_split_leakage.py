
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse : 터미널에서 실행 옵션을 입력받습니다.
# Path     : CSV 파일 경로를 관리합니다.
# pandas   : Split CSV를 읽고 목록을 비교합니다.

import argparse
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT 및 기본 검사 경로 설정
# ==================================================
# 현재 파일은 scripts 폴더 안에 있다고 가정합니다.
# parents[1]은 프로젝트의 ROOT 폴더를 가리킵니다.
#
# DEFAULT_SPLIT_DIR:
# 30번 코드에서 저장한 train.csv, val.csv,
# test.csv가 있는 폴더입니다.
#
# DEFAULT_EXPECTED_ACTIONS:
# 세 Split에 모두 있어야 하는 6개 Action ID입니다.

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPLIT_DIR = (
    ROOT
    / "data"
    / "day05"
    / "splits"
)

DEFAULT_EXPECTED_ACTIONS = (
    "A001,A002,A003,A004,A005,A006"
)


# ==================================================
# ③ 터미널 실행 옵션 설정
# ==================================================
# --split-dir:
# 검사할 세 CSV가 있는 폴더입니다.
#
# --expected-actions:
# 각 Split에 반드시 있어야 할 Action ID 목록입니다.
#
# 옵션을 생략하면 위에서 정한 기본값을 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        default=str(
            DEFAULT_SPLIT_DIR
        ),
    )

    parser.add_argument(
        "--expected-actions",
        default=DEFAULT_EXPECTED_ACTIONS,
    )

    return parser.parse_args()


# ==================================================
# ④ 필수 Action 문자열을 집합(set)으로 변환
# ==================================================
# "A001,A002,..."를 쉼표로 나눕니다.
#
# strip():
# 각 항목의 앞뒤 공백을 제거합니다.
#
# upper():
# 영문을 대문자로 통일합니다.
#
# set:
# 중복된 Action을 제거합니다.
#
# 아무 Action도 남지 않으면 ValueError가 발생합니다.

def parse_expected_actions(
    value: str,
) -> set[str]:
    actions = {
        item.strip().upper()
        for item in value.split(",")
        if item.strip()
    }

    if not actions:
        raise ValueError(
            "--expected-actions가 "
            "비어 있습니다."
        )

    return actions


# ==================================================
# ⑤ Split CSV 하나를 읽는 공통 함수
# ==================================================
# load_split()은 CSV 파일 하나를 읽습니다.
#
# split_dir:
# CSV들이 저장된 폴더입니다.
#
# name:
# train, val, test 중 하나입니다.
#
# 예:
# name = "train"
# → train.csv 경로 생성
#
# 파일이 없으면 FileNotFoundError를 발생시킵니다.
#
# 파일이 있다면 Pandas DataFrame으로 읽어 반환합니다.

def load_split(
    split_dir: Path,
    name: str,
) -> pd.DataFrame:
    path = (
        split_dir
        / f"{name}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


# ==================================================
# ⑥ main() 시작: 옵션·폴더·필수 Action 준비
# ==================================================
# main()은 실제 Leakage 검사를 진행하는 함수입니다.
#
# parse_args()로 실행 옵션을 읽습니다.
#
# split_dir:
# 세 개의 Split CSV가 저장된 폴더입니다.
#
# parse_expected_actions():
# 쉼표로 작성된 Action ID를 집합으로 바꿉니다.

def main() -> None:
    args = parse_args()

    split_dir = Path(
        args.split_dir
    )

    expected_actions = (
        parse_expected_actions(
            args.expected_actions
        )
    )

    # ==============================================
    # ⑦ Train·Validation·Test CSV 읽기
    # ==============================================
    # ⑤의 load_split()을 세 번 호출합니다.
    #
    # train.csv → train DataFrame
    # val.csv   → val DataFrame
    # test.csv  → test DataFrame
    #
    # 세 파일 중 하나라도 없으면
    # FileNotFoundError가 발생합니다.
    #
    # 이 파일들은 30번 코드에서 생성한 결과입니다.

    train = load_split(
        split_dir,
        "train",
    )
    val = load_split(
        split_dir,
        "val",
    )
    test = load_split(
        split_dir,
        "test",
    )

    # ==============================================
    # ⑧ 세 Split을 딕셔너리로 묶고 오류 목록 준비
    # ==============================================
    # datasets:
    # 각 Split 이름과 DataFrame을 연결합니다.
    #
    # errors:
    # 검사 중 발견한 문제를 저장할 빈 목록입니다.
    #
    # 여러 검사를 진행하면서 오류를 모아 두었다가
    # 마지막에 한꺼번에 출력합니다.

    datasets = {
        "TRAIN": train,
        "VALIDATION": val,
        "TEST": test,
    }

    errors = []

    # ==============================================
    # ⑨ 각 Split의 필수 Action 누락 여부 검사
    # ==============================================
    # datasets에 있는 세 Split을 하나씩 검사합니다.
    #
    # action_id:
    # 각 Clip의 Action ID입니다.
    #
    # astype(str):
    # Action ID를 문자열로 변환합니다.
    #
    # str.upper():
    # 영문을 대문자로 통일합니다.
    #
    # set():
    # 중복된 Action ID를 제거합니다.
    #
    # missing = expected_actions - actual_actions
    #
    # 필수 Action 중 실제 데이터에 없는 Action을 찾습니다.
    #
    # 누락된 Action이 있다면 errors에 기록합니다.
    #
    # 주의:
    # 이 코드는 Action이 최소 하나 존재하는지만
    # 검사하며, 각 Action의 Clip 수는 확인하지 않습니다.

    for name, df in (
        datasets.items()
    ):
        actual_actions = set(
            df["action_id"]
            .astype(str)
            .str.upper()
        )

        missing = (
            expected_actions
            - actual_actions
        )

        if missing:
            errors.append(
                f"{name} Action 누락: "
                + ", ".join(
                    sorted(missing)
                )
            )

    # ==============================================
    # ⑩ Split별 Subject 집합 생성
    # ==============================================
    # Train, Validation, Test에서
    # Subject ID를 각각 가져옵니다.
    #
    # set():
    # 같은 Subject ID의 중복을 제거합니다.
    #
    # 예:
    # train_subjects = {"S01"}
    # val_subjects   = {"S02"}
    # test_subjects  = {"S03"}
    #
    # 이렇게 만든 집합을 서로 비교하여
    # 같은 사람이 다른 Split에 있는지 확인합니다.

    train_subjects = set(
        train["subject"].astype(str)
    )
    val_subjects = set(
        val["subject"].astype(str)
    )
    test_subjects = set(
        test["subject"].astype(str)
    )

    # ==============================================
    # ⑪ Subject 교집합으로 중복 배정 여부 계산
    # ==============================================
    # & 기호:
    # 두 집합에 공통으로 존재하는 값을 찾습니다.
    #
    # 비교 대상:
    #
    # 1. Train / Validation
    # 2. Train / Test
    # 3. Validation / Test
    #
    # 예:
    # Train Subject      = {"S01", "S02"}
    # Validation Subject = {"S02"}
    #
    # 교집합 = {"S02"}
    #
    # 같은 Subject가 두 Split에 존재하므로
    # Subject Leakage에 해당합니다.

    subject_pairs = [
        (
            "Train/Validation",
            train_subjects
            & val_subjects,
        ),
        (
            "Train/Test",
            train_subjects
            & test_subjects,
        ),
        (
            "Validation/Test",
            val_subjects
            & test_subjects,
        ),
    ]

    # ==============================================
    # ⑫ Subject Leakage를 오류 목록에 기록
    # ==============================================
    # subject_pairs를 하나씩 확인합니다.
    #
    # label:
    # 비교한 Split 쌍의 이름입니다.
    #
    # overlap:
    # 두 Split에 공통으로 포함된 Subject 집합입니다.
    #
    # overlap에 값이 있다면 Subject Leakage입니다.
    #
    # 어떤 Split 쌍에서 어떤 Subject가 겹쳤는지
    # errors에 기록합니다.

    for label, overlap in (
        subject_pairs
    ):
        if overlap:
            errors.append(
                f"{label} Subject Leakage: "
                f"{sorted(overlap)}"
            )

    # ==============================================
    # ⑬ Split별 Clip ID 집합 생성
    # ==============================================
    # 이번에는 Subject가 아닌 Clip ID를 비교합니다.
    #
    # clip_id:
    # 개별 동영상 Clip을 구분하는 ID입니다.
    #
    # 예:
    # S01_A001_T01
    # S01_A002_T01
    #
    # 세 Split의 clip_id를 집합으로 만듭니다.
    #
    # 주의:
    # set은 같은 Split 내부의 중복 ID도 제거하므로
    # 이 코드는 한 CSV 내부의 중복 행을
    # 별도로 찾아내지는 않습니다.

    train_clips = set(
        train["clip_id"].astype(str)
    )
    val_clips = set(
        val["clip_id"].astype(str)
    )
    test_clips = set(
        test["clip_id"].astype(str)
    )

    # ==============================================
    # ⑭ Clip 교집합으로 중복 배정 여부 계산
    # ==============================================
    # Clip ID를 기준으로 세 Split을 비교합니다.
    #
    # 1. Train / Validation
    # 2. Train / Test
    # 3. Validation / Test
    #
    # 같은 clip_id가 둘 이상의 Split에 존재한다면
    # Clip Leakage가 발생한 것입니다.
    #
    # 교집합 연산으로 중복 Clip ID를 찾습니다.

    clip_pairs = [
        (
            "Train/Validation",
            train_clips
            & val_clips,
        ),
        (
            "Train/Test",
            train_clips
            & test_clips,
        ),
        (
            "Validation/Test",
            val_clips
            & test_clips,
        ),
    ]

    # ==============================================
    # ⑮ Clip Leakage를 오류 목록에 기록
    # ==============================================
    # 각 Split 쌍의 Clip 교집합을 확인합니다.
    #
    # overlap에 Clip ID가 존재한다면
    # errors에 해당 내용을 추가합니다.
    #
    # 예:
    # Train/Test Clip Leakage:
    # ['S01_A001_T01']
    #
    # 모든 문제는 마지막 ⑰ 단계에서 출력합니다.

    for label, overlap in clip_pairs:
        if overlap:
            errors.append(
                f"{label} Clip Leakage: "
                f"{sorted(overlap)}"
            )

    # ==============================================
    # ⑯ 각 Split에 배정된 Subject 목록 출력
    # ==============================================
    # Train에 포함된 Subject를 출력합니다.
    #
    # Validation에 포함된 Subject를 출력합니다.
    #
    # Test에 포함된 Subject를 출력합니다.
    #
    # sorted():
    # Subject ID를 정렬하여 보기 쉽게 출력합니다.
    #
    # 이를 통해 학생들이 실제 Split 구성을 확인합니다.

    print(
        "Train Subjects:",
        sorted(train_subjects),
    )
    print(
        "Validation Subjects:",
        sorted(val_subjects),
    )
    print(
        "Test Subjects:",
        sorted(test_subjects),
    )
    print()

    # ==============================================
    # ⑰ 누적된 오류 출력 및 실패 처리
    # ==============================================
    # errors에 문제가 하나라도 있는지 검사합니다.
    #
    # 기록된 오류가 있다면
    # [FAIL] 메시지로 모두 출력합니다.
    #
    # 가능한 오류:
    # Action 누락
    # Subject Leakage
    # Clip Leakage
    #
    # 오류 출력이 끝나면 RuntimeError를 발생시켜
    # 프로그램을 중단합니다.
    #
    # 오류가 없다면 마지막 PASS 단계로 이동합니다.

    if errors:
        for error in errors:
            print("[FAIL]", error)

        raise RuntimeError(
            "Split 검사에서 문제가 "
            "발견되었습니다."
        )

    # ==============================================
    # ⑱ 검사 통과 메시지 출력
    # ==============================================
    # errors가 비어 있다면 다음 조건을 만족한 것입니다.
    #
    # 1. 서로 다른 Split에 같은 Subject가 없음
    # 2. 서로 다른 Split에 같은 Clip ID가 없음
    # 3. 각 Split에 필요한 Action이 모두 존재함
    #
    # 세 가지 검사 결과를 출력합니다.
    #
    # 최종 메시지:
    # Leakage check: OK
    #
    # 이 메시지는 이번 코드가 검사한 조건을
    # 통과했다는 의미이며, 모든 데이터의
    # 품질과 무결성을 보장하는 것은 아닙니다.

    print(
        "[PASS] Subject overlap 없음"
    )
    print(
        "[PASS] Clip overlap 없음"
    )
    print(
        "[PASS] 필요한 Action Coverage 확인"
    )
    print()
    print("Leakage check: OK")


# ==================================================
# ⑲ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# 앞에서 정의한 함수들은 main()이 실행되면서
# 필요한 순서에 따라 호출됩니다.

if __name__ == "__main__":
    main()