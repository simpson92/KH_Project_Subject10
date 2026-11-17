from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# csv  : Action 설정 CSV 파일을 읽습니다.
# Path : 설정 파일과 동영상 파일의 경로를 관리합니다.

import csv
from pathlib import Path


# ==================================================
# ② 프로젝트 경로와 입력 파일 위치 설정
# ==================================================
# ROOT:
# 현재 Python 파일의 상위 프로젝트 폴더입니다.
#
# ACTION_FILE:
# 6개 Action의 ID, Label, MP4 파일명 등이
# 기록된 CSV 설정 파일입니다.
#
# VIDEO_DIR:
# 실제 6개 MP4 파일이 저장된 폴더입니다.
#
# 이후 설정 파일과 실제 영상을 비교합니다.

ROOT = Path(__file__).resolve().parents[1]

ACTION_FILE = (
    ROOT
    / "configs"
    / "action_classes.csv"
)

VIDEO_DIR = (
    ROOT
    / "data"
    / "day04"
    / "videos"
)


# ==================================================
# ③ 정상적인 Action ID, Label, MP4 기준 정의
# ==================================================
# EXPECTED_ACTIONS는 반드시 등록되어야 하는
# 6개 Action의 기준 정보입니다.
#
# 딕셔너리 Key:
# Action ID (A001 ~ A006)
#
# 딕셔너리 Value:
# (Action Label, MP4 파일명)
#
# 예:
# A001 → bend_return → bend_return.mp4
#
# CSV 파일에 작성된 정보가 이 기준과 일치하는지
# 나중에 비교합니다.

EXPECTED_ACTIONS = {
    "A001": (
        "bend_return",
        "bend_return.mp4",
    ),
    "A002": (
        "leg_raise_lower",
        "leg_raise_lower.mp4",
    ),
    "A003": (
        "walk_turn_walk",
        "walk_turn_walk.mp4",
    ),
    "A004": (
        "drink_return",
        "drink_return.mp4",
    ),
    "A005": (
        "sit_stand",
        "sit_stand.mp4",
    ),
    "A006": (
        "wave",
        "wave.mp4",
    ),
}


# ==================================================
# ④ 중복된 값을 찾는 함수 작성
# ==================================================
# find_duplicates()는 문자열 목록을 전달받아
# 두 번 이상 등장하는 값을 찾는 함수입니다.
#
# values.count(value):
# 해당 값이 목록에 몇 번 등장하는지 계산합니다.
#
# count가 1보다 크면 중복된 값입니다.
#
# set을 사용하므로 중복값 자체는
# 결과에 한 번만 저장됩니다.
#
# 예:
# ["A001", "A002", "A001"]
# → {"A001"}
#
# 이 함수는 Action ID, Label, MP4 파일명의
# 중복 여부를 확인할 때 사용합니다.

def find_duplicates(
    values: list[str],
) -> set[str]:
    return {
        value
        for value in values
        if values.count(value) > 1
    }


def main() -> None:

    # ==============================================
    # ⑤ Action 설정 CSV 파일 존재 여부 확인
    # ==============================================
    # action_classes.csv 파일이 실제로 있는지
    # 확인합니다.
    #
    # 파일이 없다면 FileNotFoundError를
    # 발생시키고 프로그램을 중단합니다.
    #
    # 파일이 있다면 CSV를 읽는 단계로 이동합니다.

    if not ACTION_FILE.exists():
        raise FileNotFoundError(
            ACTION_FILE
        )

    # ==============================================
    # ⑥ CSV 파일 읽기 및 필수 컬럼 확인
    # ==============================================
    # csv.DictReader를 이용하여 CSV를 읽습니다.
    #
    # utf-8-sig:
    # Excel 등에서 저장된 UTF-8 BOM도 처리합니다.
    #
    # 필수 컬럼은 다음 5개입니다.
    #
    # action_id     : 동작 고유 번호
    # action_label  : 동작 이름
    # type          : 동작 유형
    # video_file    : 연결된 MP4 파일명
    # description   : 동작 설명
    #
    # 실제 CSV 컬럼과 필수 컬럼을 비교합니다.
    #
    # 필수 컬럼이 빠졌다면 오류를 발생시킵니다.
    # 문제가 없다면 CSV의 모든 행을 읽습니다.

    with ACTION_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "action_id",
            "action_label",
            "type",
            "video_file",
            "description",
        }

        actual_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns
            - actual_columns
        )

        if missing_columns:
            raise RuntimeError(
                "필수 열이 없습니다: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        rows = list(reader)

    # ==============================================
    # ⑦ Action 데이터 존재 여부와 개수 확인
    # ==============================================
    # CSV에 실제 Action 데이터가 있는지 검사합니다.
    #
    # 한 줄도 없다면 오류를 발생시킵니다.
    #
    # problems:
    # 검사 과정에서 발견한 문제를 저장할 목록입니다.
    #
    # 이번 수업의 Action은 정확히 6개여야 합니다.
    #
    # 등록된 행이 6개가 아니라면
    # problems에 오류 내용을 추가합니다.
    #
    # 이 단계에서는 바로 종료하지 않고
    # 나머지 문제도 계속 확인합니다.

    if not rows:
        raise RuntimeError(
            "Action 정의가 없습니다."
        )

    problems = []

    if len(rows) != len(
        EXPECTED_ACTIONS
    ):
        problems.append(
            "Action 개수는 6개여야 합니다. "
            f"현재 {len(rows)}개입니다."
        )

    # ==============================================
    # ⑧ ID, Label, MP4 이름의 중복 검사
    # ==============================================
    # CSV에서 다음 세 가지 정보를 분리합니다.
    #
    # ids    : Action ID 목록
    # labels : Action Label 목록
    # files  : MP4 파일명 목록
    #
    # strip():
    # 문자열 앞뒤의 불필요한 공백을 제거합니다.
    #
    # find_duplicates() 함수를 호출하여
    # 세 목록에서 중복된 값을 찾습니다.
    #
    # 중복이 발견되면 problems에 기록합니다.
    #
    # 예:
    # A001이 두 번 등록되었다면
    # "중복 Action ID: A001"을 기록합니다.

    ids = [
        row["action_id"].strip()
        for row in rows
    ]

    labels = [
        row["action_label"].strip()
        for row in rows
    ]

    files = [
        row["video_file"].strip()
        for row in rows
    ]

    duplicate_ids = (
        find_duplicates(ids)
    )

    duplicate_labels = (
        find_duplicates(labels)
    )

    duplicate_files = (
        find_duplicates(files)
    )

    if duplicate_ids:
        problems.append(
            "중복 Action ID: "
            + ", ".join(
                sorted(duplicate_ids)
            )
        )

    if duplicate_labels:
        problems.append(
            "중복 Action Label: "
            + ", ".join(
                sorted(
                    duplicate_labels
                )
            )
        )

    if duplicate_files:
        problems.append(
            "중복 video_file: "
            + ", ".join(
                sorted(duplicate_files)
            )
        )

    # ==============================================
    # ⑨ 각 Action 정보와 MP4 존재 여부 출력
    # ==============================================
    # CSV에 등록된 Action 개수를 출력합니다.
    #
    # 이후 CSV의 각 행을 하나씩 검사합니다.
    #
    # Action ID
    # Action Label
    # MP4 파일명
    # Action Type
    #
    # 위 정보를 가져와 실제 영상 경로를 만듭니다.
    #
    # exists()로 MP4 파일 존재 여부를 확인합니다.
    #
    # 존재하면 True, 없으면 False입니다.
    #
    # 검사 결과를 한 줄씩 화면에 출력합니다.

    print(
        f"Action count: {len(rows)}"
    )

    print()

    for row in rows:
        action_id = (
            row["action_id"]
            .strip()
        )

        action_label = (
            row["action_label"]
            .strip()
        )

        video_file = (
            row["video_file"]
            .strip()
        )

        action_type = (
            row["type"]
            .strip()
        )

        video_path = (
            VIDEO_DIR
            / video_file
        )

        exists = (
            video_path.exists()
        )

        print(
            f"{action_id} "
            f"{action_label:18s} "
            f"file="
            f"{video_file:24s} "
            f"exists={exists}"
        )

        # ==========================================
        # ⑩ Action ID와 정답 Label 및 파일명 비교
        # ==========================================
        # EXPECTED_ACTIONS에서 현재 Action ID에
        # 해당하는 정답 정보를 가져옵니다.
        #
        # 등록되지 않은 Action ID라면
        # problems에 오류를 추가합니다.
        #
        # 정상적인 ID라면 다음 두 가지를 비교합니다.
        #
        # 1. Action Label이 기준과 같은가?
        # 2. MP4 파일명이 기준과 같은가?
        #
        # 하나라도 다르면 problems에 기록합니다.
        #
        # 예:
        # A001의 정답 Label은 bend_return입니다.
        #
        # CSV에 wave로 적었다면 오류입니다.

        expected = (
            EXPECTED_ACTIONS.get(
                action_id
            )
        )

        if expected is None:
            problems.append(
                "알 수 없는 Action ID: "
                f"{action_id}"
            )
        else:
            expected_label, (
                expected_file
            ) = expected

            if action_label != (
                expected_label
            ):
                problems.append(
                    f"{action_id}의 Label은 "
                    f"{expected_label}이어야 합니다."
                )

            if video_file != (
                expected_file
            ):
                problems.append(
                    f"{action_id}의 video_file은 "
                    f"{expected_file}이어야 합니다."
                )

        # ==========================================
        # ⑪ Action Type과 실제 동영상 존재 검사
        # ==========================================
        # 4일차의 6개 Action은 모두 dynamic입니다.
        #
        # dynamic:
        # 여러 Frame에 걸쳐 자세가 변하는 동작
        #
        # type이 dynamic이 아니면 오류를 기록합니다.
        #
        # 또한 실제 동영상 파일이 없다면
        # problems에 파일명을 기록합니다.
        #
        # 이 검사는 CSV의 각 Action에 반복됩니다.

        if action_type != "dynamic":
            problems.append(
                f"{action_id}의 type은 "
                "dynamic이어야 합니다."
            )

        if not exists:
            problems.append(
                "없는 동영상: "
                f"{video_file}"
            )

    # ==============================================
    # ⑫ 필수 Action ID 누락 여부 확인
    # ==============================================
    # EXPECTED_ACTIONS에서 반드시 필요한
    # A001~A006 ID 목록을 가져옵니다.
    #
    # 실제 CSV에 등록된 ID와 비교합니다.
    #
    # 집합의 차집합(-) 연산을 이용하여
    # 등록되지 않은 ID를 찾습니다.
    #
    # 예:
    # A001~A005만 존재한다면
    # 누락된 ID는 A006입니다.
    #
    # 누락된 ID가 있으면 problems에 기록합니다.

    expected_ids = set(
        EXPECTED_ACTIONS.keys()
    )

    missing_ids = (
        expected_ids
        - set(ids)
    )

    if missing_ids:
        problems.append(
            "누락 Action ID: "
            + ", ".join(
                sorted(missing_ids)
            )
        )

    # ==============================================
    # ⑬ 전체 오류 확인 및 최종 검사 결과 출력
    # ==============================================
    # 앞의 검사에서 problems에 저장된
    # 모든 오류를 확인합니다.
    #
    # 오류가 있다면:
    # 중복 오류 메시지를 제거하고 출력합니다.
    # RuntimeError를 발생시켜 종료합니다.
    #
    # dict.fromkeys(problems):
    # 오류 메시지의 순서를 유지하면서
    # 동일한 메시지를 한 번만 출력합니다.
    #
    # 오류가 없다면:
    # Action class check: OK를 출력합니다.
    #
    # 이 메시지는 6개 Action의 설정과
    # MP4 파일 연결이 정상이라는 의미입니다.

    print()

    if problems:
        for problem in dict.fromkeys(
            problems
        ):
            print(
                "[ERROR]",
                problem,
            )

        raise RuntimeError(
            "Action 설정을 "
            "다시 확인하세요."
        )

    print(
        "Action class check: OK"
    )


# ==================================================
# 프로그램 실행 시작
# ==================================================
# Python 파일을 직접 실행했을 때
# main() 함수를 호출합니다.

if __name__ == "__main__":
    main()