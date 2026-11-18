
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# Path: 폴더와 파일 경로를 다룹니다.
# Pandas: 18개 CSV를 읽고 하나의 표로 합칩니다.
# train_test_split: 표의 Row를 무작위로 나눕니다.
# 이 실습은 모델을 학습하지 않고 분리의 위험을 확인합니다.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)


# ==================================================
# ② 프로젝트 ROOT와 Keypoint CSV 폴더 설정
# ==================================================
# 이 파일은 프로젝트의 scripts 폴더에 저장한다고 가정합니다.
# parents[1]은 프로젝트 최상위(ROOT) 폴더를 가리킵니다.
# KEYPOINT_ROOT 아래의 모든 Subject/Action 폴더에서
# 26번 코드로 생성한 Keypoint CSV 18개를 찾습니다.

ROOT = Path(__file__).resolve().parents[1]

KEYPOINT_ROOT = (
    ROOT
    / "data"
    / "day05"
    / "keypoints"
)


def main() -> None:

    # ==============================================
    # ③ Keypoint CSV 검색 및 18개 파일 수 확인
    # ==============================================
    # rglob("*.csv"): 하위 폴더를 포함하여 CSV를 찾습니다.
    # sorted(): 검색한 경로를 정렬합니다.
    # 이번 기본 실습의 입력은 3명 × 6개 Action = 18개입니다.
    # 파일이 18개가 아니라면 오류를 발생시켜 중단합니다.

    csv_paths = sorted(
        KEYPOINT_ROOT.rglob("*.csv")
    )

    if len(csv_paths) != 18:
        raise RuntimeError(
            "기본 Keypoint CSV는 "
            f"18개여야 합니다. "
            f"현재 {len(csv_paths)}개입니다."
        )

    # ==============================================
    # ④ CSV별 Frame 정보를 읽어 목록에 모으기
    # ==============================================
    # frames: 각 영상의 Frame 데이터표를 모을 리스트입니다.
    # 18개 CSV를 하나씩 읽되 아래 네 컬럼만 선택합니다.
    #
    # clip_id:
    # 어느 영상에 속하는 Frame인지 나타냅니다.
    #
    # frame_id:
    # 해당 영상 안에서의 Frame 번호입니다.
    #
    # subject:
    # 촬영한 사람(S01/S02/S03)을 뜻합니다.
    #
    # action_label:
    # 영상의 동작 이름입니다.
    #
    # 각 CSV의 모든 Frame Row를 frames에 추가합니다.

    frames = []

    for path in csv_paths:
        df = pd.read_csv(
            path,
            usecols=[
                "clip_id",
                "frame_id",
                "subject",
                "action_label",
            ],
        )
        frames.append(df)

    # ==============================================
    # ⑤ 18개 CSV의 모든 Frame을 하나의 표로 결합
    # ==============================================
    # pd.concat():
    # frames에 모은 DataFrame을 세로로 합칩니다.
    #
    # ignore_index=True:
    # 합친 표의 행 인덱스를 새로 부여합니다.
    #
    # 여기서 각 행은 '영상 하나'가 아니라
    # 'Frame 하나'입니다.
    #
    # 같은 Clip에 속하는 수많은 Frame이
    # 하나의 표에 모입니다.

    all_frames = pd.concat(
        frames,
        ignore_index=True,
    )

    # ==============================================
    # ⑥ Frame 단위로 Train 80% / Test 20% 분리
    # ==============================================
    # train_test_split():
    # all_frames의 각 Row를 무작위로 분리합니다.
    #
    # test_size=0.20:
    # 전체 Frame 중 약 20%를 Test로 사용합니다.
    #
    # 나머지 약 80%는 Train에 포함됩니다.
    #
    # random_state=42:
    # 동일한 입력 데이터에서 같은 분할을 재현합니다.
    #
    # shuffle=True:
    # Frame Row 순서를 무작위로 섞습니다.
    #
    # 주의:
    # Subject나 Clip을 묶어 유지하지 않으므로
    # 동일한 사람이나 영상의 Frame이
    # Train과 Test 양쪽에 들어갈 수 있습니다.
    #
    # 이것은 잘못된 분할의 위험을 보여주기 위한
    # 교육용 실험입니다.

    train_df, test_df = (
        train_test_split(
            all_frames,
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )
    )

    # ==============================================
    # ⑦ Train과 Test의 Subject 목록을 집합으로 생성
    # ==============================================
    # train_df와 test_df에서 subject 컬럼을 가져옵니다.
    #
    # astype(str):
    # Subject ID를 문자열로 변환합니다.
    #
    # set():
    # 중복된 Subject ID를 제거합니다.
    #
    # 예:
    # train_subjects = {"S01", "S02", "S03"}
    #
    # 같은 사람이 두 데이터셋에 모두 있는지
    # 확인하기 위한 준비 단계입니다.

    train_subjects = set(
        train_df["subject"]
        .astype(str)
    )

    test_subjects = set(
        test_df["subject"]
        .astype(str)
    )

    # ==============================================
    # ⑧ Train과 Test에 공통으로 포함된 Subject 찾기
    # ==============================================
    # & 기호:
    # 두 집합의 교집합을 구합니다.
    #
    # 즉 Train과 Test에 모두 존재하는
    # Subject ID를 찾습니다.
    #
    # 예:
    # Train = {"S01", "S02"}
    # Test  = {"S02", "S03"}
    #
    # 교집합 = {"S02"}
    #
    # 같은 사람의 데이터가 양쪽에 존재한다면
    # Subject Leakage에 해당하는 상황입니다.

    subject_overlap = (
        train_subjects
        & test_subjects
    )

    # ==============================================
    # ⑨ Train과 Test의 Clip ID 목록 생성
    # ==============================================
    # 이번에는 Subject가 아니라 Clip을 비교합니다.
    #
    # clip_id:
    # 개별 MP4 영상 하나를 식별하는 ID입니다.
    #
    # 예:
    # S01_A006_T01
    #
    # Train과 Test에 어떤 Clip이 포함되었는지
    # 각각 집합으로 만듭니다.
    #
    # set()은 중복 Clip ID를 제거합니다.

    train_clips = set(
        train_df["clip_id"]
        .astype(str)
    )

    test_clips = set(
        test_df["clip_id"]
        .astype(str)
    )

    # ==============================================
    # ⑩ Train과 Test에 공통으로 포함된 Clip 찾기
    # ==============================================
    # Train과 Test의 Clip 집합을 비교합니다.
    #
    # 같은 clip_id가 양쪽에 있다면
    # 동일한 MP4에서 나온 서로 다른 Frame이
    # Train과 Test로 나뉘었다는 의미입니다.
    #
    # 같은 영상의 인접 Frame은
    # 사람의 자세와 배경이 매우 비슷할 수 있습니다.
    #
    # 따라서 새로운 데이터에 대한 실제 성능보다
    # 평가 점수가 높게 나타날 위험이 있습니다.

    clip_overlap = (
        train_clips
        & test_clips
    )

    # ==============================================
    # ⑪ 실험 제목과 Train·Test Frame 수 출력
    # ==============================================
    # Random Frame Split 실험임을 출력합니다.
    #
    # Train에 들어간 Frame 수를 출력합니다.
    #
    # Test에 들어간 Frame 수를 출력합니다.
    #
    # 두 데이터셋의 Frame 수를 비교하면
    # 약 80:20으로 분리되었는지 확인할 수 있습니다.

    print("Random Frame Split Demo")
    print()

    print(
        f"Train frames: "
        f"{len(train_df)}"
    )

    print(
        f"Test frames : "
        f"{len(test_df)}"
    )

    print()

    # ==============================================
    # ⑫ Subject 중복 목록과 Clip 중복 개수 출력
    # ==============================================
    # Subject overlap:
    # Train과 Test 양쪽에 포함된 Subject 목록입니다.
    #
    # Clip overlap count:
    # Train과 Test 양쪽에 포함된 Clip의 개수입니다.
    #
    # sorted():
    # Subject 목록을 정렬하여 출력합니다.
    #
    # len():
    # 중복 Clip의 개수를 계산합니다.
    #
    # 실제 중복 개수는 CSV의 Frame 구성에 따라
    # 달라질 수 있습니다.

    print(
        "Subject overlap:",
        sorted(subject_overlap),
    )

    print(
        "Clip overlap count:",
        len(clip_overlap),
    )

    print()

    # ==============================================
    # ⑬ 중복이 발견되면 LEAKAGE 경고 출력
    # ==============================================
    # subject_overlap에 값이 존재하는가?
    #
    # 예:
    # 같은 Subject가 Train과 Test에 있으므로
    # LEAKAGE 경고를 출력합니다.
    #
    # clip_overlap에 값이 존재하는가?
    #
    # 예:
    # 같은 Clip의 Frame이 양쪽에 있으므로
    # LEAKAGE 경고를 출력합니다.
    #
    # 이 코드는 문제를 보여주는 시연이므로
    # 오류를 발생시키거나 분할을 수정하지 않습니다.

    if subject_overlap:
        print(
            "[LEAKAGE] 같은 Subject가 "
            "Train과 Test에 존재"
        )

    if clip_overlap:
        print(
            "[LEAKAGE] 같은 Clip의 "
            "Frame이 Train과 Test에 존재"
        )

    # ==============================================
    # ⑭ Random Frame Split 사용 금지 안내 출력
    # ==============================================
    # 같은 사람과 같은 영상의 Frame이
    # Train과 Test에 섞일 수 있으므로
    # 이번 분할은 최종 평가용으로 사용하지 않습니다.
    #
    # 실제 5일차 데이터 분할은
    # 30번 코드의 Subject-wise Split을 사용합니다.
    #
    # 이번 코드는 분할 결과를 보여줄 뿐,
    # 별도의 CSV를 생성하거나 저장하지 않습니다.

    print()

    print(
        "이 Random Frame Split은 "
        "최종 평가용으로 "
        "사용하지 않습니다."
    )


# ==================================================
# ⑮ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# main()을 시작하여 18개 Keypoint CSV를 읽고,
# 무작위 Frame 분할과 Leakage 검사를 수행합니다.

if __name__ == "__main__":
    main()