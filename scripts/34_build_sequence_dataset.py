
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# json: 6일차 Baseline 설정 파일을 읽습니다.
# sys : 프로젝트 내부 src 모듈을 불러옵니다.
# Path: 입력·출력 파일의 경로를 관리합니다.
# pandas: Split CSV와 Keypoint CSV를 읽고 저장합니다.

import json
import sys
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT 및 모듈 검색 경로 설정
# ==================================================
# 현재 Python 파일이 scripts 폴더에 있다고 가정합니다.
# parents[1]은 프로젝트 ROOT를 가리킵니다.
#
# sys.path.insert():
# src 폴더의 공통 함수를 import할 수 있도록
# 프로젝트 ROOT를 Python 검색 경로에 추가합니다.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT),
)


# ==================================================
# ③ Sequence 변환에 필요한 공통 함수 가져오기
# ==================================================
# build_frame_feature_matrix:
# Keypoint CSV를 38-D Frame Feature로 변환합니다.
#
# flatten_sequence:
# 30 × 38 배열을 1,140개 숫자로 펼칩니다.
#
# resample_sequence:
# 원래 Frame의 시간 간격을 고려하여
# 전체 동작을 동일한 Step 수로 맞춥니다.
#
# sequence_feature_names:
# t00_nose_x처럼 Feature 컬럼 이름을 만듭니다.
#
# 이 함수들은 src/sequence_utils.py에 정의되어 있습니다.

from src.sequence_utils import (
    build_frame_feature_matrix,
    flatten_sequence,
    resample_sequence,
    sequence_feature_names,
)


# ==================================================
# ④ 설정 파일·입력 Split·출력 Sequence 경로 설정
# ==================================================
# CONFIG_PATH:
# 6일차의 Sequence Step 및 최소 Frame 수를
# 지정하는 JSON 설정 파일입니다.
#
# SPLIT_DIR:
# 5일차에서 만든 Train / Validation / Test
# Split CSV가 저장된 폴더입니다.
#
# OUTPUT_DIR:
# 변환된 6일차 Sequence CSV를 저장하는 폴더입니다.

CONFIG_PATH = (
    ROOT
    / "configs"
    / "day06_baseline.json"
)

SPLIT_DIR = (
    ROOT
    / "data"
    / "day05"
    / "splits"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
)


# ==================================================
# ⑤ 반드시 포함해야 하는 6개 Action ID 정의
# ==================================================
# 이번 기본 수업에서 사용하는 Action은 6개입니다.
#
# A001 ~ A006
#
# Train, Validation, Test 각각에
# 이 6개 Action이 모두 포함되어야 합니다.
#
# 이후 Sequence Dataset을 완성한 뒤
# 이 목록과 비교하여 Action 누락을 검사합니다.

EXPECTED_ACTIONS = {
    "A001",
    "A002",
    "A003",
    "A004",
    "A005",
    "A006",
}


# ==================================================
# ⑥ 6일차 Baseline 설정 JSON 읽기
# ==================================================
# load_day06_config()는 설정 파일을 읽는 함수입니다.
#
# day06_baseline.json에는 다음 값 등이 있습니다.
#
# sequence_steps:
# Clip 하나를 변환할 시간 Step 수입니다.
# 기본 수업에서는 30을 사용합니다.
#
# min_valid_frames:
# Sequence 생성에 필요한 최소 유효 Frame 수입니다.
# 기본 수업에서는 10을 사용합니다.
#
# json.load()로 내용을 딕셔너리로 변환합니다.

def load_day06_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ==================================================
# ⑦ Split 하나를 처리할 함수 및 입력 파일 검사
# ==================================================
# build_split_dataset()은 하나의 Split을 처리합니다.
#
# split_name:
# "train", "val", "test" 중 하나입니다.
#
# config:
# ⑥에서 읽은 Baseline 설정입니다.
#
# 예:
# split_name = "train"
# → data/day05/splits/train.csv
#
# 파일이 없다면 FileNotFoundError를 발생시킵니다.

def build_split_dataset(
    split_name: str,
    config: dict,
) -> Path:
    split_path = (
        SPLIT_DIR
        / f"{split_name}.csv"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            split_path
        )

    # ==============================================
    # ⑧ Split Manifest와 Sequence 변환 기준 읽기
    # ==============================================
    # Split CSV를 Pandas로 읽습니다.
    #
    # Manifest의 한 행에는 Clip 하나의 정보와
    # 해당 Keypoint CSV 파일 경로가 들어 있습니다.
    #
    # target_steps:
    # 한 Clip을 몇 개 시간 지점으로 맞출지 결정합니다.
    #
    # min_valid_frames:
    # 사용할 수 있는 Frame이 최소 몇 개 필요한지
    # 결정합니다.
    #
    # 이번 기본 설정:
    # target_steps = 30
    # min_valid_frames = 10

    manifest = pd.read_csv(
        split_path
    )

    target_steps = int(
        config["sequence_steps"]
    )

    min_valid_frames = int(
        config[
            "min_valid_frames"
        ]
    )

    # ==============================================
    # ⑨ Sequence Feature의 컬럼 이름 생성
    # ==============================================
    # sequence_feature_names()는 30개 Step에
    # 들어갈 모든 Feature 이름을 생성합니다.
    #
    # 한 Step에는 38개 Frame Feature가 있습니다.
    #
    # 30 Step × 38 Feature = 1,140개
    #
    # 예:
    # t00_nose_x
    # t00_nose_y
    # ...
    # t29_right_knee_angle
    #
    # 생성한 이름은 나중에 CSV의 컬럼명이 됩니다.

    feature_names = (
        sequence_feature_names(
            target_steps
        )
    )

    # ==============================================
    # ⑩ 결과 목록 준비 및 Clip별 Keypoint 경로 생성
    # ==============================================
    # rows:
    # 정상적으로 생성한 Sequence 데이터를 저장합니다.
    #
    # skipped:
    # 문제가 있어서 건너뛴 Clip과 원인을 기록합니다.
    #
    # manifest.iterrows():
    # Split에 있는 Clip을 하나씩 가져옵니다.
    #
    # 각 Clip의 keypoint_csv와 ROOT를 연결하여
    # 실제 파일 경로를 만듭니다.

    rows = []
    skipped = []

    for _, item in (
        manifest.iterrows()
    ):
        keypoint_path = (
            ROOT
            / str(
                item[
                    "keypoint_csv"
                ]
            )
        )

        # ==========================================
        # ⑪ Keypoint CSV가 없으면 건너뛰기
        # ==========================================
        # keypoint_path.exists():
        # 해당 Keypoint CSV가 존재하는지 확인합니다.
        #
        # 없다면 skipped에 다음 내용을 기록합니다.
        #
        # clip_id:
        # 문제가 있는 Clip의 ID입니다.
        #
        # reason:
        # KEYPOINT_CSV_MISSING
        #
        # continue:
        # 현재 Clip을 건너뛰고 다음 Clip으로 이동합니다.

        if not keypoint_path.exists():
            skipped.append(
                {
                    "clip_id": (
                        item["clip_id"]
                    ),
                    "reason": (
                        "KEYPOINT_CSV_MISSING"
                    ),
                }
            )
            continue

        # ==========================================
        # ⑫ Keypoint CSV를 38-D Frame Feature로 변환
        # ==========================================
        # 먼저 Clip의 Keypoint CSV를 읽습니다.
        #
        # build_frame_feature_matrix():
        # 5일차 원본 관절 좌표를 읽고
        # 각 유효 Frame을 38-D Feature로 변환합니다.
        #
        # 38-D 구성:
        # 34개 정규화 좌표 + 4개 관절각도
        #
        # frame_matrix:
        # 유효 Frame마다 38개 Feature를 저장한 배열입니다.
        #
        # frame_ids:
        # 실제 원본 Frame 번호를 저장합니다.
        #
        # 예:
        # 유효 Frame 85개
        # frame_matrix.shape = (85, 38)
        #
        # 이 변환 과정에서 사용할 수 없는 Frame은
        # 공통 함수의 검사 기준에 따라 제외됩니다.

        frame_df = pd.read_csv(
            keypoint_path
        )

        (
            frame_matrix,
            frame_ids,
        ) = (
            build_frame_feature_matrix(
                frame_df
            )
        )

        # ==========================================
        # ⑬ 최소 유효 Frame 수 확인
        # ==========================================
        # Frame Feature를 생성한 후
        # 사용할 수 있는 Frame이 충분한지 확인합니다.
        #
        # 예:
        # min_valid_frames = 10
        #
        # 유효 Frame이 7개라면
        # Sequence를 생성하지 않습니다.
        #
        # skipped에 다음 이유를 기록합니다.
        #
        # TOO_FEW_VALID_FRAMES
        #
        # 이후 다음 Clip으로 이동합니다.

        if (
            len(frame_matrix)
            < min_valid_frames
        ):
            skipped.append(
                {
                    "clip_id": (
                        item["clip_id"]
                    ),
                    "reason": (
                        "TOO_FEW_VALID_FRAMES"
                    ),
                }
            )
            continue

        # ==========================================
        # ⑭ 서로 다른 Frame 길이를 30 Step으로 통일
        # ==========================================
        # resample_sequence()는 Clip 전체의 동작을
        # 동일한 개수의 시간 Step으로 변환합니다.
        #
        # 예:
        # 85 Frame  → 30 Step
        # 125 Frame → 30 Step
        # 266 Frame → 30 Step
        #
        # 단순히 앞의 30 Frame만 선택하지 않습니다.
        #
        # frame_ids를 이용하여
        # 원래 Frame 사이의 시간 간격을 고려합니다.
        #
        # 변환 후 데이터의 기본 형태:
        #
        # (30, 38)
        #
        # 즉:
        # 30개 시간 지점마다 38개의 Feature가 있습니다.

        sequence = (
            resample_sequence(
                frame_matrix,
                frame_ids,
                target_steps=(
                    target_steps
                ),
            )
        )

        # ==========================================
        # ⑮ Sequence를 1,140개 숫자로 펼치고 검사
        # ==========================================
        # flatten_sequence()는 2차원 배열을
        # 1차원 배열로 펼치는 함수입니다.
        #
        # 변환 전:
        # (30, 38)
        #
        # 변환 후:
        # (1140,)
        #
        # 즉 Clip 하나가 1,140개 숫자로 바뀝니다.
        #
        # 생성한 Feature 개수가
        # target_steps × 38과 같은지 검사합니다.
        #
        # 개수가 다르면 RuntimeError를 발생시킵니다.

        flattened = (
            flatten_sequence(
                sequence
            )
        )

        if len(flattened) != (
            target_steps * 38
        ):
            raise RuntimeError(
                "Sequence Feature 수가 "
                "예상과 다릅니다."
            )

        # ==========================================
        # ⑯ Clip의 기본정보와 유효 Frame 범위 기록
        # ==========================================
        # 한 Clip을 최종 CSV의 한 행으로 저장하기 위해
        # 먼저 식별용 정보를 딕셔너리로 만듭니다.
        #
        # clip_id:
        # 영상 식별 ID
        #
        # subject:
        # 촬영 대상자
        #
        # action_id / action_label:
        # 동작 ID와 동작 이름
        #
        # take:
        # 촬영 번호
        #
        # valid_frames:
        # 정상적으로 38-D Feature를 만든 Frame 개수
        #
        # first_valid_frame / last_valid_frame:
        # 첫 번째와 마지막 유효 Frame 번호
        #
        # 총 8개 메타데이터 항목을 기록합니다.

        row = {
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
            "take": item["take"],
            "valid_frames": (
                len(frame_matrix)
            ),
            "first_valid_frame": (
                int(frame_ids[0])
            ),
            "last_valid_frame": (
                int(frame_ids[-1])
            ),
        }

        # ==========================================
        # ⑰ Sequence Feature 1,140개를 컬럼별로 연결
        # ==========================================
        # feature_names:
        # 1,140개 Feature의 컬럼 이름입니다.
        #
        # flattened:
        # 1,140개 Feature의 실제 숫자입니다.
        #
        # zip():
        # Feature 이름과 값을 하나씩 짝으로 묶습니다.
        #
        # 예:
        # row["t00_nose_x"] = 해당 Feature 값
        #
        # float():
        # NumPy 숫자를 일반 Python 숫자로 변환합니다.
        #
        # 결과적으로 row에는
        # 8개 기본정보와 1,140개 Feature가 들어갑니다.

        for name, value in zip(
            feature_names,
            flattened,
        ):
            row[name] = float(value)

        # ==========================================
        # ⑱ 현재 Clip 결과 저장 및 진행 상황 출력
        # ==========================================
        # 정상적으로 생성된 Clip의 결과를
        # rows 목록에 추가합니다.
        #
        # 터미널에는 다음 정보를 출력합니다.
        #
        # Split 이름
        # Clip ID
        # 유효 Frame 수
        # 변환한 Sequence Step 수
        #
        # 예:
        # train S01_A001_T01 valid=85 → 30 steps

        rows.append(row)

        print(
            f"{split_name:5s} "
            f"{item['clip_id']} "
            f"valid={len(frame_matrix)} "
            f"→ {target_steps} steps"
        )

    # ==============================================
    # ⑲ 건너뛴 Clip이 있다면 별도 CSV로 저장
    # ==============================================
    # skipped에 문제가 있는 Clip이 있다면
    # 해당 정보를 별도 CSV 파일로 저장합니다.
    #
    # 예:
    # train_skipped.csv
    #
    # 각 행에는 다음 정보가 들어 있습니다.
    #
    # clip_id
    # reason
    #
    # reason 예:
    # KEYPOINT_CSV_MISSING
    # TOO_FEW_VALID_FRAMES
    #
    # 이후 어떤 Clip에서 문제가 발생했는지
    # 파일로 확인할 수 있습니다.
    #
    # 주의:
    # 건너뛴 Clip이 없는 경우에는 새 파일을 쓰지 않습니다.
    # 기존 *_skipped.csv가 있더라도 자동 삭제하지 않습니다.

    if skipped:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        skip_path = (
            OUTPUT_DIR
            / f"{split_name}_skipped.csv"
        )

        pd.DataFrame(
            skipped
        ).to_csv(
            skip_path,
            index=False,
            encoding="utf-8-sig",
        )

    # ==============================================
    # ⑳ Split별 Sequence가 정확히 6개인지 확인
    # ==============================================
    # rows를 Pandas DataFrame으로 변환합니다.
    #
    # 이번 기본 수업에서는 각 Subject가
    # 6개의 Action을 수행했습니다.
    #
    # 따라서 Train, Validation, Test 각각
    # 6개 Sequence가 생성되어야 합니다.
    #
    # 생성된 Row가 6개가 아니면
    # RuntimeError를 발생시킵니다.

    result = pd.DataFrame(rows)

    if len(result) != 6:
        raise RuntimeError(
            f"{split_name} Sequence는 "
            "6개가 모두 생성되어야 합니다. "
            f"현재 {len(result)}개입니다."
        )

    # ==============================================
    # ㉑ 6개 Action이 빠짐없이 있는지 확인
    # ==============================================
    # 생성된 Sequence의 action_id를 가져옵니다.
    #
    # astype(str):
    # 문자열로 변환합니다.
    #
    # str.upper():
    # 영문을 대문자로 통일합니다.
    #
    # set():
    # 중복 Action ID를 제거합니다.
    #
    # EXPECTED_ACTIONS와 비교하여
    # A001~A006이 모두 있는지 검사합니다.
    #
    # 다르다면 RuntimeError를 발생시킵니다.

    actions = set(
        result["action_id"]
        .astype(str)
        .str.upper()
    )

    if actions != EXPECTED_ACTIONS:
        raise RuntimeError(
            f"{split_name}에 "
            "A001~A006이 모두 없습니다."
        )

    # ==============================================
    # ㉒ Train·Validation·Test Sequence CSV 저장
    # ==============================================
    # 출력 폴더가 없다면 생성합니다.
    #
    # 현재 처리 중인 Split 이름으로
    # CSV 저장 경로를 만듭니다.
    #
    # train → train.csv
    # val   → val.csv
    # test  → test.csv
    #
    # index=False:
    # DataFrame의 인덱스를 저장하지 않습니다.
    #
    # encoding="utf-8-sig":
    # Excel에서도 한글을 읽기 쉽게 저장합니다.
    #
    # 기존 파일이 있다면 덮어씁니다.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{split_name}.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ==============================================
    # ㉓ 생성된 Sequence 개수와 파일 경로 출력
    # ==============================================
    # 현재 Split에 생성된 Sequence 수를 출력합니다.
    #
    # 저장 경로도 함께 출력합니다.
    #
    # return output_path:
    # 생성한 CSV의 파일 경로를 반환합니다.

    print()
    print(
        f"{split_name} Sequence: "
        f"{len(result)}"
    )
    print(
        f"Saved: {output_path}"
    )
    print()

    return output_path


# ==================================================
# ㉔ main()에서 Train·Validation·Test를 모두 처리
# ==================================================
# 프로그램의 전체 실행을 관리하는 함수입니다.
#
# 먼저 6일차 Baseline 설정을 읽습니다.
#
# 다음 순서로 Split을 처리합니다.
#
# train
# → S01의 6개 Clip
#
# val
# → S02의 6개 Clip
#
# test
# → S03의 6개 Clip
#
# 각 Split에 build_split_dataset()을 호출합니다.
#
# 모두 성공하면 READY 메시지를 출력합니다.

def main() -> None:
    config = load_day06_config()

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        build_split_dataset(
            split_name,
            config,
        )

    print(
        "Day06 Sequence Dataset: READY"
    )


# ==================================================
# ㉕ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# main()에서 모든 Split을 처리하고
# 결과 CSV를 생성합니다.

if __name__ == "__main__":
    main()