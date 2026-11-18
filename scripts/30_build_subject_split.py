
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse: 터미널에서 실행 옵션을 입력받습니다.
# json    : Subject-wise Split 설정 JSON을 읽습니다.
# Path    : 파일과 폴더 경로를 관리합니다.
# pandas  : 승인된 Clip을 Subject별로 분리하고 저장합니다.

import argparse
import json
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 기본 파일·폴더 경로 설정
# ==================================================
# ROOT:
# scripts 폴더의 상위인 프로젝트 루트입니다.
#
# DEFAULT_MANIFEST_PATH:
# 29번 코드에서 만든 승인 Clip 목록입니다.
#
# DEFAULT_SPLIT_CONFIG:
# 어떤 Subject를 어떤 Split에 넣을지 정한
# JSON 설정 파일입니다.
#
# DEFAULT_OUTPUT_DIR:
# train.csv, val.csv, test.csv를 저장할 폴더입니다.

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "approved_clip_manifest.csv"
)

DEFAULT_SPLIT_CONFIG = (
    ROOT
    / "configs"
    / "day05_subject_split.json"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "day05"
    / "splits"
)


# ==================================================
# ③ 모든 Split에 있어야 할 Action ID 정의
# ==================================================
# 이번 기본 데이터에는 총 6개 Action이 있습니다.
#
# A001~A006을 쉼표로 연결하여 문자열로 지정합니다.
#
# 이후 ⑤에서 이 문자열을 set으로 변환합니다.
#
# 이 목록은 각 Split에 필요한 Action의 종류입니다.
#
# 주의:
# 각 Action의 Clip이 정확히 몇 개 있는지까지
# 검사하는 설정은 아닙니다.

DEFAULT_EXPECTED_ACTIONS = (
    "A001,A002,A003,A004,A005,A006"
)


# ==================================================
# ④ 터미널 실행 옵션 설정
# ==================================================
# --manifest-path:
# 승인된 Clip 목록 CSV 경로입니다.
#
# --split-config:
# Train / Validation / Test Subject 배정 JSON입니다.
#
# --output-dir:
# 세 개의 Split CSV를 저장할 폴더입니다.
#
# --expected-actions:
# 각 Split에 반드시 포함되어야 할 Action ID입니다.
#
# 옵션을 입력하지 않으면 위 기본값을 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-path",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
    )

    parser.add_argument(
        "--split-config",
        default=str(
            DEFAULT_SPLIT_CONFIG
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--expected-actions",
        default=DEFAULT_EXPECTED_ACTIONS,
        help=(
            "쉼표로 구분합니다. "
            "예: A006"
        ),
    )

    return parser.parse_args()


# ==================================================
# ⑤ Action ID 문자열을 중복 없는 집합으로 변환
# ==================================================
# parse_expected_actions()는 문자열 형태의
# Action 목록을 집합(set)으로 변환합니다.
#
# 예:
# "A001,A002,A003"
#        ↓
# {"A001", "A002", "A003"}
#
# split(","):
# 쉼표를 기준으로 문자열을 나눕니다.
#
# strip():
# 앞뒤 공백을 제거합니다.
#
# upper():
# 영문을 대문자로 통일합니다.
#
# set:
# 중복된 Action을 제거합니다.
#
# Action이 하나도 없으면 ValueError를 발생시킵니다.

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
# ⑥ Subject-wise Split 설정 JSON 읽기
# ==================================================
# load_split_config()는 JSON 설정 파일을 읽습니다.
#
# JSON에는 세 가지 Subject 목록이 필요합니다.
#
# train_subjects
# val_subjects
# test_subjects
#
# 예:
# "train_subjects": ["S01"]
# "val_subjects": ["S02"]
# "test_subjects": ["S03"]
#
# json.load():
# JSON 내용을 Python 딕셔너리로 변환합니다.
#
# 실제 배정은 JSON 파일의 내용으로 결정됩니다.

def load_split_config(
    split_config: Path,
) -> dict:
    with split_config.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ==================================================
# ⑦ Train·Validation·Test Subject 중복 검사
# ==================================================
# validate_subject_sets()는 서로 다른 Split에
# 같은 Subject가 들어갔는지 확인하는 함수입니다.
#
# set():
# Subject 목록을 집합으로 변환합니다.
#
# &:
# 두 집합의 교집합을 구합니다.
#
# 검사 1:
# Train과 Validation에 같은 Subject가 있는가?
#
# 검사 2:
# Train과 Test에 같은 Subject가 있는가?
#
# 검사 3:
# Validation과 Test에 같은 Subject가 있는가?
#
# 같은 사람이 서로 다른 Split에 들어가면
# Subject Leakage가 발생할 수 있으므로 중단합니다.

def validate_subject_sets(
    train_subjects,
    val_subjects,
    test_subjects,
) -> None:
    train = set(train_subjects)
    val = set(val_subjects)
    test = set(test_subjects)

    if train & val:
        raise RuntimeError(
            "Train / Validation "
            "Subject 중복"
        )

    if train & test:
        raise RuntimeError(
            "Train / Test "
            "Subject 중복"
        )

    if val & test:
        raise RuntimeError(
            "Validation / Test "
            "Subject 중복"
        )


# ==================================================
# ⑧ 각 Split에 데이터가 존재하는지 검사
# ==================================================
# require_action_coverage()는 Train, Validation,
# Test 각각에 필요한 Action이 있는지 검사합니다.
#
# split_name:
# 현재 검사할 Split 이름입니다.
#
# df:
# 현재 Split에 포함된 Clip 목록입니다.
#
# expected_actions:
# 필요한 Action ID 집합입니다.
#
# df.empty:
# 데이터가 비어 있는지 확인합니다.
#
# 비어 있다면 RuntimeError를 발생시킵니다.

def require_action_coverage(
    split_name: str,
    df: pd.DataFrame,
    expected_actions: set[str],
) -> None:
    if df.empty:
        raise RuntimeError(
            f"{split_name} 데이터가 "
            "비어 있습니다."
        )

    # ==============================================
    # ⑨ 현재 Split에 필수 Action이 모두 있는지 확인
    # ==============================================
    # 현재 Split의 action_id를 가져옵니다.
    #
    # astype(str):
    # 모든 Action ID를 문자열로 변환합니다.
    #
    # str.upper():
    # 영문을 대문자로 통일합니다.
    #
    # set():
    # 중복 Action ID를 제거합니다.
    #
    # expected_actions - actual_actions:
    # 필요한데 실제 데이터에 없는 Action을 찾습니다.
    #
    # 하나라도 빠지면 RuntimeError를 발생시킵니다.
    #
    # 주의:
    # 각 Action이 최소 한 번 존재하는지만 검사합니다.
    # 추가 Action이나 Clip 중복은 검사하지 않습니다.

    actual_actions = set(
        df["action_id"]
        .astype(str)
        .str.upper()
    )

    missing_actions = (
        expected_actions
        - actual_actions
    )

    if missing_actions:
        raise RuntimeError(
            f"{split_name}에 없는 Action: "
            + ", ".join(
                sorted(
                    missing_actions
                )
            )
        )

    # ==============================================
    # ⑩ Split별 Subject·Clip 수·Action 분포 출력
    # ==============================================
    # 현재 Split의 정보를 터미널에 출력합니다.
    #
    # Subjects:
    # 해당 Split에 포함된 Subject 목록입니다.
    #
    # Clips:
    # 해당 Split의 전체 Clip 개수입니다.
    #
    # value_counts():
    # Action Label별 Clip 개수를 계산합니다.
    #
    # sort_index():
    # Action Label을 이름순으로 정렬합니다.
    #
    # 이를 통해 데이터 구성을 한눈에 확인합니다.

    print()
    print(f"[{split_name}]")
    print(
        "Subjects:",
        sorted(
            df["subject"].unique()
        ),
    )
    print("Clips:", len(df))
    print(
        df["action_label"]
        .value_counts()
        .sort_index()
    )


# ==================================================
# ⑪ main() 실행 옵션과 파일 경로 준비
# ==================================================
# main()에서 실제 Subject-wise Split을 수행합니다.
#
# parse_args()로 실행 옵션을 읽습니다.
#
# Manifest, Split 설정, 출력 폴더 경로를 준비합니다.
#
# ⑤의 parse_expected_actions()를 호출하여
# 필수 Action ID 집합을 만듭니다.

def main() -> None:
    args = parse_args()

    manifest_path = Path(
        args.manifest_path
    )
    split_config = Path(
        args.split_config
    )
    output_dir = Path(
        args.output_dir
    )

    expected_actions = (
        parse_expected_actions(
            args.expected_actions
        )
    )

    # ==============================================
    # ⑫ 입력 Manifest와 Split 설정 파일 존재 검사
    # ==============================================
    # 먼저 Approved Manifest가 존재하는지 확인합니다.
    #
    # 없으면 FileNotFoundError를 발생시킵니다.
    #
    # Subject Split 설정 JSON도 검사합니다.
    #
    # 두 파일이 모두 존재하면
    # load_split_config()를 호출하여 JSON을 읽습니다.

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    if not split_config.exists():
        raise FileNotFoundError(
            split_config
        )

    config = load_split_config(
        split_config
    )

    # ==============================================
    # ⑬ JSON에서 Subject별 Split 목록 가져오기
    # ==============================================
    # config 딕셔너리에서 다음 목록을 읽습니다.
    #
    # train_subjects:
    # 모델 학습에 사용할 Subject 목록입니다.
    #
    # val_subjects:
    # 모델 설정과 Feature를 검증할 Subject 목록입니다.
    #
    # test_subjects:
    # 최종 성능 확인에 사용할 Subject 목록입니다.
    #
    # 이번 교육 예시에서는:
    # S01 → Train
    # S02 → Validation
    # S03 → Test
    #
    # 실제 배정은 JSON 설정에 따라 달라집니다.

    train_subjects = (
        config["train_subjects"]
    )
    val_subjects = (
        config["val_subjects"]
    )
    test_subjects = (
        config["test_subjects"]
    )

    # ==============================================
    # ⑭ 세 Split에 Subject가 중복 배정되었는지 확인
    # ==============================================
    # ⑦에서 만든 validate_subject_sets()를 호출합니다.
    #
    # Subject가 여러 Split에 중복 배정되면
    # RuntimeError를 발생시키고 중단합니다.
    #
    # 중복이 없다면 다음 단계로 이동합니다.

    validate_subject_sets(
        train_subjects,
        val_subjects,
        test_subjects,
    )

    # ==============================================
    # ⑮ Approved Manifest 읽기 및 Subject 집합 준비
    # ==============================================
    # 29번 코드에서 만든 Approved Manifest를 읽습니다.
    #
    # configured_subjects:
    # JSON에 배정된 모든 Subject의 집합입니다.
    #
    # data_subjects:
    # 실제 승인 데이터에 존재하는 Subject의 집합입니다.
    #
    # 이 두 집합을 비교하여
    # 배정되지 않았거나 누락된 Subject를 찾습니다.

    df = pd.read_csv(
        manifest_path
    )

    configured_subjects = set(
        train_subjects
        + val_subjects
        + test_subjects
    )

    data_subjects = set(
        df["subject"].astype(str)
    )

    # ==============================================
    # ⑯ 미배정 Subject와 누락 Subject 검사
    # ==============================================
    # unassigned:
    # 실제 데이터에는 있지만
    # Split 설정에는 없는 Subject입니다.
    #
    # missing_from_data:
    # Split 설정에는 있지만
    # Approved Manifest에는 없는 Subject입니다.
    #
    # 두 경우 모두 올바른 Split을 만들 수 없으므로
    # RuntimeError를 발생시킵니다.
    #
    # 두 집합이 같다면 다음 단계로 이동합니다.

    unassigned = (
        data_subjects
        - configured_subjects
    )

    missing_from_data = (
        configured_subjects
        - data_subjects
    )

    if unassigned:
        raise RuntimeError(
            "Split에 배정되지 않은 Subject: "
            + ", ".join(
                sorted(unassigned)
            )
        )

    if missing_from_data:
        raise RuntimeError(
            "Approved 데이터에 없는 "
            "Subject가 설정에 있습니다: "
            + ", ".join(
                sorted(
                    missing_from_data
                )
            )
        )

    # ==============================================
    # ⑰ Subject를 기준으로 세 가지 데이터셋 분리
    # ==============================================
    # isin():
    # DataFrame의 Subject가 지정한 목록에
    # 포함되어 있는지 검사합니다.
    #
    # train_df:
    # Train Subject의 모든 Clip을 선택합니다.
    #
    # val_df:
    # Validation Subject의 모든 Clip을 선택합니다.
    #
    # test_df:
    # Test Subject의 모든 Clip을 선택합니다.
    #
    # Clip을 무작위로 섞는 것이 아닙니다.
    #
    # 같은 Subject의 Clip 전체가
    # 하나의 Split에 들어가도록 처리합니다.

    train_df = df[
        df["subject"].isin(
            train_subjects
        )
    ].copy()

    val_df = df[
        df["subject"].isin(
            val_subjects
        )
    ].copy()

    test_df = df[
        df["subject"].isin(
            test_subjects
        )
    ].copy()

    # ==============================================
    # ⑱ Train·Validation·Test의 Action 구성 검사
    # ==============================================
    # require_action_coverage()를 세 번 호출합니다.
    #
    # TRAIN에 필요한 Action이 모두 있는가?
    #
    # VALIDATION에 필요한 Action이 모두 있는가?
    #
    # TEST에 필요한 Action이 모두 있는가?
    #
    # 어떤 Split이 비어 있거나
    # 필수 Action이 누락되면 프로그램을 중단합니다.
    #
    # 세 가지 검사를 모두 통과해야 CSV를 저장합니다.

    require_action_coverage(
        "TRAIN",
        train_df,
        expected_actions,
    )

    require_action_coverage(
        "VALIDATION",
        val_df,
        expected_actions,
    )

    require_action_coverage(
        "TEST",
        test_df,
        expected_actions,
    )

    # ==============================================
    # ⑲ Split CSV 저장 폴더 생성
    # ==============================================
    # 출력 폴더를 확인합니다.
    #
    # 폴더가 없으면 상위 폴더까지 생성합니다.
    #
    # 이미 폴더가 존재하면 그대로 사용합니다.
    #
    # 저장 위치:
    # data/day05/splits/

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # ⑳ 출력 파일명과 Split 데이터 연결
    # ==============================================
    # 각 Split의 이름을 DataFrame과 연결합니다.
    #
    # train → train_df
    # val   → val_df
    # test  → test_df
    #
    # 이를 이용하여 반복문 하나로
    # 세 개의 CSV를 저장할 수 있습니다.

    outputs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    # ==============================================
    # ㉑ 세 가지 Split CSV 저장 및 완료 메시지 출력
    # ==============================================
    # outputs.items()를 이용하여
    # 각 파일명과 데이터를 하나씩 가져옵니다.
    #
    # train_df → train.csv
    # val_df   → val.csv
    # test_df  → test.csv
    #
    # index=False:
    # Pandas 행 인덱스는 CSV에 저장하지 않습니다.
    #
    # utf-8-sig:
    # Excel에서도 한글을 읽기 쉬운 형식으로 저장합니다.
    #
    # 주의:
    # 기존 같은 이름의 CSV 파일은 덮어씁니다.
    #
    # 모든 파일을 저장하면
    # Subject-wise split: COMPLETE를 출력합니다.

    for name, split_df in (
        outputs.items()
    ):
        path = (
            output_dir
            / f"{name}.csv"
        )

        split_df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Saved: {path}")

    print()
    print(
        "Subject-wise split: COMPLETE"
    )


# ==================================================
# ㉒ Python 파일 직접 실행 시 main() 호출
# ==================================================
# 이 Python 파일을 직접 실행하면 main()을 호출합니다.
#
# 실제 처리 순서는 main()에서 시작합니다.
#
# 앞에서 정의한 함수들은 필요한 시점에 호출합니다.

if __name__ == "__main__":
    main()