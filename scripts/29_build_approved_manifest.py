
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse : 터미널에서 실행 옵션을 입력받습니다.
# Path     : 입력·출력 파일 경로를 관리합니다.
# pandas   : Human QA CSV를 읽고 PASS 데이터만 선택합니다.

import argparse
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 기본 입력·출력 경로 설정
# ==================================================
# ROOT는 scripts 폴더의 상위인 프로젝트 루트입니다.
#
# DEFAULT_REVIEW_PATH:
# Human QA 판정이 기록된 작업표입니다.
#
# DEFAULT_OUTPUT_PATH:
# 다음 단계에서 사용할 승인 Clip 목록의
# 저장 경로입니다.

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REVIEW_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_review.csv"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "approved_clip_manifest.csv"
)


# ==================================================
# ③ 터미널 실행 옵션 설정
# ==================================================
# --review-path:
# Human QA CSV 입력 경로입니다.
#
# --output-path:
# Approved Manifest CSV 출력 경로입니다.
#
# 두 옵션을 생략하면 기본 경로를 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--review-path",
        default=str(
            DEFAULT_REVIEW_PATH
        ),
    )

    parser.add_argument(
        "--output-path",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


# ==================================================
# ④ main() 실행 옵션과 파일 경로 준비
# ==================================================
# parse_args()를 호출하여 터미널 옵션을 읽습니다.
#
# Path()를 사용하여 입력받은 경로 문자열을
# 파일 경로 객체로 변환합니다.
#
# review_path:
# 읽을 Human QA 작업표입니다.
#
# output_path:
# 저장할 Approved Manifest입니다.

def main() -> None:
    args = parse_args()

    review_path = Path(
        args.review_path
    )

    output_path = Path(
        args.output_path
    )

    # ==============================================
    # ⑤ Human QA 파일 존재 여부 확인
    # ==============================================
    # keypoint_clip_review.csv가 실제로 존재하는지
    # 확인합니다.
    #
    # 파일이 존재하지 않으면
    # FileNotFoundError를 발생시킵니다.
    #
    # 파일이 있다면 다음 단계로 이동합니다.

    if not review_path.exists():
        raise FileNotFoundError(
            review_path
        )

    # ==============================================
    # ⑥ Human QA 작업표 CSV 읽기
    # ==============================================
    # Pandas의 read_csv()로 CSV 파일을 읽습니다.
    #
    # 각 Row에는 하나의 Clip 정보가 들어 있습니다.
    #
    # 주요 컬럼:
    # clip_id
    # subject
    # action_label
    # qa_status
    # human_decision
    #
    # 이번 코드에서는 human_decision을 이용해
    # 승인된 Clip을 선택합니다.

    df = pd.read_csv(
        review_path
    )

    # ==============================================
    # ⑦ Human QA 판정값 정리
    # ==============================================
    # human_decision 컬럼을 가져옵니다.
    #
    # fillna(""):
    # 비어 있는 값을 빈 문자열로 변환합니다.
    #
    # astype(str):
    # 값을 문자열로 변환합니다.
    #
    # str.strip():
    # 앞뒤 공백을 제거합니다.
    #
    # str.upper():
    # 영문을 대문자로 변환합니다.
    #
    # 예:
    # " pass " → "PASS"
    #
    # 원본 CSV 자체를 수정하지는 않습니다.

    decisions = (
        df["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ==============================================
    # ⑧ Human Decision이 PASS인 Clip만 선택
    # ==============================================
    # decisions == "PASS"인 행만 선택합니다.
    #
    # PASS:
    # Approved Manifest에 포함합니다.
    #
    # RESHOOT:
    # 재촬영이 필요하므로 포함하지 않습니다.
    #
    # EXCLUDE:
    # 제외 대상으로 분류되어 포함하지 않습니다.
    #
    # UNREVIEWED:
    # 아직 최종 판정이 없으므로 포함하지 않습니다.
    #
    # copy():
    # 선택된 데이터를 별도의 DataFrame으로 만듭니다.
    #
    # 주의:
    # 현재 코드는 qa_status가 AUTO_PASS여도
    # human_decision이 UNREVIEWED이면 제외합니다.

    approved = df[
        decisions == "PASS"
    ].copy()

    # ==============================================
    # ⑨ 승인된 Clip이 하나도 없는지 확인
    # ==============================================
    # approved.empty:
    # 선택된 데이터가 비어 있는지 확인합니다.
    #
    # PASS로 판정된 Clip이 하나도 없다면
    # RuntimeError를 발생시킵니다.
    #
    # 하나 이상 있다면 CSV 저장을 진행합니다.

    if approved.empty:
        raise RuntimeError(
            "PASS 데이터가 없습니다."
        )

    # ==============================================
    # ⑩ Approved Manifest 출력 폴더 생성
    # ==============================================
    # output_path.parent:
    # 저장 파일의 상위 폴더를 의미합니다.
    #
    # mkdir():
    # 폴더를 생성합니다.
    #
    # parents=True:
    # 필요한 상위 폴더도 함께 생성합니다.
    #
    # exist_ok=True:
    # 폴더가 이미 있어도 오류를 발생시키지 않습니다.

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # ⑪ PASS Clip만 Approved Manifest CSV로 저장
    # ==============================================
    # approved DataFrame을 새로운 CSV로 저장합니다.
    #
    # index=False:
    # DataFrame의 행 인덱스를 CSV에 저장하지 않습니다.
    #
    # utf-8-sig:
    # Excel에서 한글을 읽기 쉬운 형식입니다.
    #
    # 저장되는 파일:
    # approved_clip_manifest.csv
    #
    # 원본 Review CSV는 변경하지 않습니다.
    #
    # 다만 Approved Manifest 파일이 이미 있으면
    # 해당 출력 파일은 덮어씁니다.

    approved.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ==============================================
    # ⑫ 전체 승인 Clip 개수 출력
    # ==============================================
    # len(approved):
    # PASS로 선택된 전체 Clip 개수입니다.
    #
    # 예:
    # Approved clips: 18
    #
    # 실제 결과는 Human QA 판정에 따라 달라집니다.

    print(
        f"Approved clips: "
        f"{len(approved)}"
    )

    # ==============================================
    # ⑬ Subject별 승인 Clip 개수 계산 및 출력
    # ==============================================
    # groupby("subject"):
    # Subject가 같은 데이터를 그룹으로 묶습니다.
    #
    # size():
    # 각 그룹의 Row 수를 계산합니다.
    #
    # 예:
    # S01 → 6개
    # S02 → 6개
    # S03 → 6개
    #
    # 이를 통해 특정 Subject의 데이터가
    # 부족하지 않은지 확인할 수 있습니다.
    #
    # 현재 코드는 개수를 출력할 뿐,
    # Subject마다 6개인지 강제로 검증하지는 않습니다.

    print()
    print("[By Subject]")
    print(
        approved.groupby(
            "subject"
        ).size()
    )

    # ==============================================
    # ⑭ Action별 승인 Clip 개수 계산 및 출력
    # ==============================================
    # groupby("action_label"):
    # 같은 동작의 데이터를 그룹으로 묶습니다.
    #
    # size():
    # 각 Action에 포함된 Clip 개수를 계산합니다.
    #
    # 이번 기본 데이터가 모두 승인되었다면
    # 여섯 Action 각각 3개가 됩니다.
    #
    # 특정 Action 데이터가 부족하다면
    # 이후 Split 전에 확인해야 합니다.
    #
    # 현재 코드는 Action별 개수 역시 출력만 합니다.

    print()
    print("[By Action]")
    print(
        approved.groupby(
            "action_label"
        ).size()
    )

    # ==============================================
    # ⑮ Approved Manifest 저장 경로 출력
    # ==============================================
    # 저장한 CSV 파일의 위치를 출력합니다.
    #
    # 이후 Subject-wise Split 단계에서는
    # 이 Approved Manifest를 입력으로 사용합니다.

    print()
    print(f"Saved: {output_path}")


# ==================================================
# ⑯ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# main()에서 Human QA CSV를 읽고,
# PASS 데이터를 선택한 뒤,
# Approved Manifest로 저장합니다.

if __name__ == "__main__":
    main()