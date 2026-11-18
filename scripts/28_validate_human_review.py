
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse : 터미널 실행 옵션을 입력받습니다.
# Path     : Human QA CSV의 파일 경로를 관리합니다.
# pandas   : CSV를 읽고 판정 상태를 검사합니다.

import argparse
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 Human QA 기본 파일 경로 설정
# ==================================================
# 이 스크립트가 scripts 폴더에 있다고 가정합니다.
#
# parents[1]은 scripts의 상위인 프로젝트 ROOT입니다.
#
# 27번 코드에서 생성한 Review CSV의
# 기본 위치를 지정합니다.

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REVIEW_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_review.csv"
)


# ==================================================
# ③ 허용되는 Human QA 최종 판정값 정의
# ==================================================
# PASS:
# 해당 Clip을 사용할 수 있다고 판단한 상태입니다.
#
# RESHOOT:
# 재촬영이 필요하다고 판단한 상태입니다.
#
# EXCLUDE:
# 이번 데이터셋에서 제외하기로 판단한 상태입니다.
#
# UNREVIEWED는 아직 검토하지 않은 상태이므로
# 최종 판정값 목록에 포함하지 않습니다.

ALLOWED = {
    "PASS",
    "RESHOOT",
    "EXCLUDE",
}


# ==================================================
# ④ 터미널 실행 옵션 설정
# ==================================================
# --review-path:
# 검사할 Human QA 작업표의 경로입니다.
#
# --expected-clips:
# 검사할 예상 Clip 수입니다.
#
# 기본 본수업에서는 18개 Clip을 사용합니다.
#
# 별도 옵션을 입력하지 않으면
# 기본 경로와 예상 개수 18을 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--review-path",
        default=str(
            DEFAULT_REVIEW_PATH
        ),
    )

    parser.add_argument(
        "--expected-clips",
        type=int,
        default=18,
    )

    return parser.parse_args()


# ==================================================
# ⑤ main() 실행 옵션 읽기 및 Review CSV 경로 준비
# ==================================================
# main()은 Human QA 입력 상태를 검사하는 함수입니다.
#
# parse_args()를 실행하여 터미널 옵션을 읽습니다.
#
# review_path:
# 사람이 PASS, RESHOOT, EXCLUDE를 기록한
# CSV 파일의 위치입니다.
#
# Path()를 사용하여 문자열 경로를
# 파일 경로 객체로 변환합니다.

def main() -> None:
    args = parse_args()

    review_path = Path(
        args.review_path
    )

    # ==============================================
    # ⑥ Human QA 파일 존재 여부 확인
    # ==============================================
    # 먼저 Review CSV가 실제로 존재하는지 확인합니다.
    #
    # 파일이 없다면:
    # [ERROR] 메시지를 출력합니다.
    #
    # 찾을 수 없는 파일 경로도 출력합니다.
    #
    # SystemExit(1):
    # 프로그램이 오류 상태로 종료되도록 합니다.
    #
    # 파일이 정상적으로 존재하면
    # 다음 단계에서 CSV를 읽습니다.

    if not review_path.exists():
        print(
            "[ERROR] Human QA 파일을 찾을 수 없습니다."
        )
        print(review_path)
        raise SystemExit(1)

    # ==============================================
    # ⑦ Review CSV 읽기 및 전체 Clip 개수 검사
    # ==============================================
    # pd.read_csv():
    # CSV 파일을 Pandas DataFrame으로 읽습니다.
    #
    # len(df):
    # Review CSV에 기록된 전체 Row 수입니다.
    #
    # 이번 수업에서는 18개 Clip이 있어야 합니다.
    #
    # 예상 개수와 실제 개수가 다르면
    # 오류 메시지를 출력합니다.
    #
    # 예:
    # expected=18, actual=17
    #
    # 개수가 다르면 더 이상 검사하지 않고 종료합니다.
    #
    # 이 코드는 행 개수를 확인하는 것이며
    # Subject·Action이 모두 존재하는지까지
    # 검사하는 코드는 아닙니다.

    df = pd.read_csv(
        review_path
    )

    if len(df) != args.expected_clips:
        print(
            "[ERROR] Human QA Clip 수가 "
            "예상과 다릅니다."
        )
        print(
            f"expected={args.expected_clips}, "
            f"actual={len(df)}"
        )
        raise SystemExit(1)

    # ==============================================
    # ⑧ human_decision 값을 비교하기 쉽게 정리
    # ==============================================
    # human_decision 컬럼을 가져옵니다.
    #
    # fillna(""):
    # 비어 있는 값을 빈 문자열로 변환합니다.
    #
    # astype(str):
    # 모든 값을 문자열로 변환합니다.
    #
    # str.strip():
    # 문자열의 앞뒤 공백을 제거합니다.
    #
    # str.upper():
    # 영문 소문자를 대문자로 변환합니다.
    #
    # 예:
    # " pass " → "PASS"
    #
    # 이렇게 하면 공백이나 대소문자 차이 때문에
    # 올바른 판정이 잘못 처리되는 것을 줄일 수 있습니다.
    #
    # 원본 CSV 파일 자체를 수정하지는 않습니다.

    decisions = (
        df["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ==============================================
    # ⑨ 아직 검토하지 않은 Clip 찾기
    # ==============================================
    # human_decision이 다음 중 하나인지 검사합니다.
    #
    # 빈 문자열 ""
    # UNREVIEWED
    #
    # 두 상태는 아직 최종 판정이 없다는 의미입니다.
    #
    # | 기호:
    # 두 조건 중 하나만 참이어도 True입니다.
    #
    # unreviewed_mask:
    # 각 Row가 미검토 상태인지 기록한 Boolean 데이터입니다.
    #
    # 주의:
    # 현재 코드는 qa_status가 AUTO_PASS인 Clip도
    # UNREVIEWED이면 미검토로 판단합니다.
    #
    # 즉 CHECK 영상만 검사하는 코드가 아닙니다.

    unreviewed_mask = (
        (decisions == "")
        | (decisions == "UNREVIEWED")
    )

    # ==============================================
    # ⑩ 미검토 Clip 목록 출력 및 프로그램 중단
    # ==============================================
    # any():
    # 미검토 상태인 Clip이 하나라도 있는지 확인합니다.
    #
    # 미검토 Clip이 존재하면 다음 정보를 가져옵니다.
    #
    # clip_id:
    # 어떤 Clip인지 나타내는 ID입니다.
    #
    # qa_status:
    # 27번 자동 QA 결과입니다.
    #
    # human_decision:
    # 사람이 작성한 최종 판정 상태입니다.
    #
    # 미검토 Clip 목록과 개수를 출력합니다.
    #
    # 이후 PASS / RESHOOT / EXCLUDE 중
    # 하나를 입력하도록 안내합니다.
    #
    # 미검토 상태가 남아 있다면
    # SystemExit(1)로 프로그램을 종료합니다.

    if unreviewed_mask.any():
        unreviewed = df.loc[
            unreviewed_mask,
            [
                "clip_id",
                "qa_status",
                "human_decision",
            ],
        ]

        print()
        print(
            "[HUMAN QA NOT COMPLETE]"
        )
        print()
        print(
            unreviewed.to_string(
                index=False
            )
        )

        print()
        print(
            f"남은 Clip: "
            f"{int(unreviewed_mask.sum())}"
        )

        print()
        print(
            "원본 MP4를 확인한 뒤 "
            "human_decision을"
        )
        print(
            "PASS / RESHOOT / EXCLUDE 중 "
            "하나로 입력하세요."
        )

        raise SystemExit(1)

    # ==============================================
    # ⑪ 허용되지 않은 Human QA 판정값 검사
    # ==============================================
    # 모든 Clip에 판정값이 입력되었다면
    # 이번에는 입력한 값이 올바른지 확인합니다.
    #
    # isin(ALLOWED):
    # PASS / RESHOOT / EXCLUDE 중
    # 하나에 해당하는지 검사합니다.
    #
    # ~ 기호:
    # True와 False를 반대로 바꿉니다.
    #
    # 따라서 invalid_mask=True는
    # 허용되지 않은 값을 의미합니다.
    #
    # 예:
    # APPROVED
    # PENDING
    # PASSX
    #
    # 이러한 값은 최종 판정으로 사용할 수 없습니다.

    invalid_mask = (
        ~decisions.isin(ALLOWED)
    )

    # ==============================================
    # ⑫ 잘못된 판정값 출력 및 프로그램 중단
    # ==============================================
    # invalid_mask에 True가 있다면
    # 잘못된 판정이 입력된 Clip을 선택합니다.
    #
    # 다음 정보를 출력합니다.
    #
    # clip_id
    # qa_status
    # human_decision
    #
    # 사용자에게 허용된 세 가지 판정값을 안내합니다.
    #
    # 잘못된 값이 있다면 SystemExit(1)로 종료합니다.
    #
    # 모든 판정이 올바르면 다음 단계로 이동합니다.

    if invalid_mask.any():
        invalid = df.loc[
            invalid_mask,
            [
                "clip_id",
                "qa_status",
                "human_decision",
            ],
        ]

        print()
        print(
            "[INVALID HUMAN DECISION]"
        )
        print()
        print(
            invalid.to_string(
                index=False
            )
        )

        print()
        print(
            "human_decision은 "
            "PASS / RESHOOT / EXCLUDE만 "
            "사용할 수 있습니다."
        )

        raise SystemExit(1)

    # ==============================================
    # ⑬ 최종 판정별 개수 및 검사 완료 결과 출력
    # ==============================================
    # 여기까지 왔다면 모든 Clip의 human_decision에
    # 허용된 최종 판정값이 입력된 상태입니다.
    #
    # value_counts():
    # 판정별 개수를 계산합니다.
    #
    # 예:
    # PASS       16
    # RESHOOT     1
    # EXCLUDE     1
    #
    # Reviewed clips:
    # 입력 검사를 통과한 전체 Clip 개수입니다.
    #
    # Human review: OK:
    # Review CSV의 판정 누락 및 값 검사를
    # 정상적으로 마쳤다는 의미입니다.
    #
    # 실제 영상을 사람이 시청했는지 여부까지
    # 프로그램이 확인하는 것은 아닙니다.

    print()
    print("[Human QA Result]")
    print(
        decisions.value_counts()
    )

    print()
    print(
        f"Reviewed clips: {len(df)}"
    )
    print(
        "Human review: OK"
    )


# ==================================================
# ⑭ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
#
# main()에서 실제 Human QA 검사를 진행합니다.
#
# 모든 판정 검사를 통과해야
# Human review: OK 메시지가 출력됩니다.

if __name__ == "__main__":
    main()