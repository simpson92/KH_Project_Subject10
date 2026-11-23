from __future__ import annotations

# ====================================================
# ① 필요한 라이브러리 불러오기
# ====================================================
# argparse는 터미널에서 입력한 옵션을 읽습니다.
# Path는 파일과 폴더 경로를 관리합니다.
# pandas는 47번의 Tracking CSV를 읽고 통계표를 만듭니다.

import argparse
from pathlib import Path

import pandas as pd


# ====================================================
# ② 프로젝트 ROOT 경로 찾기
# ====================================================
# 현재 실행 파일이 scripts 폴더 안에 있다고 가정합니다.
# parents[1]은 scripts의 상위인 프로젝트 폴더입니다.

ROOT = Path(__file__).resolve().parents[1]

# ====================================================
# ③ Track 분석 결과의 기본 저장 폴더 지정
# ====================================================
# 사람별 Tracking 결과를 요약한 CSV 2개를 저장할 폴더입니다.
# 기본 저장 위치는 reports/day08/logs입니다.

DEFAULT_OUTPUT_DIR = (
    ROOT / "reports" / "day08" / "logs"
)


# ====================================================
# ④ 실행 옵션을 읽는 parse_args 함수 작성
# ====================================================
# 명령행에서 입력 Tracking Log와 결과 폴더를 지정합니다.
# ArgumentParser는 --log, --output-dir 옵션을 해석합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    # ====================================================
    # ⑤ 분석할 Tracking 로그 --log 옵션 지정
    # ====================================================
    # --log는 필수 입력값입니다.
    # 예: --log reports/day08/logs/01_sbu_shaking_hands_tracking.csv

    parser.add_argument(
        "--log",
        required=True,
    )

    # ====================================================
    # ⑥ 결과 폴더 --output-dir 옵션 및 반환
    # ====================================================
    # --output-dir를 생략하면 기본 저장 폴더를 사용합니다.
    # parse_args()로 사용자가 입력한 옵션을 반환합니다.

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    return parser.parse_args()


# ====================================================
# ⑦ main 함수 시작 및 실행 옵션 읽기
# ====================================================
# parse_args()를 호출하여 분석할 입력 로그와 출력 폴더를 확인합니다.

def main() -> None:
    args = parse_args()

    # ====================================================
    # ⑧ 입력 로그 및 출력 폴더를 Path 객체로 변환
    # ====================================================
    # 입력한 경로 문자열을 Path로 변환합니다.
    # 경로 객체로 만들면 ROOT와 경로를 쉽게 결합할 수 있습니다.

    log_path = Path(args.log)
    output_dir = Path(
        args.output_dir
    )

    # ====================================================
    # ⑨ 입력·출력 상대경로를 ROOT 기준으로 변환
    # ====================================================
    # 상대경로로 입력했다면 프로젝트 ROOT를 앞에 붙입니다.
    # 절대경로로 입력한 경우에는 경로를 변경하지 않습니다.

    if not log_path.is_absolute():
        log_path = ROOT / log_path

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    # ====================================================
    # ⑩ 입력 Tracking CSV 파일의 존재 여부 확인
    # ====================================================
    # 파일이 없으면 FileNotFoundError로 실행을 중단합니다.
    # 실습 전에 47번 Tracking Log가 생성되었는지 확인합니다.

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    # ====================================================
    # ⑪ 47번 Tracking CSV 읽기
    # ====================================================
    # Pandas로 Frame별·사람별 추적 결과를 읽습니다.
    # 한 Frame에서 두 명이 추적되면 두 개 이상의 행이 있을 수 있습니다.

    df = pd.read_csv(log_path)

    # ====================================================
    # ⑫ Tracking Log가 비었는지 확인
    # ====================================================
    # CSV에 데이터 행이 하나도 없다면 계산할 내용이 없습니다.
    # RuntimeError를 발생시켜 잘못된 빈 분석을 방지합니다.

    if df.empty:
        raise RuntimeError("Tracking Log가 비어 있습니다.")

    # ====================================================
    # ⑬ 유효한 Track ID가 기록된 행만 선택
    # ====================================================
    # 47번 코드에서는 미추적 Frame을 track_id=-1로 기록합니다.
    # to_numeric(errors="coerce")은 비숫자를 NaN으로 바꿉니다.
    # fillna(-1) 후 0 이상의 ID를 가진 행만 남깁니다.

    valid = df[
        pd.to_numeric(
            df["track_id"],
            errors="coerce",
        ).fillna(-1) >= 0
    ].copy()

    # ====================================================
    # ⑭ 영상의 총 Frame 수 계산
    # ====================================================
    # 47번 코드의 Frame 번호는 0부터 시작합니다.
    # 따라서 최대 frame_id에 1을 더해 총 Frame 수를 구합니다.
    # 이 계산은 Frame 번호가 연속이고 0부터 시작한다는 전제입니다.

    total_frames = int(df["frame_id"].max()) + 1

    # ====================================================
    # ⑮ 유효한 Track ID가 전혀 없는 경우 처리
    # ====================================================
    # 사람 검출이 있었더라도 ID가 없으면 valid는 비게 됩니다.
    # 이때 추적 비율과 2-ID Frame 비율을 0으로 설정합니다.
    # 분석 결과를 출력할 수 있도록 빈 Track ID 요약표를 만듭니다.

    if valid.empty:
        tracked_frame_ratio = 0.0
        two_id_frame_ratio = 0.0
        max_ids_in_frame = 0
        unique_ids = []
        # ====================================================
        # ⑯ 미추적 영상의 빈 ID 요약표 컬럼 준비
        # ====================================================
        # 유효 Track ID가 없어도 결과 CSV의 컬럼 구조를 유지합니다.
        # Track ID, 시작·마지막 Frame, 검출 횟수, 평균 신뢰도 컬럼입니다.

        id_summary = pd.DataFrame(
            columns=[
                "track_id",
                "first_frame",
                "last_frame",
                "detected_frames",
                "mean_box_conf",
                "mean_keypoint_conf",
            ]
        )
    # ====================================================
    # ⑰ Frame마다 서로 다른 Track ID 개수 계산
    # ====================================================
    # valid를 frame_id로 묶고 nunique()로 고유 ID 수를 셉니다.
    # 예: Frame 10에 ID 1과 2가 있으면 2명으로 계산합니다.
    # 추적 ID가 없는 Frame은 이 결과에 포함되지 않습니다.

    else:
        ids_per_frame = (
            valid.groupby("frame_id")["track_id"]
            .nunique()
        )

        # ====================================================
        # ⑱ Track ID가 하나 이상 있는 Frame 비율 계산
        # ====================================================
        # tracked_frame_ratio = 추적 ID가 기록된 Frame 수 / 전체 Frame 수입니다.
        # 이 값은 ID가 1개 이상 나타난 Frame의 비율이지
        # 각 사람의 Tracking 성공률을 뜻하지는 않습니다.

        tracked_frame_ratio = (
            ids_per_frame.index.nunique()
            / total_frames
        )

        # ====================================================
        # ⑲ Track ID가 정확히 2개인 Frame 비율 계산
        # ====================================================
        # ids_per_frame == 2인 Frame 수를 세어 전체 Frame 수로 나눕니다.
        # 두 명 이상이 아니라 정확히 두 개의 Track ID가 있는 경우입니다.

        two_id_frame_ratio = (
            (ids_per_frame == 2).sum()
            / total_frames
        )

        # ====================================================
        # ⑳ 한 Frame에서 관찰된 최대 Track ID 수 확인
        # ====================================================
        # 각 Frame의 고유 Track ID 수 가운데 최댓값을 가져옵니다.
        # 다중 인물 추적에서 동시에 몇 개 ID가 나왔는지 확인합니다.

        max_ids_in_frame = int(ids_per_frame.max())

        # ====================================================
        # ㉑ 영상 전체에서 등장한 고유 Track ID 목록 수집
        # ====================================================
        # track_id를 정수로 변환하여 중복을 제거하고 정렬합니다.
        # 예: [1, 2, 5]는 전체 영상에서 세 종류의 ID가 등장한 상태입니다.
        # 고유 ID 수가 실제 등장한 사람 수와 반드시 같지는 않습니다.

        unique_ids = sorted(
            valid["track_id"]
            .astype(int)
            .unique()
            .tolist()
        )

        # ====================================================
        # ㉒ Track ID별 최초·최종 Frame과 신뢰도 요약
        # ====================================================
        # Track ID별로 그룹을 만듭니다.
        # min/max로 첫·마지막 Frame을 확인하고, nunique로 등장 Frame 수를 셉니다.
        # Box 및 Keypoint Confidence의 평균을 계산합니다.

        id_summary = (
            valid.groupby("track_id")
            .agg(
                first_frame=("frame_id", "min"),
                last_frame=("frame_id", "max"),
                detected_frames=("frame_id", "nunique"),
                mean_box_conf=("box_conf", "mean"),
                mean_keypoint_conf=(
                    "mean_keypoint_conf",
                    "mean",
                ),
            )
            .reset_index()
        )

    # ====================================================
    # ㉓ 입력 파일명에서 실습 Scenario 이름 추출
    # ====================================================
    # 확장자를 제거한 파일명에서 _tracking을 삭제합니다.
    # 예: 01_sbu_shaking_hands_tracking.csv → 01_sbu_shaking_hands

    scenario = log_path.stem.replace(
        "_tracking",
        "",
    )

    # ====================================================
    # ㉔ 영상 전체의 Tracking 요약 결과표 생성
    # ====================================================
    # 영상별로 한 행을 작성합니다.
    # 전체 Frame 수, 등장 ID 수와 목록, 추적 Frame 비율,
    # 정확히 두 ID인 Frame 비율, 최대 ID 수를 기록합니다.
    # extra_id_count는 max(0, 고유 ID 수-2)로 계산한 참고값이며
    # 실제 ID Switch 발생 횟수가 아닙니다.

    summary = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "total_frames": total_frames,
                "unique_track_ids": len(unique_ids),
                "track_ids": "|".join(
                    str(value) for value in unique_ids
                ),
                "tracked_frame_ratio": round(
                    tracked_frame_ratio,
                    4,
                ),
                "two_id_frame_ratio": round(
                    two_id_frame_ratio,
                    4,
                ),
                "max_ids_in_frame": max_ids_in_frame,
                "extra_id_count": max(
                    0,
                    len(unique_ids) - 2,
                ),
            }
        ]
    )

    # ====================================================
    # ㉕ 결과 CSV를 저장할 폴더 생성
    # ====================================================
    # 부모 폴더까지 없으면 자동 생성합니다.
    # 이미 폴더가 있으면 오류 없이 그대로 사용합니다.

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ====================================================
    # ㉖ 영상 전체 요약 CSV 파일명 결정
    # ====================================================
    # Scenario 이름 뒤에 _tracking_summary.csv를 붙입니다.
    # 한 영상에 대한 전체 Tracking 지표를 저장합니다.

    summary_path = (
        output_dir
        / f"{scenario}_tracking_summary.csv"
    )

    # ====================================================
    # ㉗ Track ID별 요약 CSV 파일명 결정
    # ====================================================
    # Scenario 이름 뒤에 _track_id_summary.csv를 붙입니다.
    # 각 Track ID의 등장 구간과 평균 Confidence를 저장합니다.

    id_path = (
        output_dir
        / f"{scenario}_track_id_summary.csv"
    )

    # ====================================================
    # ㉘ 영상 전체 Tracking 요약 CSV 저장
    # ====================================================
    # 행 인덱스를 제외하고 UTF-8-SIG로 저장합니다.
    # 파일이 이미 있다면 덮어씁니다.

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ====================================================
    # ㉙ Track ID별 요약 CSV 저장
    # ====================================================
    # 각 ID의 최초·마지막 Frame, 검출 Frame 수, Confidence를 기록합니다.
    # 추적 ID가 없는 경우에는 컬럼만 있는 빈 CSV를 저장합니다.

    id_summary.to_csv(
        id_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ====================================================
    # ㉚ 영상 전체 Tracking 요약표 출력
    # ====================================================
    # Pandas DataFrame을 화면에 표 형태로 출력합니다.
    # print()로 다음 출력과 사이를 띄웁니다.

    print(summary.to_string(index=False))
    print()

    # ====================================================
    # ㉛ Track ID별 요약표가 있으면 화면에 출력
    # ====================================================
    # 유효 ID가 존재할 때만 ID별 상세표를 출력합니다.
    # ID가 없는 경우에는 빈 표를 출력하지 않습니다.

    if not id_summary.empty:
        print(id_summary.to_string(index=False))
        print()

    # ====================================================
    # ㉜ ID Switch 오해를 방지하는 주의사항 출력
    # ====================================================
    # Unique ID가 3개 이상이어도 ID Switch가 확정되는 것은 아닙니다.
    # Track Lost 뒤 새 ID가 생긴 경우와 서로 다른 사람의 ID 교환은
    # 추적 결과 영상을 직접 확인해 구분해야 합니다.

    print(
        "주의: Unique ID가 3개 이상이라고 해서 "
        "그 자체로 ID Switch가 확정되는 것은 아닙니다."
    )
    print(
        "Track Lost 후 새 ID가 생성된 경우와 "
        "실제 ID Switch를 결과영상에서 구분하세요."
    )
    print()
    # ====================================================
    # ㉝ 두 결과 CSV의 저장 경로 출력
    # ====================================================
    # 영상 전체 요약과 ID별 요약 CSV의 위치를 출력합니다.
    # 실습 결과를 쉽게 찾아 확인할 수 있습니다.

    print(f"Saved: {summary_path}")
    print(f"Saved: {id_path}")


# ====================================================
# ㉞ Python 파일 직접 실행 시 main 함수 호출
# ====================================================
# python scripts/48_summarize_tracking_log.py처럼 직접 실행하면 main()을 호출합니다.
# 다른 파일에서 import하는 경우에는 자동으로 실행되지 않습니다.

if __name__ == "__main__":
    main()