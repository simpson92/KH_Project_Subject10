
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse : 터미널 실행 옵션을 처리합니다.
# csv      : 자동 QA 결과를 CSV 파일로 저장합니다.
# sys      : 프로젝트 내부 모듈을 불러옵니다.
# Path     : 파일과 폴더 경로를 관리합니다.
# NumPy    : 관절 Confidence의 수치 계산에 사용합니다.
# Pandas   : Keypoint CSV를 읽고 통계를 계산합니다.

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 공통 설정 함수 연결
# ==================================================
# 현재 Python 파일은 scripts 폴더에 있습니다.
#
# parents[1]을 이용하여 프로젝트 ROOT를 찾습니다.
#
# sys.path.insert()로 ROOT를 모듈 검색 경로에 추가합니다.
#
# src.common의 load_config()를 불러옵니다.
# 이 함수는 프로젝트 설정값을 읽을 때 사용합니다.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config


# ==================================================
# ③ Manifest 입력 경로와 QA 출력 경로 설정
# ==================================================
# DEFAULT_MANIFEST_PATH:
# 26번 코드에서 생성한 전체 Keypoint Dataset 목록입니다.
#
# DEFAULT_QA_PATH:
# 27번 코드에서 자동 계산한 QA 결과를 저장합니다.
#
# DEFAULT_REVIEW_PATH:
# 사람이 직접 검토할 Human QA 작업표입니다.
#
# 두 출력 파일은 처음 생성할 때 같은 내용을 갖지만,
# 이후 Review 파일에는 Human QA 결과를 기록합니다.

DEFAULT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "keypoint_dataset_manifest.csv"
)

DEFAULT_QA_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_qa.csv"
)

DEFAULT_REVIEW_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_review.csv"
)


# ==================================================
# ④ Action별 중요 Keypoint 정의
# ==================================================
# 모든 동작에서 같은 관절이 중요한 것은 아닙니다.
#
# bend_return:
# Shoulder, Hip, Knee를 확인합니다.
#
# leg_raise_lower:
# Hip, Knee, Ankle을 확인합니다.
#
# walk_turn_walk:
# Shoulder, Hip, Knee, Ankle을 확인합니다.
#
# drink_return:
# Shoulder, Elbow, Wrist를 확인합니다.
#
# sit_stand:
# Shoulder, Hip, Knee, Ankle을 확인합니다.
#
# wave:
# Shoulder, Elbow, Wrist를 확인합니다.
#
# 이후 각 Action의 중요 관절 Confidence를
# 별도로 검사합니다.

ACTION_CRITICAL_KEYPOINTS = {
    "bend_return": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    ],
    "leg_raise_lower": [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "walk_turn_walk": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "drink_return": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ],
    "sit_stand": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "wave": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ],
}


# ==================================================
# ⑤ 터미널 실행 옵션 설정
# ==================================================
# --manifest-path:
# 분석할 Keypoint Dataset Manifest 경로입니다.
#
# --qa-path:
# 자동 QA 결과 CSV 저장 경로입니다.
#
# --review-path:
# Human QA 작업표 CSV 저장 경로입니다.
#
# --expected-clips:
# 검사할 Clip의 예상 개수입니다.
#
# 기본값은 18개입니다.
#
# 별도 옵션을 입력하지 않으면 기본 설정을 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-path",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
    )

    parser.add_argument(
        "--qa-path",
        default=str(
            DEFAULT_QA_PATH
        ),
    )

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
# ⑥ Clip 하나의 Keypoint CSV 읽기 및 빈 파일 검사
# ==================================================
# inspect_keypoint_csv()는 Clip 하나를 검사하는 함수입니다.
#
# csv_path:
# 분석할 Keypoint CSV 파일입니다.
#
# action_label:
# 현재 Clip의 동작 이름입니다.
#
# threshold:
# 낮은 Keypoint Confidence를 판단할 기준입니다.
#
# CSV에 데이터가 없다면 CHECK를 반환합니다.
#
# 빈 데이터는 정상 학습 데이터로 판단할 수 없으므로
# AUTO_PASS로 처리하지 않습니다.

def inspect_keypoint_csv(
    csv_path: Path,
    action_label: str,
    threshold: float,
) -> dict:
    df = pd.read_csv(csv_path)

    if df.empty:
        return {
            "qa_status": "CHECK",
            "frame_count": 0,
            "pose_ratio": 0.0,
            "single_person_ratio": 0.0,
            "mean_keypoint_conf": 0.0,
            "low_critical_ratio": 1.0,
            "missing_pose_frames": 0,
            "critical_keypoints": "",
        }

    # ==============================================
    # ⑦ 현재 Action에서 중요한 관절 선택
    # ==============================================
    # action_label을 이용하여 중요 관절을 찾습니다.
    #
    # 예:
    # action_label = "wave"
    #
    # 중요 관절:
    # Shoulder / Elbow / Wrist
    #
    # Action이 딕셔너리에 없다면
    # 어떤 관절을 검사해야 하는지 알 수 없으므로
    # RuntimeError를 발생시킵니다.

    critical_names = (
        ACTION_CRITICAL_KEYPOINTS.get(
            action_label
        )
    )

    if critical_names is None:
        raise RuntimeError(
            "QA 중요 Keypoint 정의가 "
            f"없는 Action: {action_label}"
        )

    # ==============================================
    # ⑧ Frame 수와 Pose·사람 검출 정보를 숫자로 변환
    # ==============================================
    # 26번 코드에서는 Frame 하나가 CSV의 한 행입니다.
    #
    # 따라서 len(df)는 전체 Frame 수입니다.
    #
    # pose_detected:
    # 해당 Frame에서 Pose를 검출했는지 나타냅니다.
    #
    # person_count:
    # 해당 Frame에서 검출된 사람 수입니다.
    #
    # pd.to_numeric():
    # CSV의 값을 숫자로 변환합니다.
    #
    # errors="coerce":
    # 숫자로 변환할 수 없는 값은 NaN으로 만듭니다.
    #
    # fillna(0):
    # NaN을 0으로 채웁니다.

    frame_count = len(df)

    pose_series = (
        pd.to_numeric(
            df["pose_detected"],
            errors="coerce",
        )
        .fillna(0)
    )

    person_series = (
        pd.to_numeric(
            df["person_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    # ==============================================
    # ⑨ Pose 검출률과 단일 인물 검출률 계산
    # ==============================================
    # pose_ratio:
    # 전체 Frame 중 Pose가 검출된 비율입니다.
    #
    # pose_detected가 0 또는 1이므로
    # 평균값을 계산하면 성공 비율이 됩니다.
    #
    # single_person_ratio:
    # 전체 Frame 중 정확히 한 사람만 검출된 비율입니다.
    #
    # 예:
    # 전체 100 Frame 중 95 Frame에서 Pose 검출
    # → pose_ratio = 0.95
    #
    # 전체 100 Frame 중 80 Frame에서 한 사람 검출
    # → single_person_ratio = 0.80

    pose_ratio = float(
        pose_series.mean()
    )

    single_person_ratio = float(
        (person_series == 1).mean()
    )

    # ==============================================
    # ⑩ 전체 Keypoint 평균 Confidence 계산
    # ==============================================
    # CSV에서 이름이 _conf로 끝나는 컬럼을 찾습니다.
    #
    # selected_box_conf는 사람 BBox 신뢰도이므로
    # 관절 Confidence 계산에서 제외합니다.
    #
    # 관절 Confidence 값들을 숫자로 변환합니다.
    #
    # np.nanmean():
    # NaN을 제외한 유효한 숫자의 평균을 계산합니다.
    #
    # 유효한 숫자가 하나도 없다면
    # mean_keypoint_conf = 0.0으로 설정합니다.

    confidence_columns = [
        column
        for column in df.columns
        if column.endswith("_conf")
        and column
        != "selected_box_conf"
    ]

    confidence_values = (
        df[confidence_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    mean_keypoint_conf = (
        float(
            np.nanmean(
                confidence_values
            )
        )
        if np.isfinite(
            confidence_values
        ).any()
        else 0.0
    )

    # ==============================================
    # ⑪ 중요 Keypoint Confidence 컬럼 준비 및 확인
    # ==============================================
    # ⑦에서 선택한 중요 관절 이름에 _conf를 붙입니다.
    #
    # 예:
    # left_wrist → left_wrist_conf
    # right_wrist → right_wrist_conf
    #
    # 필요한 컬럼이 실제 CSV에 존재하는지 확인합니다.
    #
    # 빠진 컬럼이 있다면 RuntimeError를 발생시킵니다.

    critical_columns = [
        f"{name}_conf"
        for name in critical_names
    ]

    missing_columns = [
        column
        for column in critical_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "중요 Keypoint Confidence 열이 "
            "없습니다: "
            + ", ".join(
                missing_columns
            )
        )

    # ==============================================
    # ⑫ 중요 Keypoint Confidence를 숫자 배열로 변환
    # ==============================================
    # 현재 Action에서 중요한 관절의 Confidence만
    # 선택합니다.
    #
    # 숫자로 변환할 수 없는 값은 NaN으로 만듭니다.
    #
    # to_numpy():
    # Pandas DataFrame을 NumPy 배열로 변환합니다.
    #
    # 이후 기준값보다 낮은 Confidence를 계산합니다.

    critical_values = (
        df[critical_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    # ==============================================
    # ⑬ 중요 관절의 Low Confidence 비율 계산
    # ==============================================
    # np.isfinite():
    # NaN과 Inf가 아닌 유효한 숫자인지 검사합니다.
    #
    # valid_count:
    # 실제 계산에 사용할 수 있는 Confidence 개수입니다.
    #
    # low_count:
    # threshold보다 낮은 Confidence 개수입니다.
    #
    # low_critical_ratio:
    # 낮은 Confidence 수 / 유효한 Confidence 수
    #
    # 유효한 Confidence가 하나도 없다면
    # low_critical_ratio = 1.0으로 설정합니다.
    #
    # 주의:
    # 빈 Confidence 값은 이 비율의 분모에서 제외합니다.
    # Pose Missing은 별도의 pose_ratio로 검사합니다.

    valid_mask = np.isfinite(
        critical_values
    )

    valid_count = int(
        valid_mask.sum()
    )

    if valid_count > 0:
        low_count = int(
            (
                (
                    critical_values
                    < threshold
                )
                & valid_mask
            ).sum()
        )

        low_critical_ratio = (
            low_count / valid_count
        )
    else:
        low_critical_ratio = 1.0

    # ==============================================
    # ⑭ Pose Missing 계산 및 AUTO_PASS·CHECK 판정
    # ==============================================
    # pose_detected=0인 Frame 개수를 계산합니다.
    #
    # 먼저 qa_status를 AUTO_PASS로 설정합니다.
    #
    # 다음 조건 중 하나라도 해당하면 CHECK로 변경합니다.
    #
    # 1. pose_ratio < 0.90
    # 2. single_person_ratio < 0.90
    # 3. mean_keypoint_conf < 0.50
    # 4. low_critical_ratio > 0.20
    #
    # AUTO_PASS:
    # 교육용 자동 QA 기준을 통과한 상태입니다.
    #
    # CHECK:
    # 사람이 확인해야 할 가능성이 있는 상태입니다.
    #
    # AUTO_PASS가 Human QA의 최종 PASS는 아닙니다.

    missing_pose_frames = int(
        (pose_series == 0).sum()
    )

    qa_status = "AUTO_PASS"

    if pose_ratio < 0.90:
        qa_status = "CHECK"

    if single_person_ratio < 0.90:
        qa_status = "CHECK"

    if mean_keypoint_conf < 0.50:
        qa_status = "CHECK"

    if low_critical_ratio > 0.20:
        qa_status = "CHECK"

    # ==============================================
    # ⑮ Clip 하나의 QA 검사 결과 반환
    # ==============================================
    # 검사한 통계와 QA 판정값을 딕셔너리로 정리합니다.
    #
    # qa_status:
    # AUTO_PASS 또는 CHECK
    #
    # frame_count:
    # 전체 Frame 수
    #
    # pose_ratio:
    # Pose 검출 비율
    #
    # single_person_ratio:
    # 한 사람만 검출된 Frame 비율
    #
    # mean_keypoint_conf:
    # 전체 관절 Confidence 평균
    #
    # low_critical_ratio:
    # 중요 관절의 저신뢰도 비율
    #
    # missing_pose_frames:
    # Pose를 검출하지 못한 Frame 수
    #
    # critical_keypoints:
    # 현재 Action에서 중요하게 검사한 관절 목록

    return {
        "qa_status": qa_status,
        "frame_count": frame_count,
        "pose_ratio": round(
            pose_ratio,
            4,
        ),
        "single_person_ratio": round(
            single_person_ratio,
            4,
        ),
        "mean_keypoint_conf": round(
            mean_keypoint_conf,
            4,
        ),
        "low_critical_ratio": round(
            low_critical_ratio,
            4,
        ),
        "missing_pose_frames": (
            missing_pose_frames
        ),
        "critical_keypoints": "|".join(
            critical_names
        ),
    }


# ==================================================
# ⑯ main() 실행 옵션 읽기 및 경로 준비
# ==================================================
# main()은 Manifest에 등록된 모든 Clip을 처리합니다.
#
# 실행 옵션을 읽고 세 가지 경로를 설정합니다.
#
# manifest_path:
# 입력 Keypoint Dataset 목록
#
# qa_path:
# 자동 QA 결과 CSV
#
# review_path:
# Human QA 작업표 CSV

def main() -> None:
    args = parse_args()

    manifest_path = Path(
        args.manifest_path
    )

    qa_path = Path(
        args.qa_path
    )

    review_path = Path(
        args.review_path
    )

    # ==============================================
    # ⑰ 예상 Clip 수·입력 파일·Confidence 기준 확인
    # ==============================================
    # expected_clips가 1 미만이면 오류를 발생시킵니다.
    #
    # 입력 Manifest가 실제로 존재하는지 확인합니다.
    #
    # load_config()를 이용하여 프로젝트 설정을 읽습니다.
    #
    # keypoint_confidence:
    # 낮은 관절 Confidence를 판단할 기준입니다.
    #
    # 설정값이 없다면 0.50을 기본값으로 사용합니다.
    #
    # 이 값은 ⑬의 Low Confidence 검사에 사용됩니다.

    if args.expected_clips < 1:
        raise ValueError(
            "--expected-clips는 "
            "1 이상이어야 합니다."
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    config = load_config()

    threshold = float(
        config.get(
            "keypoint_confidence",
            0.50,
        )
    )

    # ==============================================
    # ⑱ Manifest 읽기 및 예상 Clip 개수 확인
    # ==============================================
    # 26번 코드에서 생성한 Manifest를 읽습니다.
    #
    # 이번 기본 수업에서는 18개 Clip이 있어야 합니다.
    #
    # 실제 Row 수와 expected_clips를 비교합니다.
    #
    # 개수가 다르다면 RuntimeError를 발생시킵니다.
    #
    # rows:
    # 모든 Clip의 QA 결과를 모을 목록입니다.

    manifest = pd.read_csv(
        manifest_path
    )

    if len(manifest) != args.expected_clips:
        raise RuntimeError(
            "Manifest Clip 수가 "
            "예상과 다릅니다. "
            f"expected={args.expected_clips}, "
            f"actual={len(manifest)}"
        )

    rows = []

    # ==============================================
    # ⑲ Manifest의 Clip별 Keypoint CSV 경로 확인
    # ==============================================
    # manifest.iterrows():
    # Manifest의 각 Clip을 하나씩 가져옵니다.
    #
    # keypoint_csv 컬럼에는 26번 코드에서 저장한
    # Clip별 CSV 경로가 들어 있습니다.
    #
    # ROOT와 상대 경로를 연결하여 실제 경로를 만듭니다.
    #
    # CSV 파일이 없다면 FileNotFoundError를 발생시킵니다.

    for _, item in (
        manifest.iterrows()
    ):
        csv_path = (
            ROOT
            / str(
                item["keypoint_csv"]
            )
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                csv_path
            )

        # ==========================================
        # ⑳ Clip별 자동 QA 실행 및 결과 행 생성
        # ==========================================
        # inspect_keypoint_csv()를 호출합니다.
        #
        # 현재 Clip의 Action에 맞는 중요 관절을 선택하고
        # QA 통계와 AUTO_PASS/CHECK를 계산합니다.
        #
        # 기존 Manifest 정보와 QA 결과를 합칩니다.
        #
        # **qa:
        # QA 딕셔너리의 내용을 현재 row에 추가합니다.
        #
        # human_decision:
        # 아직 사람이 검토하지 않았으므로 UNREVIEWED입니다.
        #
        # human_note:
        # 사람이 검토한 이유를 기록할 빈 칸입니다.

        qa = inspect_keypoint_csv(
            csv_path=csv_path,
            action_label=str(
                item["action_label"]
            ),
            threshold=threshold,
        )

        row = {
            "clip_id": item["clip_id"],
            "subject": item["subject"],
            "action_id": item["action_id"],
            "action_label": (
                item["action_label"]
            ),
            "take": item["take"],
            "video_path": (
                item["video_path"]
            ),
            "keypoint_csv": (
                item["keypoint_csv"]
            ),
            **qa,
            "human_decision": (
                "UNREVIEWED"
            ),
            "human_note": "",
        }

        # ==========================================
        # ㉑ QA 결과 저장 및 검사 진행 상황 출력
        # ==========================================
        # 현재 Clip의 QA 결과를 rows에 추가합니다.
        #
        # 터미널에 다음 정보를 표시합니다.
        #
        # Clip ID
        # QA 상태
        # Pose 검출률
        # 평균 Keypoint Confidence
        # 중요 Keypoint 저신뢰도 비율
        #
        # 다음 Clip이 있으면 같은 작업을 반복합니다.

        rows.append(row)

        print(
            f"{item['clip_id']:16s} "
            f"{qa['qa_status']:9s} "
            f"pose={qa['pose_ratio']:.3f} "
            f"conf="
            f"{qa['mean_keypoint_conf']:.3f} "
            f"low="
            f"{qa['low_critical_ratio']:.3f}"
        )

    # ==============================================
    # ㉒ QA·Review 결과 폴더와 CSV 컬럼 준비
    # ==============================================
    # QA와 Review 파일이 저장될 폴더를 생성합니다.
    #
    # parents=True:
    # 상위 폴더까지 필요한 경우 자동으로 생성합니다.
    #
    # exist_ok=True:
    # 이미 폴더가 있어도 오류를 발생시키지 않습니다.
    #
    # rows[0].keys():
    # 첫 번째 결과 딕셔너리의 Key를 CSV 컬럼명으로 사용합니다.

    qa_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    # ==============================================
    # ㉓ 자동 QA CSV와 Human QA 작업표 저장
    # ==============================================
    # 두 개의 CSV 파일에 같은 초기 결과를 저장합니다.
    #
    # keypoint_clip_qa.csv:
    # 자동 QA 결과 확인용
    #
    # keypoint_clip_review.csv:
    # 사람이 PASS / RESHOOT / EXCLUDE 등을 기록할 작업표
    #
    # utf-8-sig로 저장하여 Excel에서도 한글을
    # 읽기 쉽게 합니다.
    #
    # 주의:
    # 기존 Review 파일이 있어도 "w" 모드로 덮어씁니다.
    # 이 코드를 재실행하면 기존 Human QA 입력 내용이
    # 초기화될 수 있으므로 검수 이후에는 주의해야 합니다.

    for output_path in [
        qa_path,
        review_path,
    ]:
        with output_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    # ==============================================
    # ㉔ 결과 CSV 저장 경로 출력
    # ==============================================
    # 자동 QA 결과와 Human QA 작업표의
    # 실제 저장 경로를 터미널에 출력합니다.
    #
    # 이후 학생들은 Review CSV를 열어
    # 사람이 확인해야 할 Clip을 검토합니다.

    print()
    print(f"QA     : {qa_path}")
    print(f"Review : {review_path}")


# ==================================================
# ㉕ Python 파일 직접 실행 시 main() 호출
# ==================================================
# 터미널에서 이 파일을 실행하면 main()이 호출됩니다.
#
# main()에서 Manifest를 읽고 Clip별 QA를 진행한 뒤
# 자동 QA 결과와 Human QA 작업표를 저장합니다.

if __name__ == "__main__":
    main()