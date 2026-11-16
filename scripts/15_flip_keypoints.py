from __future__ import annotations

# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
# argparse : 실행 옵션 처리
# csv      : 비교 결과를 CSV 파일로 저장
# sys      : Python 모듈 검색 경로 설정
# Path     : 파일 및 폴더 경로 처리
# numpy    : 관절 좌표 차이와 거리 계산
# YOLO     : 이미지에서 Keypoint 추출

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ==================================================
# [02] 프로젝트 ROOT 경로 설정
# ==================================================
# 현재 스크립트는 scripts 폴더에 있습니다.
#
# parents[1]로 프로젝트 최상위 폴더를 찾고,
# src 폴더의 공통 함수를 가져올 수 있도록 합니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ==================================================
# [03] 다른 파일의 공통 함수 가져오기
# ==================================================
# common.py
# → 설정 파일 읽기와 CPU/GPU 확인
#
# feature_utils.py
# → 골반 중심 및 어깨 폭 정규화
# → Keypoint의 좌우 반전
#
# pose_utils.py
# → COCO 17 Keypoint의 관절 이름 확인

from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    flip_keypoints_horizontal,
    normalize_by_shoulder_width,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# [04] 실행 옵션 정의
# ==================================================
# --left
# → 왼팔을 올린 이미지 경로
#
# --right
# → 오른팔을 올린 이미지 경로
#
# 옵션을 지정하지 않으면
# 3일차에 준비한 기본 이미지를 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    # 왼팔을 올린 이미지
    parser.add_argument(
        "--left",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "left_arm_up.jpg"
        ),
    )

    # 오른팔을 올린 이미지
    parser.add_argument(
        "--right",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "right_arm_up.jpg"
        ),
    )

    return parser.parse_args()


# ==================================================
# [05] extract_normalized_pose()
#      - 이미지에서 YOLO Pose 추론
# ==================================================
# 한 장의 이미지를 입력받아
# 정규화된 Keypoint를 반환하는 함수입니다.
#
# 왼팔 이미지와 오른팔 이미지에
# 동일한 모델과 정규화 방식을 사용합니다.

def extract_normalized_pose(
    model,
    source: Path,
    config,
    device,
):

    # YOLO Pose로 사람과 관절 위치 추론
    results = model.predict(
        source=str(source),
        imgsz=config["image_size"],
        conf=config[
            "person_confidence"
        ],
        device=device,
        verbose=False,
    )

    # 첫 번째 이미지의 추론 결과 선택
    result = results[0]

    # ==============================================
    # [06] Keypoint 검출 여부 확인
    # ==============================================
    # 사람의 Keypoint를 찾지 못하면
    # 좌우 반전과 관절 비교를 진행할 수 없습니다.
    #
    # 검출에 실패한 이미지 이름을 표시하고
    # RuntimeError를 발생시킵니다.

    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or len(
            result.keypoints.xy
        ) == 0
    ):
        raise RuntimeError(
            f"Pose를 찾지 못했습니다: "
            f"{source.name}"
        )

    # ==============================================
    # [07] 첫 번째 사람의 관절 좌표 추출
    # ==============================================
    # result.keypoints.xy[0]
    # → 첫 번째 사람의 COCO 17 관절 좌표
    #
    # cpu()
    # → CPU 메모리로 이동
    #
    # numpy()
    # → NumPy 배열로 변환
    #
    # 결과 형태: (17, 2)

    keypoints = (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )

    # ==============================================
    # [08] 관절 좌표 정규화 후 반환
    # ==============================================
    # normalize_by_shoulder_width()를 호출합니다.
    #
    # 1. 좌·우 골반의 중심 계산
    # 2. 골반 중심을 (0, 0)으로 이동
    # 3. 좌·우 어깨 사이 거리 계산
    # 4. 중심 이동 좌표를 어깨 폭으로 나누기
    #
    # 반환값 중 normalized만 사용합니다.
    # 골반 중심과 어깨 폭은 _로 받습니다.

    normalized, _, _ = (
        normalize_by_shoulder_width(
            keypoints
        )
    )

    return normalized


# ==================================================
# [09] mean_joint_distance()
#      - 두 Pose의 평균 관절 거리 계산
# ==================================================
# 두 Pose의 같은 번호 관절끼리 거리를 계산합니다.
#
# 예:
# A의 Left Wrist(9)
#      ↕
# B의 Left Wrist(9)
#
# 17개 관절의 거리 평균을 반환합니다.
#
# 두 Pose는 동일한 관절 순서를 사용해야 합니다.

def mean_joint_distance(
    pose_a,
    pose_b,
) -> float:

    # 같은 번호 관절의 좌표 차이를 구한 뒤
    # 각 관절의 2D 직선거리를 계산합니다.
    #
    # axis=1
    # → 관절별 x, y 좌표를 하나의 거리로 계산

    distances = np.linalg.norm(
        pose_a - pose_b,
        axis=1,
    )

    # 17개 거리값의 평균 반환
    return float(
        np.mean(distances)
    )


def main() -> None:

    # ==============================================
    # [10] 실행 옵션과 이미지 경로 설정
    # ==============================================
    # --left, --right 옵션을 가져옵니다.
    #
    # 왼팔 이미지와 오른팔 이미지의
    # 파일 경로를 Path 객체로 만듭니다.

    args = parse_args()

    left_source = Path(
        args.left
    )

    right_source = Path(
        args.right
    )

    # ==============================================
    # [11] 입력 이미지 존재 여부 확인
    # ==============================================
    # 비교할 두 이미지가 모두 필요합니다.
    #
    # 하나라도 없다면
    # FileNotFoundError를 발생시킵니다.

    if not left_source.exists():
        raise FileNotFoundError(
            left_source
        )

    if not right_source.exists():
        raise FileNotFoundError(
            right_source
        )

    # ==============================================
    # [12] 설정값과 YOLO Pose 모델 준비
    # ==============================================
    # load_config()
    # → 모델 경로, 이미지 입력 크기 등의 설정 확인
    #
    # resolve_device()
    # → CPU 또는 GPU 결정
    #
    # YOLO()
    # → 사전학습된 Pose 모델 불러오기

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

    # ==============================================
    # [13] 왼팔을 올린 이미지 정규화
    # ==============================================
    # left_arm_up.jpg를 YOLO Pose로 분석하고,
    # 17개 관절 좌표를 정규화합니다.
    #
    # 결과:
    # left_normalized → (17, 2)

    left_normalized = (
        extract_normalized_pose(
            model,
            left_source,
            config,
            device,
        )
    )

    # ==============================================
    # [14] 오른팔을 올린 이미지 정규화
    # ==============================================
    # right_arm_up.jpg도 같은 방법으로
    # 관절 좌표를 추출하고 정규화합니다.
    #
    # 결과:
    # right_normalized → (17, 2)

    right_normalized = (
        extract_normalized_pose(
            model,
            right_source,
            config,
            device,
        )
    )

    # ==============================================
    # [15] 왼팔 Pose를 좌우 반전
    # ==============================================
    # feature_utils.py의
    # flip_keypoints_horizontal()을 호출합니다.
    #
    # 좌우 반전 과정:
    #
    # 1. 모든 관절의 x 좌표 부호 변경
    #    x → -x
    #
    # 2. 왼쪽·오른쪽 관절 의미 교환
    #    Left Shoulder(5) ↔ Right Shoulder(6)
    #    Left Elbow(7)    ↔ Right Elbow(8)
    #    Left Wrist(9)    ↔ Right Wrist(10)
    #
    # 원본 left_normalized는 변경하지 않습니다.

    flipped_left = (
        flip_keypoints_horizontal(
            left_normalized
        )
    )

    # ==============================================
    # [16] 좌우 반전 전·후 평균 거리 비교
    # ==============================================
    # original_distance
    # → 원본 왼팔 Pose와 실제 오른팔 Pose의 거리
    #
    # flipped_distance
    # → 반전한 왼팔 Pose와 실제 오른팔 Pose의 거리
    #
    # 두 값 모두 정규화 좌표 기준의
    # 17개 관절 평균 거리입니다.
    #
    # 거리가 작을수록 두 Pose의 대응 관절 좌표가
    # 서로 가깝다는 의미입니다.

    # 좌우 반전 전 평균 거리
    original_distance = (
        mean_joint_distance(
            left_normalized,
            right_normalized,
        )
    )

    # 좌우 반전 후 평균 거리
    flipped_distance = (
        mean_joint_distance(
            flipped_left,
            right_normalized,
        )
    )

    # ==============================================
    # [17] CSV 저장 경로 생성
    # ==============================================
    # 비교 결과를 다음 경로에 저장합니다.
    #
    # reports/day03/tables/
    # left_right_flip_comparison.csv
    #
    # 저장 폴더가 없다면 자동 생성합니다.

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / "left_right_flip_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # [18] 17개 관절의 비교 데이터 만들기
    # ==============================================
    # COCO 17 Keypoint를 0번부터 16번까지
    # 하나씩 확인합니다.
    #
    # 각 관절에 대해 다음 데이터를 저장합니다.
    #
    # 1. 원본 왼팔 Pose 좌표
    # 2. 좌우 반전된 왼팔 Pose 좌표
    # 3. 실제 오른팔 Pose 좌표
    # 4. 반전 전 관절 거리
    # 5. 반전 후 관절 거리
    #
    # 결과는 rows 리스트에 저장합니다.

    rows = []

    for index, name in enumerate(
        COCO_KEYPOINT_NAMES
    ):

        # 원본 왼팔 Pose의 현재 관절 좌표
        left_x, left_y = (
            left_normalized[index]
        )

        # 좌우 반전된 Pose의 현재 관절 좌표
        flipped_x, flipped_y = (
            flipped_left[index]
        )

        # 실제 오른팔 Pose의 현재 관절 좌표
        right_x, right_y = (
            right_normalized[index]
        )

        # ------------------------------------------
        # 반전 전 관절 거리 계산
        # ------------------------------------------
        # 원본 왼팔 Pose와 실제 오른팔 Pose에서
        # 같은 번호 관절의 거리를 계산합니다.

        original_joint_distance = (
            np.linalg.norm(
                left_normalized[index]
                - right_normalized[index]
            )
        )

        # ------------------------------------------
        # 반전 후 관절 거리 계산
        # ------------------------------------------
        # 반전한 왼팔 Pose와 실제 오른팔 Pose에서
        # 같은 번호 관절의 거리를 계산합니다.

        flipped_joint_distance = (
            np.linalg.norm(
                flipped_left[index]
                - right_normalized[index]
            )
        )

        # ------------------------------------------
        # 하나의 관절 데이터를 딕셔너리로 저장
        # ------------------------------------------
        # 좌표와 거리값을 소수점 여섯째 자리까지
        # 정리하여 rows에 추가합니다.

        rows.append(
            {
                "keypoint_id": index,
                "keypoint_name": name,

                "left_x": round(
                    float(left_x),
                    6,
                ),
                "left_y": round(
                    float(left_y),
                    6,
                ),

                "flipped_left_x": round(
                    float(flipped_x),
                    6,
                ),
                "flipped_left_y": round(
                    float(flipped_y),
                    6,
                ),

                "actual_right_x": round(
                    float(right_x),
                    6,
                ),
                "actual_right_y": round(
                    float(right_y),
                    6,
                ),

                "original_distance": round(
                    float(
                        original_joint_distance
                    ),
                    6,
                ),

                "flipped_distance": round(
                    float(
                        flipped_joint_distance
                    ),
                    6,
                ),
            }
        )

    # ==============================================
    # [19] CSV 파일 저장
    # ==============================================
    # output_path 경로에 CSV를 저장합니다.
    #
    # writeheader()
    # → 첫 번째 행에 열 이름 기록
    #
    # writerows(rows)
    # → 17개 관절의 비교 데이터 기록
    #
    # utf-8-sig
    # → Excel에서 한글이 깨지는 문제를 줄입니다.

    with output_path.open(
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
    # [20] 이미지 이름과 평균 관절 거리 출력
    # ==============================================
    # 비교에 사용한 두 이미지 이름을 출력합니다.
    #
    # Original Left vs Right
    # → 좌우 반전 전 평균 거리
    #
    # Flipped Left vs Right
    # → 좌우 반전 후 평균 거리
    #
    # 반전 후 거리가 작아졌다면
    # 두 Pose가 좌표상 더 가까워진 것입니다.

    print(
        f"Left image : "
        f"{left_source.name}"
    )

    print(
        f"Right image: "
        f"{right_source.name}"
    )

    print()

    print(
        "Mean Joint Distance"
    )

    print(
        f"Original Left vs Right : "
        f"{original_distance:.4f}"
    )

    print(
        f"Flipped Left vs Right  : "
        f"{flipped_distance:.4f}"
    )

    # ==============================================
    # [21] 주요 팔 관절의 좌표 비교 출력
    # ==============================================
    # 전체 17개 관절 중
    # 어깨·팔꿈치·손목 6개를 확인합니다.
    #
    # 5, 6  : 좌·우 어깨
    # 7, 8  : 좌·우 팔꿈치
    # 9, 10 : 좌·우 손목
    #
    # 각 관절의 x 좌표를 나란히 출력합니다.
    #
    # left_x  : 원본 왼팔 Pose
    # flip_x  : 좌우 반전된 왼팔 Pose
    # right_x : 실제 오른팔 Pose

    print()

    print(
        "Selected Keypoints"
    )

    for index in [
        5, 6,
        7, 8,
        9, 10,
    ]:

        # [18]에서 저장한 현재 관절의 데이터
        row = rows[index]

        print(
            f"{index:02d} "
            f"{row['keypoint_name']:16s} "
            f"left_x="
            f"{row['left_x']:7.3f} "
            f"flip_x="
            f"{row['flipped_left_x']:7.3f} "
            f"right_x="
            f"{row['actual_right_x']:7.3f}"
        )

    # ==============================================
    # [22] CSV 저장 위치 출력
    # ==============================================
    # 저장한 파일의 전체 경로를 출력합니다.

    print()

    print(
        f"Saved: {output_path}"
    )


# ==================================================
# [23] Python 프로그램 실행 시작점
# ==================================================
# 이 파일을 직접 실행하면 main()을 호출합니다.

if __name__ == "__main__":
    main()