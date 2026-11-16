from __future__ import annotations

# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
# argparse : 실행 옵션 처리
# csv      : 비교 결과를 CSV 파일로 저장
# sys      : Python 모듈 검색 경로 설정
# Path     : 파일 및 폴더 경로 처리
# numpy    : 관절 사이의 거리와 평균 계산
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
# 현재 파일은 scripts 폴더에 있습니다.
#
# parents[1]을 이용하여
# 프로젝트 최상위 폴더를 찾습니다.
#
# ROOT를 sys.path에 추가하면
# src 폴더의 공통 함수를 가져올 수 있습니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ==================================================
# [03] 다른 파일의 공통 함수 가져오기
# ==================================================
# common.py
# → 설정값 읽기 및 CPU/GPU 확인
#
# feature_utils.py
# → 골반 중심 이동 및 어깨 폭 정규화

from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    normalize_by_shoulder_width,
)


# ==================================================
# [04] 실행 옵션 정의
# ==================================================
# --a
# → 가까이 촬영한 이미지 경로
#
# --b
# → 멀리 촬영한 이미지 경로
#
# --output
# → 비교 결과를 저장할 CSV 파일명
#
# 옵션을 생략하면 기본 이미지를 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    # 이미지 A: 가까운 거리
    parser.add_argument(
        "--a",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "arms_up_near.jpg"
        ),
    )

    # 이미지 B: 먼 거리
    parser.add_argument(
        "--b",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "arms_up_far.jpg"
        ),
    )

    # 비교 결과 CSV 파일명
    parser.add_argument(
        "--output",
        default="normalization_comparison.csv",
        help=(
            "reports/day03/tables 아래에 "
            "저장할 CSV 파일명"
        ),
    )

    return parser.parse_args()


# ==================================================
# [05] extract_first_person()
#      - YOLO Pose로 관절 추론
# ==================================================
# 입력받은 이미지에서
# 첫 번째 사람의 Keypoint를 추출하는 함수입니다.
#
# model  : YOLO Pose 모델
# source : 분석할 이미지 경로
# config : 모델 설정값
# device : CPU 또는 GPU

def extract_first_person(
    model,
    source,
    config,
    device,
):

    # 이미지에서 사람과 Keypoint 추론
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
    # 사람의 Keypoint가 검출되지 않았다면
    # 관절 좌표를 비교할 수 없습니다.
    #
    # 따라서 RuntimeError를 발생시킵니다.

    if (
        result.keypoints is None
        or result.keypoints.xy
        is None
        or len(
            result.keypoints.xy
        ) == 0
    ):
        raise RuntimeError(
            f"Pose 검출 실패: "
            f"{source}"
        )

    # ==============================================
    # [07] 첫 번째 사람의 관절 좌표 반환
    # ==============================================
    # result.keypoints.xy[0]
    # → 첫 번째 사람의 17개 관절 좌표
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

    return keypoints


# ==================================================
# [08] mean_joint_distance()
#      - 두 Pose의 평균 관절 거리 계산
# ==================================================
# 두 Pose에서 같은 번호 관절끼리
# 2차원 좌표 사이의 거리를 계산합니다.
#
# 예:
# A의 왼쪽 손목 ↔ B의 왼쪽 손목
# A의 오른쪽 무릎 ↔ B의 오른쪽 무릎
#
# 17개 관절의 거리를 계산한 뒤
# 평균값을 반환합니다.
#
# 주의:
# 두 Pose는 같은 관절 순서와
# 동일한 좌표 표현을 사용해야 합니다.

def mean_joint_distance(
    pose_a,
    pose_b,
):

    # 두 Pose의 대응 관절 좌표를 뺍니다.
    #
    # np.linalg.norm(..., axis=1)
    # → 각 관절의 2D 직선거리 계산
    #
    # distances에는 17개 거리값이 저장됩니다.

    distances = np.linalg.norm(
        pose_a - pose_b,
        axis=1,
    )

    # 17개 관절 거리의 평균 계산
    return float(
        np.mean(distances)
    )


def main() -> None:

    # ==============================================
    # [09] 실행 옵션 및 이미지 경로 설정
    # ==============================================
    # parse_args()를 호출하여
    # 이미지 A, B의 경로를 가져옵니다.

    args = parse_args()

    source_a = Path(args.a)
    source_b = Path(args.b)

    # ==============================================
    # [10] 입력 이미지 존재 여부 확인
    # ==============================================
    # A와 B 이미지가 모두 있어야
    # 정규화 전후 차이를 비교할 수 있습니다.
    #
    # 하나라도 없으면 오류를 발생시킵니다.

    if not source_a.exists():
        raise FileNotFoundError(
            source_a
        )

    if not source_b.exists():
        raise FileNotFoundError(
            source_b
        )

    # ==============================================
    # [11] 설정값과 YOLO Pose 모델 준비
    # ==============================================
    # 설정 파일에서 모델 정보와
    # 실행 장치를 확인합니다.
    #
    # 두 이미지에 동일한 YOLO Pose 모델을
    # 사용하여 관절을 추출합니다.

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

    # ==============================================
    # [12] 이미지 A와 B에서 Keypoint 추출
    # ==============================================
    # [05] extract_first_person()을 호출합니다.
    #
    # 이미지 A:
    # arms_up_near.jpg
    #
    # 이미지 B:
    # arms_up_far.jpg
    #
    # 각 이미지에서 첫 번째 사람의
    # 17개 Pixel 좌표를 가져옵니다.

    keypoints_a = (
        extract_first_person(
            model,
            source_a,
            config,
            device,
        )
    )

    keypoints_b = (
        extract_first_person(
            model,
            source_b,
            config,
            device,
        )
    )

    # ==============================================
    # [13] 이미지 A와 B의 좌표 정규화
    # ==============================================
    # 앞에서 작성한
    # normalize_by_shoulder_width()를 호출합니다.
    #
    # 정규화 과정:
    # 1. 골반 중심 계산
    # 2. 골반 중심을 원점으로 이동
    # 3. 어깨 폭 계산
    # 4. 중심 이동 좌표를 어깨 폭으로 나누기
    #
    # 반환값:
    # normalized : 정규화 좌표
    # center     : 골반 중심
    # scale      : 어깨 폭

    # 이미지 A 정규화
    normalized_a, (
        center_a
    ), scale_a = (
        normalize_by_shoulder_width(
            keypoints_a
        )
    )

    # 이미지 B 정규화
    normalized_b, (
        center_b
    ), scale_b = (
        normalize_by_shoulder_width(
            keypoints_b
        )
    )

    # ==============================================
    # [14] 정규화 전 평균 관절 거리 계산
    # ==============================================
    # 두 이미지의 원본 Pixel 좌표를 비교합니다.
    #
    # 같은 번호 관절끼리 거리를 구한 뒤
    # 17개 관절 거리의 평균을 계산합니다.
    #
    # 결과 단위: Pixel

    pixel_difference = (
        mean_joint_distance(
            keypoints_a,
            keypoints_b,
        )
    )

    # ==============================================
    # [15] 정규화 후 평균 관절 거리 계산
    # ==============================================
    # 두 이미지의 정규화 좌표를 비교합니다.
    #
    # 골반 중심 위치와 어깨 폭 차이를
    # 보정한 상태에서 비교합니다.
    #
    # 결과는 Pixel이 아닌
    # 어깨 폭 기준의 무차원 거리입니다.

    normalized_difference = (
        mean_joint_distance(
            normalized_a,
            normalized_b,
        )
    )

    # ==============================================
    # [16] 비교 이미지와 평균 거리 출력
    # ==============================================
    # A와 B 이미지 이름을 출력합니다.
    #
    # Pixel space
    # → 정규화 전 평균 관절 거리
    #
    # Normalized space
    # → 정규화 후 평균 관절 거리

    print(
        f"A: {source_a.name}"
    )

    print(
        f"B: {source_b.name}"
    )

    print()

    print(
        "Mean joint distance"
    )

    print(
        f"Pixel space      : "
        f"{pixel_difference:.4f}"
    )

    print(
        f"Normalized space : "
        f"{normalized_difference:.4f}"
    )

    # ==============================================
    # [17] 골반 중심과 어깨 폭 출력
    # ==============================================
    # 이미지마다 골반 중심의 위치와
    # 어깨 폭이 어떻게 다른지 확인합니다.
    #
    # 가까이 촬영한 사람은 일반적으로
    # 화면에서 어깨 폭이 더 크게 나타납니다.
    #
    # 단, 자세나 카메라 각도의 영향도 받습니다.

    print()

    print(
        f"A hip center     : "
        f"{center_a}"
    )

    print(
        f"B hip center     : "
        f"{center_b}"
    )

    print(
        f"A shoulder width : "
        f"{scale_a:.2f}"
    )

    print(
        f"B shoulder width : "
        f"{scale_b:.2f}"
    )

    # ==============================================
    # [18] CSV 저장 경로 생성
    # ==============================================
    # reports/day03/tables 폴더 아래에
    # 비교 결과 CSV를 저장합니다.
    #
    # 기본 파일명:
    # normalization_comparison.csv

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / args.output
    )

    # 저장 폴더가 없다면 자동 생성합니다.
    # 기존 코드에 추가한 오류 방지 코드입니다.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # [19] CSV 저장용 결과 데이터 생성
    # ==============================================
    # 다음 다섯 가지 값을 하나의 행으로 저장합니다.
    #
    # 1. 비교한 이미지 이름
    # 2. Pixel 평균 관절 거리
    # 3. 정규화 후 평균 관절 거리
    # 4. 이미지 A의 어깨 폭
    # 5. 이미지 B의 어깨 폭
    #
    # 소수점 자리수를 정리하여 저장합니다.

    rows = [
        {
            "pair": (
                f"{source_a.name}"
                f" vs "
                f"{source_b.name}"
            ),

            "pixel_mean_distance": (
                round(
                    pixel_difference,
                    6,
                )
            ),

            "normalized_mean_distance": (
                round(
                    normalized_difference,
                    6,
                )
            ),

            "a_shoulder_width": (
                round(
                    scale_a,
                    4,
                )
            ),

            "b_shoulder_width": (
                round(
                    scale_b,
                    4,
                )
            ),
        }
    ]

    # ==============================================
    # [20] CSV 파일 저장
    # ==============================================
    # CSV 파일을 쓰기 모드로 엽니다.
    #
    # utf-8-sig
    # → Excel에서 한글 깨짐을 줄이는 인코딩
    #
    # writeheader()
    # → CSV 열 이름 기록
    #
    # writerows()
    # → 비교 결과 기록

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
    # [21] CSV 저장 위치 출력
    # ==============================================
    # 결과 파일이 저장된 경로를 출력하여
    # 학생들이 CSV 파일을 확인하도록 합니다.

    print()
    print(
        f"Saved: {output_path}"
    )


# ==================================================
# [22] Python 프로그램 실행 시작점
# ==================================================
# 파일을 직접 실행하면 main() 함수를 호출합니다.

if __name__ == "__main__":
    main()