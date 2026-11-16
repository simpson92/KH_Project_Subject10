from __future__ import annotations

# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
# argparse : 실행 옵션 처리
# csv      : Feature를 CSV 파일로 저장
# sys      : Python 모듈 검색 경로 설정
# Path     : 파일 및 폴더 경로 처리
# numpy    : Confidence 평균과 개수 계산
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
# parents[1]로 프로젝트 최상위 폴더를 찾고
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
# → 설정 파일 읽기 및 CPU/GPU 확인
#
# feature_utils.py
# → 관절 좌표 정규화
# → 팔꿈치·무릎 관절각도 계산
#
# pose_utils.py
# → COCO 17 Keypoint 이름 확인

from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# [04] 실행 옵션 정의
# ==================================================
# --source
# → 분석할 이미지 파일 경로
#
# --kp-conf
# → 낮은 Confidence를 판단할 기준값
#
# 기본값은 0.50입니다.
#
# 주의:
# 현재 코드에서는 Confidence가 낮은 관절을
# 제거하지 않고 개수만 기록합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "arms_up_person_a.jpg"
        ),
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


# ==================================================
# [05] make_feature_names()
#      - 38개 Feature 이름 생성
# ==================================================
# COCO 17 Keypoint 각각에
# x, y 이름을 붙여 34개 이름을 만듭니다.
#
# 예:
# nose_x, nose_y
# left_shoulder_x, left_shoulder_y
#
# 이후 관절각도 이름 4개를 추가합니다.
#
# 최종 Feature 이름 개수:
# 34 + 4 = 38개

def make_feature_names():
    names = []

    # 17개 관절의 x, y 이름 생성
    for keypoint_name in (
        COCO_KEYPOINT_NAMES
    ):
        names.append(
            f"{keypoint_name}_x"
        )

        names.append(
            f"{keypoint_name}_y"
        )

    # 관절각도 Feature 이름 4개 추가
    names.extend(
        [
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
        ]
    )

    # 총 38개 Feature 이름 반환
    return names


def main() -> None:

    # ==============================================
    # [06] 실행 옵션과 이미지 경로 설정
    # ==============================================
    # parse_args()로 실행 옵션을 가져옵니다.
    #
    # source에는 분석할 이미지 경로를 저장합니다.

    args = parse_args()

    source = Path(
        args.source
    )

    # ==============================================
    # [07] 입력 이미지 존재 여부 확인
    # ==============================================
    # 이미지가 없다면
    # FileNotFoundError를 발생시킵니다.

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    # ==============================================
    # [08] 설정값과 실행 장치 확인
    # ==============================================
    # load_config()
    # → 모델 경로, 입력 이미지 크기 등의 설정 읽기
    #
    # resolve_device()
    # → CPU 또는 GPU 실행 장치 결정

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    # ==============================================
    # [09] 사전학습된 YOLO Pose 모델 불러오기
    # ==============================================
    # 설정 파일에 지정된 Pose 모델을 로드합니다.
    #
    # 새로운 AI 모델을 학습하는 과정이 아니라
    # 사전학습된 모델을 사용하는 과정입니다.

    model = YOLO(
        config["pose_model"]
    )

    # ==============================================
    # [10] YOLO Pose로 이미지 추론
    # ==============================================
    # 입력 이미지에서 사람을 검출하고
    # COCO 17 Keypoint를 추정합니다.
    #
    # source : 입력 이미지
    # imgsz  : 모델 입력 크기
    # conf   : 사람 검출 Confidence 기준
    # device : CPU 또는 GPU

    results = model.predict(
        source=str(source),
        imgsz=config["image_size"],
        conf=config[
            "person_confidence"
        ],
        device=device,
        verbose=False,
    )

    result = results[0]

    # ==============================================
    # [11] Keypoint 검출 여부 확인
    # ==============================================
    # 사람의 Keypoint가 없다면
    # 38-D Feature를 생성할 수 없습니다.
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
            "Pose를 찾지 못했습니다."
        )

    # ==============================================
    # [12] 첫 번째 사람의 Keypoint 추출
    # ==============================================
    # 여러 사람이 검출되더라도
    # 첫 번째 사람의 관절 좌표만 사용합니다.
    #
    # keypoints.shape = (17, 2)
    #
    # 각 관절의 Pixel 좌표:
    # [x, y]

    keypoints = (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )

    # ==============================================
    # [13] Keypoint Confidence 가져오기
    # ==============================================
    # 17개 관절의 Confidence를 가져옵니다.
    #
    # 값이 제공되지 않으면
    # 코드의 기존 처리 방식에 따라
    # 모두 1.0으로 대체합니다.
    #
    # 주의:
    # 대체값 1.0은 실제 검출 신뢰도를 뜻하지 않습니다.

    if (
        result.keypoints.conf
        is not None
    ):
        confidence = (
            result.keypoints.conf[0]
            .cpu()
            .numpy()
        )
    else:
        confidence = np.ones(
            17,
            dtype=np.float32,
        )

    # ==============================================
    # [14] 관절 좌표 정규화
    # ==============================================
    # feature_utils.py의
    # normalize_by_shoulder_width()를 호출합니다.
    #
    # 정규화 과정:
    # 1. 좌·우 골반의 중간점을 계산
    # 2. 골반 중심을 원점으로 이동
    # 3. 좌·우 어깨 사이 거리 계산
    # 4. 이동된 좌표를 어깨 폭으로 나누기
    #
    # 반환값:
    # normalized : 정규화된 17개 관절 좌표
    # center     : 골반 중심
    # scale      : 어깨 폭

    normalized, (
        center
    ), scale = (
        normalize_by_shoulder_width(
            keypoints
        )
    )

    # ==============================================
    # [15] 팔꿈치·무릎 관절각도 4개 계산
    # ==============================================
    # selected_joint_angles()를 호출합니다.
    #
    # 원본 Pixel 좌표를 이용하여
    # 다음 네 관절각도를 계산합니다.
    #
    # 1. 왼쪽 팔꿈치
    # 2. 오른쪽 팔꿈치
    # 3. 왼쪽 무릎
    # 4. 오른쪽 무릎
    #
    # 계산 결과는 딕셔너리로 반환됩니다.

    angles = selected_joint_angles(
        keypoints
    )

    # ==============================================
    # [16] 정규화 좌표 34개를 리스트에 추가
    # ==============================================
    # normalized에는 17개 관절의
    # x, y 좌표가 저장되어 있습니다.
    #
    # 각 관절을 하나씩 반복하면서
    # x, y 값을 feature_values에 추가합니다.
    #
    # 17개 관절 × 2개 좌표 = 34개 숫자
    #
    # extend()는 여러 값을
    # 기존 리스트 뒤에 이어 붙이는 함수입니다.

    feature_values = []

    for point in normalized:
        feature_values.extend(
            [
                float(point[0]),
                float(point[1]),
            ]
        )

    # ==============================================
    # [17] 관절각도 4개를 Feature에 추가
    # ==============================================
    # 앞에서 만든 정규화 좌표 34개 뒤에
    # 관절각도 4개를 추가합니다.
    #
    # Feature 순서:
    # 정규화 좌표 34개
    # + 관절각도 4개
    #
    # 최종 결과:
    # 38-D Frame Feature

    feature_values.extend(
        [
            angles[
                "left_elbow_angle"
            ],
            angles[
                "right_elbow_angle"
            ],
            angles[
                "left_knee_angle"
            ],
            angles[
                "right_knee_angle"
            ],
        ]
    )

    # ==============================================
    # [18] Feature 이름 38개 가져오기
    # ==============================================
    # [05] make_feature_names()를 호출합니다.
    #
    # feature_names
    # → Feature의 이름 38개
    #
    # feature_values
    # → 실제 계산된 숫자 38개
    #
    # 두 리스트는 같은 순서로 대응됩니다.

    feature_names = (
        make_feature_names()
    )

    # ==============================================
    # [19] Keypoint Confidence 품질 정보 계산
    # ==============================================
    # mean_confidence
    # → 17개 관절 Confidence의 평균
    #
    # low_keypoint_count
    # → Confidence가 --kp-conf 미만인 관절 개수
    #
    # 예:
    # kp-conf = 0.50
    #
    # 0.30, 0.40인 관절이 있다면
    # low_keypoint_count = 2
    #
    # 주의:
    # 현재 코드는 개수만 기록하고
    # Feature 계산에서는 제외하지 않습니다.

    mean_confidence = float(
        np.mean(
            confidence
        )
    )

    low_keypoint_count = int(
        np.sum(
            confidence
            < args.kp_conf
        )
    )

    # ==============================================
    # [20] CSV에 저장할 기본 정보 생성
    # ==============================================
    # Feature 외에 결과를 해석하는 데 필요한
    # Metadata를 먼저 저장합니다.
    #
    # source             : 이미지 이름
    # mean_keypoint_conf : 평균 Confidence
    # low_keypoint_count : 낮은 Confidence 관절 수
    # hip_center_x/y     : 골반 중심 좌표
    # shoulder_width     : 정규화 기준 어깨 폭
    #
    # 이 6개 항목은 38-D Feature와 구분합니다.

    row = {
        "source": source.name,

        "mean_keypoint_conf": (
            round(
                mean_confidence,
                4,
            )
        ),

        "low_keypoint_count": (
            low_keypoint_count
        ),

        "hip_center_x": round(
            float(center[0]),
            4,
        ),

        "hip_center_y": round(
            float(center[1]),
            4,
        ),

        "shoulder_width": round(
            float(scale),
            4,
        ),
    }

    # ==============================================
    # [21] 38개 Feature 이름과 값 연결
    # ==============================================
    # zip()을 사용하면
    # Feature 이름과 실제 값을 짝지을 수 있습니다.
    #
    # 예:
    # nose_x → 0.125
    # nose_y → -2.130
    #
    # 이런 방식으로 38개 Feature를
    # row 딕셔너리에 추가합니다.
    #
    # Feature 값은 소수점 여섯째 자리까지
    # 반올림하여 저장합니다.

    for name, value in zip(
        feature_names,
        feature_values,
    ):
        row[name] = round(
            float(value),
            6,
        )

    # ==============================================
    # [22] CSV 저장 경로 생성
    # ==============================================
    # reports/day03/tables 폴더 아래에
    # 이미지별 Feature CSV를 저장합니다.
    #
    # 출력 파일 예시:
    # arms_up_person_a_frame_feature.csv

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / (
            f"{source.stem}"
            "_frame_feature.csv"
        )
    )

    # 저장 폴더가 없으면 자동 생성합니다.
    # 기존 코드의 파일 저장 오류를 예방합니다.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # [23] 38-D Frame Feature CSV 저장
    # ==============================================
    # output_path에 CSV 파일을 생성합니다.
    #
    # writeheader()
    # → CSV의 열 이름 기록
    #
    # writerow(row)
    # → 입력 이미지 한 장의 Feature를
    #   하나의 행으로 저장
    #
    # utf-8-sig
    # → Excel에서 한글 깨짐을 줄입니다.

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(row)

    # ==============================================
    # [24] 이미지와 Confidence 품질 정보 출력
    # ==============================================
    # 분석한 이미지 이름을 출력합니다.
    #
    # 평균 Confidence와
    # 낮은 Confidence 관절 개수를 확인합니다.
    #
    # len(feature_values)
    # → 생성된 Feature 개수
    # → 정상적인 COCO 17 입력이면 38개

    print(
        f"Source: {source.name}"
    )

    print(
        f"Mean keypoint confidence: "
        f"{mean_confidence:.3f}"
    )

    print(
        f"Low keypoint count: "
        f"{low_keypoint_count}"
    )

    print(
        f"Feature count: "
        f"{len(feature_values)}"
    )

    # ==============================================
    # [25] 계산된 관절각도 4개 출력
    # ==============================================
    # 왼쪽·오른쪽 팔꿈치 및
    # 왼쪽·오른쪽 무릎 각도를 출력합니다.
    #
    # 관절각도는 도(°) 단위이며
    # 소수점 둘째 자리까지 표시합니다.

    print()

    print(
        "Joint angles"
    )

    for name, value in (
        angles.items()
    ):
        print(
            f"{name:20s}: "
            f"{value:7.2f}°"
        )

    # ==============================================
    # [26] CSV 저장 위치 출력
    # ==============================================
    # 저장한 파일의 전체 경로를 출력합니다.
    #
    # 학생들은 해당 CSV를 열어
    # 38개 Feature를 확인할 수 있습니다.

    print()

    print(
        f"Saved: {output_path}"
    )


# ==================================================
# [27] Python 프로그램 실행 시작점
# ==================================================
# 이 파일을 직접 실행할 때만
# main() 함수를 호출합니다.

if __name__ == "__main__":
    main()