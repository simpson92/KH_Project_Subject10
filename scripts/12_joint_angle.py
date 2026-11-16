from __future__ import annotations

# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
# argparse : 실행할 때 이미지 경로와 옵션 전달
# sys      : Python 모듈 검색 경로 설정
# pathlib  : 파일 및 폴더 경로 처리
# numpy    : 좌표 및 Confidence 배열 처리
# YOLO     : 이미지에서 사람의 Keypoint 추출

import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ==================================================
# [02] 프로젝트 ROOT 경로 설정
# ==================================================
# 현재 파일은 scripts 폴더 안에 있습니다.
#
# __file__ : 현재 Python 파일의 경로
# parents[1] : 프로젝트 최상위 폴더
#
# ROOT를 sys.path에 추가하면
# src 폴더에 있는 공통 함수를 불러올 수 있습니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ==================================================
# [03] 다른 파일의 공통 함수 가져오기
# ==================================================
# common.py
# → 설정값과 CPU/GPU 실행 장치 확인
#
# feature_utils.py
# → 팔꿈치·무릎 관절각도 계산
#
# pose_utils.py
# → COCO 17 Keypoint의 관절 이름 확인

from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# [04] 실행 옵션 정의
# ==================================================
# --source
# → 분석할 이미지 파일의 경로
#
# --kp-conf
# → 관절 Confidence 판단 기준
# → 기본값은 0.50
#
# 옵션을 입력하지 않으면
# arms_up_person_a.jpg를 분석합니다.

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


def main() -> None:

    # ==============================================
    # [05] 입력 이미지 경로 및 존재 여부 확인
    # ==============================================
    # parse_args()에서 실행 옵션을 가져옵니다.
    #
    # 이미지가 없다면
    # FileNotFoundError를 발생시킵니다.

    args = parse_args()

    source = Path(
        args.source
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    # ==============================================
    # [06] 설정값과 실행 장치 확인
    # ==============================================
    # load_config()
    # → 모델 경로, 이미지 크기 등의 설정 읽기
    #
    # resolve_device()
    # → 설정값을 바탕으로 실행 장치 결정

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    # ==============================================
    # [07] 사전학습된 YOLO Pose 모델 불러오기
    # ==============================================
    # config["pose_model"]에 지정된
    # 모델을 불러옵니다.
    #
    # 여기서는 새로운 AI 모델을 학습하지 않고,
    # 이미 학습된 모델을 사용합니다.

    model = YOLO(
        config["pose_model"]
    )

    # ==============================================
    # [08] YOLO Pose로 이미지 추론
    # ==============================================
    # source : 분석할 이미지
    # imgsz  : 모델 입력 이미지 크기
    # conf   : 사람 검출 Confidence 기준
    # device : CPU 또는 GPU
    #
    # 추론 결과에는 사람의 관절 좌표와
    # 각 관절의 Confidence가 포함될 수 있습니다.

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
    # [09] Keypoint 검출 여부 확인
    # ==============================================
    # 이미지에서 사람의 Keypoint를
    # 찾지 못했다면 안내 메시지를 출력합니다.
    #
    # 관절 좌표가 없으므로
    # 이후 각도 계산은 진행하지 않습니다.

    if (
        result.keypoints is None
        or result.keypoints.xy
        is None
        or len(
            result.keypoints.xy
        ) == 0
    ):
        print(
            "사람의 Keypoint를 "
            "찾지 못했습니다."
        )
        return

    # ==============================================
    # [10] Keypoint 좌표를 NumPy 배열로 변환
    # ==============================================
    # result.keypoints.xy
    # → 사람별 COCO 17 Keypoint의 (x, y) 좌표
    #
    # cpu()
    # → CPU 메모리로 이동
    #
    # numpy()
    # → NumPy 배열로 변환

    xy = (
        result.keypoints.xy
        .cpu()
        .numpy()
    )

    # ==============================================
    # [11] Keypoint Confidence 가져오기
    # ==============================================
    # Confidence는 관절 위치 추정의
    # 신뢰도를 나타내는 값입니다.
    #
    # Confidence가 있으면 NumPy로 변환합니다.
    #
    # 값이 제공되지 않으면
    # 코드의 기본 처리 방식에 따라
    # 모든 Confidence를 1.0으로 설정합니다.

    if (
        result.keypoints.conf
        is not None
    ):
        confidence = (
            result.keypoints.conf
            .cpu()
            .numpy()
        )
    else:
        confidence = np.ones(
            xy.shape[:2],
            dtype=np.float32,
        )

    # ==============================================
    # [12] 첫 번째 사람의 데이터 선택
    # ==============================================
    # 여러 사람이 검출되었더라도
    # 이번 기본 실습에서는 첫 번째 사람만 사용합니다.
    #
    # xy[0]
    # → 첫 번째 사람의 17개 관절 좌표
    #
    # confidence[0]
    # → 첫 번째 사람의 관절 Confidence

    person_xy = xy[0]

    person_conf = confidence[0]

    # ==============================================
    # [13] 분석 이미지 이름 출력
    # ==============================================
    # 어떤 이미지를 분석했는지 확인합니다.

    print(
        f"Source: {source.name}"
    )

    print()

    # ==============================================
    # [14] 필요한 관절의 Confidence 출력
    # ==============================================
    # 팔꿈치와 무릎 각도에 필요한
    # 관절 번호를 선택합니다.
    #
    # 5, 7, 9   : 왼쪽 팔
    # 6, 8, 10  : 오른쪽 팔
    # 11, 13, 15: 왼쪽 다리
    # 12, 14, 16: 오른쪽 다리
    #
    # 관절 번호, 이름, Confidence를 출력합니다.

    print(
        "관절 Confidence 확인"
    )

    for keypoint_id in [
        5, 7, 9,
        6, 8, 10,
        11, 13, 15,
        12, 14, 16,
    ]:
        print(
            f"{keypoint_id:02d} "
            f"{COCO_KEYPOINT_NAMES[keypoint_id]:16s} "
            f"{person_conf[keypoint_id]:.3f}"
        )

    # ==============================================
    # [15] 각도 계산에 필요한 관절 번호 정의
    # ==============================================
    # 관절각도는 세 개의 Keypoint로 계산합니다.
    #
    # 가운데 관절이 각도의 중심입니다.
    #
    # 왼쪽 팔꿈치:
    # Shoulder(5) - Elbow(7) - Wrist(9)
    #
    # 오른쪽 팔꿈치:
    # Shoulder(6) - Elbow(8) - Wrist(10)
    #
    # 왼쪽 무릎:
    # Hip(11) - Knee(13) - Ankle(15)
    #
    # 오른쪽 무릎:
    # Hip(12) - Knee(14) - Ankle(16)

    angle_specs = {
        "left_elbow_angle": (
            5, 7, 9
        ),

        "right_elbow_angle": (
            6, 8, 10
        ),

        "left_knee_angle": (
            11, 13, 15
        ),

        "right_knee_angle": (
            12, 14, 16
        ),
    }

    # ==============================================
    # [16] 팔꿈치·무릎 관절각도 4개 계산
    # ==============================================
    # feature_utils.py에 작성한
    # selected_joint_angles()를 호출합니다.
    #
    # 내부에서는 calculate_angle()을 사용하여
    # 각 관절의 각도를 계산합니다.
    #
    # 계산 결과:
    # left_elbow_angle
    # right_elbow_angle
    # left_knee_angle
    # right_knee_angle

    angles = selected_joint_angles(
        person_xy
    )

    print()
    print(
        "Joint Angles"
    )

    # ==============================================
    # [17] Confidence 확인 후 관절각도 출력
    # ==============================================
    # 각 관절각도를 하나씩 확인합니다.
    #
    # 1. 각도를 만드는 관절 3개 선택
    # 2. 세 관절의 Confidence 가져오기
    # 3. 가장 낮은 Confidence 확인
    # 4. 기준 미만이면 SKIP
    # 5. 기준 이상이면 각도 출력
    #
    # 예:
    # Shoulder = 0.95
    # Elbow    = 0.92
    # Wrist    = 0.30
    #
    # 최소 Confidence = 0.30
    # 기준 0.50 미만이므로 SKIP합니다.

    for name, angle in (
        angles.items()
    ):

        # 현재 각도에 필요한 관절 번호 3개 선택
        required_ids = (
            angle_specs[name]
        )

        # 세 관절의 Confidence 가져오기
        required_conf = [
            person_conf[index]
            for index
            in required_ids
        ]

        # 세 관절 중 가장 낮은 Confidence
        min_conf = min(
            required_conf
        )

        # Confidence가 기준보다 낮다면
        # 각도 출력 대신 SKIP 메시지 출력
        if min_conf < args.kp_conf:
            print(
                f"{name:20s}: "
                f"SKIP "
                f"(min conf={min_conf:.3f})"
            )

            continue

        # 기준을 통과한 관절각도 출력
        # 소수점 둘째 자리까지 표시
        print(
            f"{name:20s}: "
            f"{angle:7.2f}°"
        )


# ==================================================
# [18] Python 프로그램 실행 시작점
# ==================================================
# 이 파일을 직접 실행하면
# main() 함수를 호출합니다.

if __name__ == "__main__":
    main()