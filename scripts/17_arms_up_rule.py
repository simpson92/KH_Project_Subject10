from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리와 공통 함수 가져오기
# ==================================================
# argparse : 터미널에서 입력한 실행 옵션 처리
# sys      : Python 모듈 검색 경로 설정
# Path     : 이미지 파일 및 프로젝트 경로 관리
# numpy    : 관절 Confidence 기본값 생성
# YOLO     : 이미지에서 사람의 관절 위치 추정

import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ==================================================
# ② 프로젝트 경로 설정 및 공통 모듈 가져오기
# ==================================================
# 현재 Python 파일의 상위 프로젝트 폴더를 찾습니다.
# 프로젝트 폴더를 모듈 검색 경로에 추가합니다.
# 이후 src 폴더의 공통 함수들을 사용할 수 있습니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_WRIST,
    RIGHT_WRIST,
    normalize_by_shoulder_width,
)


# ==================================================
# ③ 터미널 실행 옵션 설정
# ==================================================
# --source : 분석할 이미지 파일 경로 (필수)
# --kp-conf : 관절 신뢰도 최소 기준 (기본 0.50)
#
# 예시:
# python scripts/check_arms.py \
#     --source samples/arms_up.jpg \
#     --kp-conf 0.5

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


def main() -> None:

    # ==============================================
    # ④ 입력 이미지와 AI 실행환경 확인
    # ==============================================
    # 터미널에서 전달한 이미지 경로를 읽습니다.
    # 이미지 파일이 없으면 오류를 발생시킵니다.
    # 설정 파일에서 모델과 실행 장치 정보를 읽습니다.
    # CPU 또는 GPU 실행환경을 결정합니다.

    args = parse_args()

    source = Path(
        args.source
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    # ==============================================
    # ⑤ YOLO Pose 모델로 관절 추정
    # ==============================================
    # 사전학습된 YOLO Pose 모델을 불러옵니다.
    # 입력 이미지에서 사람을 검출합니다.
    # 검출된 사람의 관절 좌표를 추정합니다.
    # 결과가 없으면 오류를 발생시킵니다.

    model = YOLO(
        config["pose_model"]
    )

    result = model.predict(
        source=str(source),
        imgsz=config["image_size"],
        conf=config[
            "person_confidence"
        ],
        device=device,
        verbose=False,
    )[0]

    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or len(
            result.keypoints.xy
        ) == 0
    ):
        raise RuntimeError(
            "Pose를 찾지 못했습니다."
        )

    # ==============================================
    # ⑥ 관절 좌표와 Confidence 추출
    # ==============================================
    # 첫 번째 검출된 사람의 관절 좌표를 가져옵니다.
    # GPU Tensor를 CPU NumPy 배열로 변환합니다.
    # 관절별 Confidence도 함께 가져옵니다.
    # Confidence 정보가 없다면 기본값 1.0을 사용합니다.

    keypoints = (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )

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
    # ⑦ 필요한 네 개 관절의 신뢰도 검사
    # ==============================================
    # 양팔 들기 판단에 필요한 관절은 네 개입니다.
    #
    # 왼쪽 어깨 / 오른쪽 어깨
    # 왼쪽 손목 / 오른쪽 손목
    #
    # 네 관절 중 가장 낮은 Confidence를 확인합니다.
    # 설정한 기준보다 낮으면 분석을 건너뜁니다.
    # 신뢰도가 낮은 좌표로 잘못 판단하는 것을
    # 방지하기 위한 단계입니다.

    required_ids = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_WRIST,
        RIGHT_WRIST,
    ]

    min_conf = min(
        float(
            confidence[index]
        )
        for index in required_ids
    )

    if min_conf < args.kp_conf:
        print(
            f"SKIP: 필요한 관절의 "
            f"최소 Confidence가 "
            f"{min_conf:.3f}입니다."
        )
        return

    # ==============================================
    # ⑧ 어깨 너비를 기준으로 관절 좌표 정규화
    # ==============================================
    # 사람마다 키와 어깨 너비가 다를 수 있습니다.
    # 카메라와 사람 사이의 거리도 다를 수 있습니다.
    #
    # 어깨 너비를 기준으로 좌표를 정규화하여
    # 사람의 크기 차이에 따른 영향을 줄입니다.
    #
    # 정규화된 관절 좌표를 normalized에 저장합니다.

    normalized, _, _ = (
        normalize_by_shoulder_width(
            keypoints
        )
    )

    # ==============================================
    # ⑨ 손목과 어깨의 Y좌표를 비교하여 양팔 판단
    # ==============================================
    # 정규화된 좌표에서 네 관절의 Y값을 가져옵니다.
    #
    # 이미지 좌표계는 위쪽으로 갈수록 Y값이 작습니다.
    #
    # 손목 Y < 어깨 Y : 손목이 어깨보다 위에 있음
    # 손목 Y >= 어깨 Y : 손목이 어깨보다 아래 또는 같음
    #
    # 왼손과 오른손이 모두 어깨보다 위에 있다면
    # Arms Up을 True로 판단합니다.

    left_wrist_y = (
        normalized[
            LEFT_WRIST,
            1,
        ]
    )

    right_wrist_y = (
        normalized[
            RIGHT_WRIST,
            1,
        ]
    )

    left_shoulder_y = (
        normalized[
            LEFT_SHOULDER,
            1,
        ]
    )

    right_shoulder_y = (
        normalized[
            RIGHT_SHOULDER,
            1,
        ]
    )

    left_above = (
        left_wrist_y
        < left_shoulder_y
    )

    right_above = (
        right_wrist_y
        < right_shoulder_y
    )

    arms_up = (
        left_above
        and right_above
    )

    # ==============================================
    # ⑩ 최종 판정 결과 출력
    # ==============================================
    # 분석한 이미지 파일명을 출력합니다.
    # 네 관절의 최소 Confidence를 출력합니다.
    # 왼손목과 왼어깨의 Y좌표 차이를 출력합니다.
    # 오른손목과 오른어깨의 Y좌표 차이를 출력합니다.
    #
    # Y좌표 차이가 음수이면 손목이 어깨보다 위입니다.
    #
    # Arms Up:
    # True  : 양쪽 손목이 모두 어깨보다 위에 있음
    # False : 한쪽 이상이 어깨보다 위에 있지 않음

    print(
        f"Source: {source.name}"
    )

    print(
        f"Min required confidence: "
        f"{min_conf:.3f}"
    )

    print(
        f"Left wrist - shoulder: "
        f"{left_wrist_y - left_shoulder_y:.3f}"
    )

    print(
        f"Right wrist - shoulder: "
        f"{right_wrist_y - right_shoulder_y:.3f}"
    )

    print(
        f"Arms Up: {arms_up}"
    )


# ==================================================
# 프로그램 실행 시작
# ==================================================
# 이 Python 파일을 직접 실행했을 때
# main() 함수를 호출합니다.

if __name__ == "__main__":
    main()