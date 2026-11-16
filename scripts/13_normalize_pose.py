from __future__ import annotations

# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
# argparse : 실행 옵션 처리
# csv      : CSV 파일 저장
# sys      : 모듈 검색 경로 설정
# Path     : 파일 및 폴더 경로 처리
# numpy    : 관절 좌표 계산
# YOLO     : 사람의 Keypoint 추출

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ==================================================
# [02] 프로젝트 ROOT 경로 설정
# ==================================================
# 현재 스크립트의 상위 폴더에서
# 프로젝트 최상위 경로를 찾습니다.
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
# → 설정값과 CPU/GPU 확인
#
# feature_utils.py
# → 골반 중심 및 어깨 폭 정규화 계산
#
# pose_utils.py
# → COCO 17 Keypoint의 이름 확인

from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    body_center,
    normalize_by_shoulder_width,
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
# → Confidence 기준값을 전달받는 옵션
#
# 주의:
# 현재 코드에서는 --kp-conf를 사용해
# 관절을 제외하거나 정규화를 건너뛰지 않습니다.

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
    # 실행 옵션에서 이미지 경로를 가져옵니다.
    #
    # 해당 경로가 없다면
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
    # → 설정 파일의 모델 경로와 입력 크기 확인
    #
    # resolve_device()
    # → CPU 또는 GPU 실행 장치 결정

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    # ==============================================
    # [07] 사전학습된 YOLO Pose 모델 불러오기
    # ==============================================
    # 설정 파일에 지정된 Pose 모델을
    # 불러옵니다.
    #
    # 이번 실습에서는 새롭게 학습하지 않고
    # 사전학습된 모델로 추론합니다.

    model = YOLO(
        config["pose_model"]
    )

    # ==============================================
    # [08] YOLO Pose로 이미지 추론
    # ==============================================
    # source : 분석할 이미지 경로
    # imgsz  : 모델 입력 이미지 크기
    # conf   : 사람 검출 Confidence 기준
    # device : CPU 또는 GPU
    #
    # 사람을 검출한 후
    # COCO 17 Keypoint를 추정합니다.

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
    # 사람의 Keypoint가 검출되지 않으면
    # 좌표 정규화를 진행할 수 없습니다.
    #
    # 안내 메시지를 출력하고 종료합니다.

    if (
        result.keypoints is None
        or result.keypoints.xy
        is None
        or len(
            result.keypoints.xy
        ) == 0
    ):
        print(
            "Keypoint를 찾지 "
            "못했습니다."
        )
        return

    # ==============================================
    # [10] 첫 번째 사람의 Pixel 좌표 가져오기
    # ==============================================
    # result.keypoints.xy[0]
    # → 첫 번째 사람의 17개 관절 좌표
    #
    # cpu()
    # → CPU로 이동
    #
    # numpy()
    # → NumPy 배열로 변환
    #
    # 좌표 형태:
    # [[x0, y0], [x1, y1], ..., [x16, y16]]

    keypoints = (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )

    # ==============================================
    # [11] Keypoint Confidence 가져오기
    # ==============================================
    # 관절마다 추정된 Confidence를 가져옵니다.
    #
    # Confidence가 없다면
    # 17개 관절 모두 1.0으로 채웁니다.
    #
    # 주의:
    # 이 경우 1.0은 임시 대체값이며
    # 실제 검출 신뢰도가 1.0이라는 뜻은 아닙니다.

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
    # [12] 골반 중심 좌표 계산
    # ==============================================
    # feature_utils.py의 body_center()를 사용합니다.
    #
    # Left Hip(11), Right Hip(12)의 중간점을
    # 몸 중심으로 계산합니다.
    #
    # center = (center_x, center_y)

    center = body_center(
        keypoints
    )

    # ==============================================
    # [13] 어깨 폭 기준 좌표 정규화
    # ==============================================
    # normalize_by_shoulder_width()를 호출합니다.
    #
    # 내부 처리:
    # 1. 골반 중심 계산
    # 2. 모든 관절을 골반 중심으로 이동
    # 3. 왼쪽·오른쪽 어깨 사이 거리 계산
    # 4. 이동된 좌표를 어깨 폭으로 나누기
    #
    # 반환값:
    # normalized  : 정규화된 관절 좌표
    # used_center : 정규화에 사용된 골반 중심
    # scale       : 어깨 폭
    #
    # 정규화 좌표는 0~1 범위로 제한되지 않습니다.

    normalized, (
        used_center
    ), scale = (
        normalize_by_shoulder_width(
            keypoints
        )
    )

    # ==============================================
    # [14] 중심 이동 좌표 계산
    # ==============================================
    # 원본 Pixel 좌표에서
    # 골반 중심 좌표를 뺍니다.
    #
    # centered_x = pixel_x - center_x
    # centered_y = pixel_y - center_y
    #
    # 골반 중심이 (0, 0)이 되도록
    # 좌표를 이동한 결과입니다.

    centered = (
        keypoints
        - used_center
    )

    # ==============================================
    # [15] CSV 저장 경로 생성
    # ==============================================
    # reports/day03/tables 폴더에
    # 정규화 결과를 저장합니다.
    #
    # 출력 파일 예시:
    # arms_up_person_a_normalization.csv
    #
    # 저장 폴더가 없으면 자동 생성합니다.

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / (
            f"{source.stem}"
            "_normalization.csv"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # [16] CSV 열 이름 정의
    # ==============================================
    # keypoint_id   : COCO 관절 번호
    # keypoint_name : 관절 이름
    # confidence    : 관절 Confidence
    # pixel_x/y     : 원본 이미지 좌표
    # centered_x/y  : 골반 중심 이동 좌표
    # normalized_x/y: 어깨 폭 정규화 좌표

    fieldnames = [
        "keypoint_id",
        "keypoint_name",
        "confidence",
        "pixel_x",
        "pixel_y",
        "centered_x",
        "centered_y",
        "normalized_x",
        "normalized_y",
    ]

    # ==============================================
    # [17] 17개 관절의 CSV 데이터 만들기
    # ==============================================
    # COCO 17 Keypoint를 하나씩 반복합니다.
    #
    # 각 관절에서 다음 값을 가져옵니다.
    #
    # 1. 관절 번호와 이름
    # 2. 원본 Pixel 좌표
    # 3. 중심 이동 좌표
    # 4. 정규화 좌표
    # 5. Confidence
    #
    # 각 관절 데이터를 딕셔너리로 만들고
    # rows 리스트에 추가합니다.

    rows = []

    for keypoint_id, name in (
        enumerate(
            COCO_KEYPOINT_NAMES
        )
    ):

        # 원본 Pixel 좌표
        x, y = keypoints[
            keypoint_id
        ]

        # 골반 중심 이동 좌표
        cx, cy = centered[
            keypoint_id
        ]

        # 어깨 폭 정규화 좌표
        nx, ny = normalized[
            keypoint_id
        ]

        # 현재 관절의 Confidence
        conf = confidence[
            keypoint_id
        ]

        # 하나의 관절 데이터를 딕셔너리로 추가
        rows.append(
            {
                "keypoint_id": (
                    keypoint_id
                ),
                "keypoint_name": name,

                "confidence": round(
                    float(conf),
                    4,
                ),

                "pixel_x": round(
                    float(x),
                    2,
                ),

                "pixel_y": round(
                    float(y),
                    2,
                ),

                "centered_x": round(
                    float(cx),
                    4,
                ),

                "centered_y": round(
                    float(cy),
                    4,
                ),

                "normalized_x": round(
                    float(nx),
                    6,
                ),

                "normalized_y": round(
                    float(ny),
                    6,
                ),
            }
        )

    # ==============================================
    # [18] CSV 파일 저장
    # ==============================================
    # output_path 경로에 CSV 파일을 생성합니다.
    #
    # encoding="utf-8-sig"
    # → Excel에서 한글이 깨지는 문제를 줄입니다.
    #
    # writeheader()
    # → CSV 열 이름 기록
    #
    # writerows()
    # → 17개 관절 데이터 기록

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
    # [19] 분석 결과 요약 출력
    # ==============================================
    # 분석 이미지 이름을 출력합니다.
    #
    # Hip center
    # → 좌·우 골반 중간점의 Pixel 좌표
    #
    # Shoulder width
    # → 좌·우 어깨 사이 Pixel 거리

    print(
        f"Source: {source.name}"
    )

    print(
        "Hip center:",
        f"({center[0]:.2f}, "
        f"{center[1]:.2f})"
    )

    print(
        f"Shoulder width: "
        f"{scale:.2f}"
    )

    # ==============================================
    # [20] 주요 관절의 정규화 결과 출력
    # ==============================================
    # 전체 17개 관절 중 8개만 선택합니다.
    #
    # 5, 6   : 왼쪽·오른쪽 어깨
    # 9, 10  : 왼쪽·오른쪽 손목
    # 11, 12 : 왼쪽·오른쪽 골반
    # 15, 16 : 왼쪽·오른쪽 발목
    #
    # 원본 Pixel 좌표와
    # 정규화된 좌표를 나란히 출력합니다.

    print()
    print(
        "Selected Keypoints"
    )

    for index in [
        5, 6,
        9, 10,
        11, 12,
        15, 16,
    ]:

        # 현재 관절의 이름 가져오기
        name = (
            COCO_KEYPOINT_NAMES[
                index
            ]
        )

        # 원본 Pixel 좌표
        x, y = keypoints[index]

        # 어깨 폭 정규화 좌표
        nx, ny = normalized[
            index
        ]

        # 관절 번호, 이름, 좌표 비교 출력
        print(
            f"{index:02d} "
            f"{name:16s} "
            f"pixel=({x:7.1f}, "
            f"{y:7.1f}) "
            f"normalized="
            f"({nx:7.3f}, "
            f"{ny:7.3f})"
        )

    # ==============================================
    # [21] CSV 파일 저장 위치 출력
    # ==============================================
    # CSV가 저장된 전체 경로를 출력하여
    # 학생들이 결과 파일을 확인하도록 합니다.

    print()
    print(
        f"Saved: {output_path}"
    )


# ==================================================
# [22] Python 프로그램 실행 시작점
# ==================================================
# 이 파일을 직접 실행하면
# main() 함수를 호출합니다.

if __name__ == "__main__":
    main()