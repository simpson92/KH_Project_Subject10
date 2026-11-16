from __future__ import annotations

# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
# math  : acos(), degrees()로 각도 계산
# numpy : 관절 좌표, 벡터, 거리 계산

import math

import numpy as np


# ==================================================
# [02] COCO 17 Keypoint 번호 정의
# ==================================================
# YOLO Pose가 추출한 관절 중
# 오늘 계산에 사용할 관절 번호를 지정합니다.
#
# Left / Right는 사람 본인 기준입니다.

NOSE = 0

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_ELBOW = 7
RIGHT_ELBOW = 8

LEFT_WRIST = 9
RIGHT_WRIST = 10

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


# ==================================================
# [03] 좌우 관절 쌍 정의
# ==================================================
# 좌우 반전할 때 관절 번호도 교환해야 합니다.
#
# 예: Left Shoulder(5) ↔ Right Shoulder(6)
#
# 0번 Nose는 좌우 쌍이 없으므로 제외합니다.

LEFT_RIGHT_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]


# ==================================================
# [04] calculate_angle() - 관절각도 계산
# ==================================================
# A, B, C 세 점에서 B를 중심으로 각도를 구합니다.
#
# 예:
# A = Shoulder
# B = Elbow
# C = Wrist
#
# B에서 A 방향과 C 방향의 벡터를 만들고
# 두 벡터 사이의 각도를 계산합니다.

def calculate_angle(
    point_a,
    point_b,
    point_c,
) -> float:
    """
    point_b를 중심으로
    A-B-C 세 점의 각도를 계산합니다.
    """

    # 관절 좌표를 NumPy 배열로 변환합니다.
    a = np.asarray(
        point_a,
        dtype=np.float32,
    )

    b = np.asarray(
        point_b,
        dtype=np.float32,
    )

    c = np.asarray(
        point_c,
        dtype=np.float32,
    )

    # B를 시작점으로 두 벡터를 만듭니다.
    # BA = A - B
    # BC = C - B

    vector_ba = a - b
    vector_bc = c - b

    # 각 벡터의 길이를 계산합니다.
    norm_ba = np.linalg.norm(
        vector_ba
    )

    norm_bc = np.linalg.norm(
        vector_bc
    )

    # 벡터 길이가 0에 가까우면
    # 정상적인 각도를 계산할 수 없습니다.
    if (
        norm_ba < 1e-8
        or norm_bc < 1e-8
    ):
        return float("nan")

    # 두 벡터의 내적을 길이의 곱으로 나눠
    # cos(theta)를 계산합니다.
    cosine = np.dot(
        vector_ba,
        vector_bc,
    ) / (
        norm_ba
        * norm_bc
    )

    # 부동소수점 계산 오차를 방지하기 위해
    # cos(theta)를 -1~1로 제한합니다.
    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    # acos()로 라디안 각도를 구하고,
    # degrees()로 도(°) 단위로 변환합니다.
    angle = math.degrees(
        math.acos(
            float(cosine)
        )
    )

    return float(angle)


# ==================================================
# [05] midpoint() - 두 점의 중간 좌표 계산
# ==================================================
# 두 관절 좌표 A, B의 평균을 구합니다.
#
# 예:
# 왼쪽 골반과 오른쪽 골반의 중간점 계산

def midpoint(
    point_a,
    point_b,
):
    a = np.asarray(
        point_a,
        dtype=np.float32,
    )

    b = np.asarray(
        point_b,
        dtype=np.float32,
    )

    # 두 좌표의 평균을 반환합니다.
    return (
        a + b
    ) / 2.0


# ==================================================
# [06] body_center() - 골반 중심 계산
# ==================================================
# 왼쪽 골반(11)과 오른쪽 골반(12)의
# 중간점을 몸의 중심으로 사용합니다.
#
# [05] midpoint()를 재사용합니다.

def body_center(
    keypoints,
):
    """
    좌·우 골반의 중점을
    몸 중심으로 사용합니다.
    """

    return midpoint(
        keypoints[LEFT_HIP],
        keypoints[RIGHT_HIP],
    )


# ==================================================
# [07] shoulder_width() - 어깨 폭 계산
# ==================================================
# 왼쪽 어깨와 오른쪽 어깨 사이의
# 직선거리를 계산합니다.
#
# 이후 사람 크기를 정규화할 때
# 기준 길이로 사용합니다.

def shoulder_width(
    keypoints,
) -> float:
    """
    좌·우 어깨 사이 거리를 계산합니다.
    """

    left = np.asarray(
        keypoints[LEFT_SHOULDER],
        dtype=np.float32,
    )

    right = np.asarray(
        keypoints[RIGHT_SHOULDER],
        dtype=np.float32,
    )

    # 두 어깨 사이의 유클리드 거리 계산
    return float(
        np.linalg.norm(
            left - right
        )
    )


# ==================================================
# [08] center_keypoints() - 몸 중심 이동
# ==================================================
# 모든 Keypoint에서 골반 중심을 빼서
# 몸 중심을 원점 (0, 0)으로 이동합니다.
#
# [06] body_center()를 호출합니다.
#
# 목적:
# 사람이 화면의 왼쪽이나 오른쪽에 있어도
# 몸 기준 상대 위치를 표현합니다.

def center_keypoints(
    keypoints,
):
    """
    골반 중심이 (0, 0)이 되도록
    모든 관절 좌표를 이동합니다.
    """

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    # 골반 중심 계산
    center = body_center(
        keypoints
    )

    # 모든 관절에서 골반 중심을 뺍니다.
    centered = (
        keypoints
        - center
    )

    # 이동된 좌표와 원래 골반 중심 반환
    return centered, center


# ==================================================
# [09] normalize_by_shoulder_width()
#      - 신체 크기 정규화
# ==================================================
# [08]로 몸 중심을 이동한 뒤,
# [07]에서 계산한 어깨 폭으로 나눕니다.
#
# 목적:
# 화면 속 사람의 크기가 달라도
# 비슷한 자세를 비슷한 숫자로 표현합니다.
#
# 주의:
# 정규화 결과는 반드시 0~1 사이가 아니며
# 음수 또는 1보다 큰 값도 가능합니다.

def normalize_by_shoulder_width(
    keypoints,
):
    """
    몸 중심 이동 후
    어깨 폭을 기준으로 크기를 정규화합니다.
    """

    # 먼저 골반 중심을 원점으로 이동합니다.
    centered, center = (
        center_keypoints(
            keypoints
        )
    )

    # 정규화 기준 길이인 어깨 폭을 구합니다.
    scale = shoulder_width(
        keypoints
    )

    # 어깨 폭이 0에 가까우면 나눌 수 없습니다.
    if scale < 1e-8:
        raise ValueError(
            "어깨 폭을 계산할 수 없습니다."
        )

    # 모든 중심 이동 좌표를 어깨 폭으로 나눕니다.
    normalized = (
        centered
        / scale
    )

    # 정규화 좌표, 골반 중심, 어깨 폭 반환
    return (
        normalized,
        center,
        scale,
    )


# ==================================================
# [10] flip_keypoints_horizontal()
#      - Keypoint 좌우 반전
# ==================================================
# 정규화된 Keypoint를 좌우 반전합니다.
#
# 1. x 좌표의 부호 반전
# 2. Left / Right 관절 번호 교환
#
# 좌우 반전에서는 위치뿐 아니라
# 관절의 의미도 함께 바꾸어야 합니다.

def flip_keypoints_horizontal(
    normalized_keypoints,
):
    """
    몸 중심 기준 정규화된 좌표를
    좌우 반전합니다.

    1. x 좌표의 부호를 반전
    2. Left/Right 관절 의미도 교환
    """

    # 원본을 변경하지 않도록 복사합니다.
    flipped = np.asarray(
        normalized_keypoints,
        dtype=np.float32,
    ).copy()

    # 모든 관절의 x 좌표 부호를 반전합니다.
    flipped[:, 0] *= -1.0

    # [03]에 정의한 좌우 관절 쌍을 순회합니다.
    for left_id, right_id in (
        LEFT_RIGHT_PAIRS
    ):

        # 왼쪽 관절 좌표를 임시 저장합니다.
        temp = flipped[
            left_id
        ].copy()

        # 왼쪽 관절에 오른쪽 좌표를 넣습니다.
        flipped[left_id] = (
            flipped[right_id]
        )

        # 오른쪽 관절에 기존 왼쪽 좌표를 넣습니다.
        flipped[right_id] = (
            temp
        )

    # 좌우 반전된 Keypoint 반환
    return flipped


# ==================================================
# [11] selected_joint_angles()
#      - 팔꿈치·무릎 각도 4개 계산
# ==================================================
# [04] calculate_angle()을 네 번 호출합니다.
#
# 1. 왼쪽 팔꿈치
# 2. 오른쪽 팔꿈치
# 3. 왼쪽 무릎
# 4. 오른쪽 무릎
#
# 각도 결과를 딕셔너리로 반환합니다.

def selected_joint_angles(
    keypoints,
):
    """
    자주 사용하는 관절각도를 계산합니다.
    """

    return {

        # ------------------------------------------
        # 왼쪽 팔꿈치 각도
        # Shoulder(5) - Elbow(7) - Wrist(9)
        # ------------------------------------------
        "left_elbow_angle": (
            calculate_angle(
                keypoints[
                    LEFT_SHOULDER
                ],
                keypoints[
                    LEFT_ELBOW
                ],
                keypoints[
                    LEFT_WRIST
                ],
            )
        ),

        # ------------------------------------------
        # 오른쪽 팔꿈치 각도
        # Shoulder(6) - Elbow(8) - Wrist(10)
        # ------------------------------------------
        "right_elbow_angle": (
            calculate_angle(
                keypoints[
                    RIGHT_SHOULDER
                ],
                keypoints[
                    RIGHT_ELBOW
                ],
                keypoints[
                    RIGHT_WRIST
                ],
            )
        ),

        # ------------------------------------------
        # 왼쪽 무릎 각도
        # Hip(11) - Knee(13) - Ankle(15)
        # ------------------------------------------
        "left_knee_angle": (
            calculate_angle(
                keypoints[
                    LEFT_HIP
                ],
                keypoints[
                    LEFT_KNEE
                ],
                keypoints[
                    LEFT_ANKLE
                ],
            )
        ),

        # ------------------------------------------
        # 오른쪽 무릎 각도
        # Hip(12) - Knee(14) - Ankle(16)
        # ------------------------------------------
        "right_knee_angle": (
            calculate_angle(
                keypoints[
                    RIGHT_HIP
                ],
                keypoints[
                    RIGHT_KNEE
                ],
                keypoints[
                    RIGHT_ANKLE
                ],
            )
        ),
    }