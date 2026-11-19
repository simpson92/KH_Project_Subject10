from __future__ import annotations

import numpy as np


# ① COCO 17개 Keypoint의 이름 정의
# YOLO Pose가 검출한 관절 번호(0~16)에 해당하는 이름
COCO_KEYPOINT_NAMES = [
    "nose",            # 0: 코
    "left_eye",        # 1: 왼쪽 눈
    "right_eye",       # 2: 오른쪽 눈
    "left_ear",        # 3: 왼쪽 귀
    "right_ear",       # 4: 오른쪽 귀
    "left_shoulder",   # 5: 왼쪽 어깨
    "right_shoulder",  # 6: 오른쪽 어깨
    "left_elbow",      # 7: 왼쪽 팔꿈치
    "right_elbow",     # 8: 오른쪽 팔꿈치
    "left_wrist",      # 9: 왼쪽 손목
    "right_wrist",     # 10: 오른쪽 손목
    "left_hip",       # 11: 왼쪽 골반
    "right_hip",      # 12: 오른쪽 골반
    "left_knee",      # 13: 왼쪽 무릎
    "right_knee",     # 14: 오른쪽 무릎
    "left_ankle",     # 15: 왼쪽 발목
    "right_ankle",    # 16: 오른쪽 발목
]

'''
VICON에서 추출한 JSON이 다음과 같다고 가정하겠습니다.
{
  "LeftArm": [120, 250, 30],
  "LeftForeArm": [140, 300, 35],
  "LeftHand": [160, 350, 40],
  "LeftUpLeg": [110, 400, 20]
}

VICON 측 관절 이름과 COCO 17의 이름이 다르거나 일치하지 않는다면 COCO 17의 관절 이름과 순서에 맞도록 변환해야 합니다.
'''

# ② Skeleton을 그릴 관절 연결 관계 정의
# (시작 관절 번호, 끝 관절 번호) 형태로 작성
COCO_SKELETON = [
    (15, 13),  # 왼쪽 발목 ↔ 왼쪽 무릎
    (13, 11),  # 왼쪽 무릎 ↔ 왼쪽 골반
    (16, 14),  # 오른쪽 발목 ↔ 오른쪽 무릎
    (14, 12),  # 오른쪽 무릎 ↔ 오른쪽 골반
    (11, 12),  # 왼쪽 골반 ↔ 오른쪽 골반
    (5, 11),   # 왼쪽 어깨 ↔ 왼쪽 골반
    (6, 12),   # 오른쪽 어깨 ↔ 오른쪽 골반
    (5, 6),    # 왼쪽 어깨 ↔ 오른쪽 어깨
    (5, 7),    # 왼쪽 어깨 ↔ 왼쪽 팔꿈치
    (6, 8),    # 오른쪽 어깨 ↔ 오른쪽 팔꿈치
    (7, 9),    # 왼쪽 팔꿈치 ↔ 왼쪽 손목
    (8, 10),   # 오른쪽 팔꿈치 ↔ 오른쪽 손목
    (1, 2),    # 왼쪽 눈 ↔ 오른쪽 눈
    (0, 1),    # 코 ↔ 왼쪽 눈
    (0, 2),    # 코 ↔ 오른쪽 눈
    (1, 3),    # 왼쪽 눈 ↔ 왼쪽 귀
    (2, 4),    # 오른쪽 눈 ↔ 오른쪽 귀
    (3, 5),    # 왼쪽 귀 ↔ 왼쪽 어깨
    (4, 6),    # 오른쪽 귀 ↔ 오른쪽 어깨
]


# ③ 관절 Confidence 판정 함수 정의
def confidence_state(
    confidence: float,
    valid_threshold: float = 0.50,
    warning_threshold: float = 0.30,
) -> str:

    # ④ Confidence 값에 따라 상태 반환

    # 0.50 이상이면 정상 신뢰도
    if confidence >= valid_threshold:
        return "VALID"

    # 0.30 이상 ~ 0.50 미만이면 주의
    if confidence >= warning_threshold:
        return "WARNING"

    # 0.30 미만이면 낮은 신뢰도
    return "LOW"


# ⑤ 여러 사람이 검출된 화면에서 대표 인물 한 명 고르기
# 1조 결정 D6 (docs/referee/01_관절_매핑_명세.md, 2026-11-13)
#   "area"  : BBox 넓이가 가장 큰 사람 (D6 결정값, 심판이 카메라에 가장 가깝다는 전제)
#   "conf"  : BBox Confidence가 가장 높은 사람 (수업 09번 코드 방식)
#   "first" : YOLO 결과의 첫 번째 사람 (교재 12~18번 코드 방식, 교재 결과 재현용)
MAIN_PERSON_MODES = ("area", "conf", "first")


def _to_numpy(value) -> np.ndarray:
    # torch Tensor(GPU 포함)와 NumPy 배열을 모두 받기 위한 변환
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def select_main_person(result, mode: str = "area") -> int | None:
    """YOLO Pose 결과(result)에서 대표 인물의 번호를 반환합니다.

    사람이 없으면 None을 반환합니다.
    반환한 번호로 result.keypoints.xy[번호], result.keypoints.conf[번호]를 꺼내 씁니다.
    """
    if mode not in MAIN_PERSON_MODES:
        raise ValueError(f"mode는 {MAIN_PERSON_MODES} 중 하나여야 합니다: {mode}")

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None

    people = len(keypoints.xy)
    if mode == "first" or people == 1:
        return 0

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) != people:
        # BBox 정보가 Keypoint와 맞지 않으면 교재 방식(첫 번째 사람)으로 처리
        return 0

    if mode == "conf":
        return int(np.argmax(_to_numpy(boxes.conf)))

    xyxy = _to_numpy(boxes.xyxy)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))
