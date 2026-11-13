from __future__ import annotations


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