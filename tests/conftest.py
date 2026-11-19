from __future__ import annotations

# ==================================================
# tests 공통 준비물 (pytest fixture)
# ==================================================
# 실제 사진·영상 대신 "좌표를 직접 만든 가짜 Pose"를 사용합니다.
# - 정답을 미리 알고 있는 입력이라 계산 결과를 정확히 비교할 수 있습니다.
# - YOLO·GPU·data/ 폴더가 없어도 실행됩니다.
# - 조원 얼굴이 담긴 파일이 레포에 들어갈 일이 없습니다.

import numpy as np
import pytest


@pytest.fixture
def standing_pose() -> np.ndarray:
    """정면으로 서서 팔을 옆으로 내린 가짜 COCO 17 Pose (Pixel 좌표, shape = (17, 2)).

    - 화면 기준 x는 오른쪽으로, y는 아래로 커집니다.
    - COCO Left/Right는 사람 본인 기준 → 정면 인물의 Left는 화면 오른쪽(x가 큼)
    - 어깨 폭 = 100 px, 골반 중심 = (300, 400)
    """
    return np.array(
        [
            [300, 180],  # 0  nose
            [310, 170],  # 1  left_eye
            [290, 170],  # 2  right_eye
            [320, 175],  # 3  left_ear
            [280, 175],  # 4  right_ear
            [350, 220],  # 5  left_shoulder
            [250, 220],  # 6  right_shoulder
            [360, 300],  # 7  left_elbow
            [240, 300],  # 8  right_elbow
            [365, 380],  # 9  left_wrist
            [235, 380],  # 10 right_wrist
            [330, 400],  # 11 left_hip
            [270, 400],  # 12 right_hip
            [330, 500],  # 13 left_knee
            [270, 500],  # 14 right_knee
            [330, 600],  # 15 left_ankle
            [270, 600],  # 16 right_ankle
        ],
        dtype=np.float32,
    )
