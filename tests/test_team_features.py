from __future__ import annotations

# ==================================================
# src/team_feature_utils.py 테스트 — G-01 ~ G-10 (04 문서 5.2)
# ==================================================
import numpy as np
import pandas as pd

from src.pose_utils import COCO_KEYPOINT_NAMES
from src.team_feature_utils import SIGNAL_FEATURE_NAMES, add_signal_features, head_y, signal_features


def _standing():
    """정규화 좌표로 선 자세 (골반 중심 0, 어깨 폭 1, y 아래 +). 정면 → 왼쪽(COCO Left)이 화면 오른쪽."""
    p = np.zeros((17, 2))
    p[0:5] = [[0, -2.3], [0.1, -2.4], [-0.1, -2.4], [0.2, -2.35], [-0.2, -2.35]]
    p[5], p[6] = [0.5, -1.6], [-0.5, -1.6]          # 어깨
    p[7], p[8] = [0.55, -0.8], [-0.55, -0.8]        # 팔꿈치
    p[9], p[10] = [0.6, 0.0], [-0.6, 0.0]           # 손목
    p[11], p[12] = [0.3, 0.0], [-0.3, 0.0]          # 골반
    p[13], p[14] = [0.3, 1.0], [-0.3, 1.0]
    p[15], p[16] = [0.3, 2.0], [-0.3, 2.0]          # 발목
    return p


def test_standing_values():
    g = dict(zip(SIGNAL_FEATURE_NAMES, signal_features(_standing())))
    assert np.isclose(g["wrist_distance"], 1.2)
    assert g["r_wrist_height"] < -2 and g["l_wrist_height"] < -2       # 손이 머리보다 한참 아래
    assert g["wrist_cross"] == 1.0
    assert np.isclose(g["wrist_hip_dist"], 0.3)
    assert np.isclose(g["ankle_spread"], 0.6)
    assert g["r_shoulder_angle"] < 15                                   # 팔을 내림


def test_hand_up_and_crossed_wrists():
    p = _standing()
    p[8], p[10] = [-0.6, -2.4], [-0.6, -3.4]                            # 오른팔 머리 위 (C01)
    g = dict(zip(SIGNAL_FEATURE_NAMES, signal_features(p)))
    assert g["r_wrist_height"] > 1 and g["r_elbow_over_shoulder"] > 0
    assert g["r_shoulder_angle"] > 150
    p = _standing()
    p[9], p[10] = [-0.2, -1.0], [0.2, -1.0]                             # 손목이 서로 반대편 (X자)
    assert signal_features(p)[SIGNAL_FEATURE_NAMES.index("wrist_cross")] == -1.0


def test_head_uses_confident_points_only():
    p = _standing()
    p[0, 1] = -5.0                                                      # 코가 크게 튐
    conf = np.ones(17)
    conf[0] = 0.1
    assert np.isclose(head_y(p, conf), np.mean(p[1:5, 1]))


def test_add_signal_features_keeps_nan_for_failed_frames():
    p = _standing()
    rows = []
    for ok in (1, 0):
        row = {"feature_ok": ok}
        for i, name in enumerate(COCO_KEYPOINT_NAMES):
            row[f"{name}_x"], row[f"{name}_y"] = p[i]
            row[f"conf_{name}"] = 0.9
        rows.append(row)
    out = add_signal_features(pd.DataFrame(rows))
    assert list(out.columns[-10:]) == SIGNAL_FEATURE_NAMES
    assert out.loc[0, SIGNAL_FEATURE_NAMES].notna().all()
    assert out.loc[1, SIGNAL_FEATURE_NAMES].isna().all()
