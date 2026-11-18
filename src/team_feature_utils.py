from __future__ import annotations

# ==================================================
# team_feature_utils.py
# 1조 신호 특화 Feature G-01 ~ G-10 (04_특징_모델_설계서 5.2) — 38-D + 10 = 48-D
# ==================================================
# 입력: 정규화된 17개 관절 (골반 중심 0, 어깨 폭 1, y는 아래로 +) — STEP 4 Sequence CSV의 <관절>_x · _y
#   - 거리는 이미 어깨 폭으로 나뉜 값, 높이는 −y 기준 (위로 갈수록 +)
#   - H(머리 중심) = 0~4번 중 conf ≥ 0.50 점의 평균 (01 문서 M-05). 모두 낮으면 0~4번 전체 평균
#   - 수업 src/feature_utils.py 는 고치지 않고 calculate_angle 만 불러 씀 (FR-02)
#   - 각도를 못 구하면 NaN 그대로 (FR-05), feature_ok = 0 Frame은 전부 NaN (FR-01)
# 학습 · 추론이 같은 함수를 써야 함 (6일차 RF · 7일차 FastAPI 공용)

import numpy as np
import pandas as pd

from src.feature_utils import calculate_angle
from src.pose_utils import COCO_KEYPOINT_NAMES

SIGNAL_FEATURE_NAMES = [
    "wrist_distance",          # G-01 ‖P9 − P10‖
    "r_wrist_height",          # G-02 −(y10 − yH)
    "l_wrist_height",          # G-03 −(y9 − yH)
    "r_wrist_shoulder_dist",   # G-04 ‖P10 − P6‖
    "r_elbow_over_shoulder",   # G-05 −(y8 − y6)
    "r_shoulder_angle",        # G-06 ∠(12-6-8)
    "l_shoulder_angle",        # G-07 ∠(11-5-7)
    "wrist_cross",             # G-08 sign(x9 − x10): +1 정상, −1 교차
    "wrist_hip_dist",          # G-09 (‖P9 − P11‖ + ‖P10 − P12‖) / 2
    "ankle_spread",            # G-10 ‖P16 − P15‖
]
HEAD_IDS = [0, 1, 2, 3, 4]


def head_y(points: np.ndarray, conf: np.ndarray | None = None, kp_conf: float = 0.50) -> float:
    """H(머리 중심)의 y — 01 문서 M-05."""
    ys = points[HEAD_IDS, 1].astype(float)
    if conf is not None:
        good = np.asarray(conf, dtype=float)[HEAD_IDS] >= kp_conf
        if good.any():
            return float(np.nanmean(ys[good]))
    return float(np.nanmean(ys)) if np.isfinite(ys).any() else float("nan")


def signal_features(points: np.ndarray, conf: np.ndarray | None = None,
                    kp_conf: float = 0.50) -> np.ndarray:
    """정규화 좌표 (17, 2) → G-01 ~ G-10 (10,)."""
    p = np.asarray(points, dtype=float)

    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(p[i] - p[j]))

    yh = head_y(p, conf, kp_conf)
    cross = p[9, 0] - p[10, 0]
    return np.array([
        dist(9, 10),
        -(p[10, 1] - yh),
        -(p[9, 1] - yh),
        dist(10, 6),
        -(p[8, 1] - p[6, 1]),
        calculate_angle(p[12], p[6], p[8]),
        calculate_angle(p[11], p[5], p[7]),
        float(np.sign(cross)) if np.isfinite(cross) else float("nan"),
        (dist(9, 11) + dist(10, 12)) / 2,
        dist(16, 15),
    ], dtype=float)


def add_signal_features(seq: pd.DataFrame, kp_conf: float = 0.50) -> pd.DataFrame:
    """STEP 4 Sequence DataFrame에 G-01~G-10 열 10개를 붙인 사본."""
    xy = np.stack([seq[[f"{n}_x", f"{n}_y"]].to_numpy(dtype=float) for n in COCO_KEYPOINT_NAMES], 1)
    conf_cols = [f"conf_{n}" for n in COCO_KEYPOINT_NAMES]
    conf = seq[conf_cols].to_numpy(dtype=float) if all(c in seq for c in conf_cols) else None
    ok = seq["feature_ok"].to_numpy() == 1 if "feature_ok" in seq else np.ones(len(seq), bool)
    values = np.full((len(seq), len(SIGNAL_FEATURE_NAMES)), np.nan)
    for i in np.where(ok)[0]:
        values[i] = signal_features(xy[i], None if conf is None else conf[i], kp_conf)
    out = seq.copy()
    for k, name in enumerate(SIGNAL_FEATURE_NAMES):
        out[name] = np.round(values[:, k], 6)
    return out
