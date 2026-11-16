from __future__ import annotations

# ==================================================
# scripts/day03_plot_feature_timeline.py 테스트
# ==================================================
# 튐(Spike) 탐지 기준과 18번 CSV 읽기를 검사합니다.

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def timeline():
    pytest.importorskip("matplotlib")
    path = ROOT / "scripts" / "day03_plot_feature_timeline.py"
    spec = importlib.util.spec_from_file_location("day03_plot_feature_timeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frames(angles, frame_ids=None):
    frame_ids = frame_ids or list(range(1, len(angles) + 1))
    return pd.DataFrame(
        {
            "frame_id": frame_ids,
            "time_sec": [(f - 1) / 30 for f in frame_ids],
            "left_elbow_angle": angles,
        }
    )


def test_single_frame_spike_gives_two_events(timeline):
    # 170 → 35 → 170 : 튀는 순간 + 돌아오는 순간
    out = timeline.detect_spikes(
        frames([170, 171, 35, 169, 170]), "left_elbow_angle", 40
    )
    assert out["frame_id"].tolist() == [3, 4]
    assert out["delta"].tolist() == pytest.approx([-136, 134])
    assert out["prev_value"].tolist() == pytest.approx([171, 35])


def test_smooth_motion_is_not_spike(timeline):
    # 실제로 팔을 굽히는 동작: Frame마다 10°씩 → 튐 아님
    angles = [170 - 10 * i for i in range(16)]
    out = timeline.detect_spikes(frames(angles), "left_elbow_angle", 40)
    assert out.empty


def test_threshold_is_inclusive(timeline):
    out = timeline.detect_spikes(frames([100, 140, 100.5]), "left_elbow_angle", 40)
    assert out["frame_id"].tolist() == [2]  # |Δ| = 40 → 튐, 39.5 → 아님


def test_frame_gap_recorded(timeline):
    # valid_pose = 0 Frame(3~5)이 빠져서 2 → 6으로 건너뛴 경우
    out = timeline.detect_spikes(
        frames([170, 170, 60], frame_ids=[1, 2, 6]), "left_elbow_angle", 40
    )
    assert out["frame_gap"].tolist() == [4]


def test_load_frames_keeps_valid_and_adds_relative_feature(timeline, tmp_path):
    csv = tmp_path / "S01_ufoul_T01_frame_features.csv"
    pd.DataFrame(
        {
            "frame_id": [1, 2, 3],
            "time_sec": [0.0, 0.033, 0.067],
            "valid_pose": [1, 0, 1],
            "left_wrist_y": [0.5, None, -3.8],
            "nose_y": [-2.6, None, -2.6],
        }
    ).to_csv(csv, index=False)

    df = timeline.load_frames(csv)
    assert df["frame_id"].tolist() == [1, 3]
    assert df["left_wrist_minus_nose"].tolist() == pytest.approx([3.1, -1.2])
    assert timeline.series_label(csv) == "S01_ufoul_T01"


def test_load_frames_rejects_other_csv(timeline, tmp_path):
    csv = tmp_path / "other.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        timeline.load_frames(csv)
