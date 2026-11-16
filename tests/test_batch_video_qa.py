from __future__ import annotations

# ==================================================
# scripts/day03_batch_video_qa.py 요약 로직 테스트
# ==================================================
# YOLO 없이, 18번 CSV 모양의 가짜 Frame 표로 영상 판정을 검사합니다.

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    pytest.importorskip("matplotlib")
    stubbed = []
    for name in ("cv2", "torch", "ultralytics"):
        if importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            module.YOLO = None
            sys.modules[name] = module
            stubbed.append(name)
    yield (
        _load("day03_batch_video_qa.py"),
        _load("17_referee_rule.py"),
        _load("day03_plot_feature_timeline.py"),
    )
    for name in stubbed:
        sys.modules.pop(name, None)


ARGS = argparse.Namespace(max_wrist_dist=1.0, spike_deg=40.0, min_run_sec=0.3)


def fake_frames(n=60, ufoul_frames=range(0), people=1, shoulder=200.0):
    """정규화 좌표 기준 가짜 18번 Frame 표 (30 FPS)."""
    rows = []
    for i in range(n):
        up = i in ufoul_frames
        rows.append(
            {
                "frame_id": i + 1,
                "time_sec": i / 30,
                "valid_pose": 1,
                "mean_keypoint_conf": 0.95,
                "low_keypoint_count": 0,
                "shoulder_width": shoulder,
                "nose_y": -2.6,
                "left_wrist_x": 0.2 if up else 0.7,
                "left_wrist_y": -3.7 if up else 0.0,
                "right_wrist_x": -0.2 if up else -0.7,
                "right_wrist_y": -3.7 if up else 0.0,
                "left_elbow_angle": 140.0 if up else 170.0,
                "right_elbow_angle": 140.0 if up else 170.0,
                "people": people,
            }
        )
    return pd.DataFrame(rows)


def test_longest_true_run(modules):
    batch, _, _ = modules
    assert batch.longest_true_run([True, True, False, True, True, True, False]) == 3
    assert batch.longest_true_run([]) == 0


def test_ufoul_video_pass(modules):
    batch, s17, timeline = modules
    row = batch.summarize_video(
        "S01_ufoul_T01", fake_frames(ufoul_frames=range(20, 40)), ARGS, s17, timeline
    )
    assert row["subject"] == "S01" and row["action"] == "ufoul"
    assert row["ufoul_true_frames"] == 20
    assert row["longest_true_run_sec"] == pytest.approx(0.67, abs=0.01)
    assert row["rule_pred"] == "ufoul" and row["rule_correct"]
    assert row["status"] == "PASS"


def test_short_flicker_is_not_ufoul(modules):
    # 0.3초(9 Frame) 미만으로 잠깐 True → 영상 판정은 other
    batch, s17, timeline = modules
    row = batch.summarize_video(
        "S02_ufoul_T03", fake_frames(ufoul_frames=range(20, 25)), ARGS, s17, timeline
    )
    assert row["rule_pred"] == "other"
    assert not row["rule_correct"]
    assert row["status"] == "REVIEW" and "규칙 오판" in row["reasons"]


def test_pushing_video_pass(modules):
    batch, s17, timeline = modules
    row = batch.summarize_video("S03_pushing_T02", fake_frames(), ARGS, s17, timeline)
    assert row["rule_pred"] == "other" and row["rule_correct"]
    assert row["best_both_wrist_minus_nose"] > 0
    assert row["status"] == "PASS"


def test_multi_person_and_invalid_frames_go_to_review(modules):
    batch, s17, timeline = modules
    frames = fake_frames(people=2)
    frames.loc[:9, "valid_pose"] = 0  # 60 Frame 중 10개 실패 → 83%
    row = batch.summarize_video("S02_pushing_T06", frames, ARGS, s17, timeline)
    assert row["valid_ratio"] == pytest.approx(0.833, abs=0.001)
    assert row["multi_person_frames"] == 60
    assert row["status"] == "REVIEW"
    assert "다른 사람" in row["reasons"] and "95%" in row["reasons"]


# --------------------------------------------------
# 파일 이름 → ID (원본 촬영 이름도 인식)
# --------------------------------------------------
def test_identify_original_and_team_names(modules, tmp_path):
    batch, _, _ = modules
    rename = _load("referee_00_rename_videos.py")
    rename.SUBJECT_CSV = tmp_path / "no_subjects.csv"  # 대응표 없음 → 가나다순
    videos = [
        Path("raw/s05 푸싱-라마바 (10).mp4"),
        Path("raw/s03 u파울-가나다 (1).mp4"),
        Path("raw/S03_ufoul_T02.mp4"),
        Path("raw/random_clip.mp4"),
    ]
    ids = batch.identify_videos(videos, rename)
    assert ids[videos[0]]["video_id"] == "S02_pushing_T10"
    assert ids[videos[1]]["video_id"] == "S01_ufoul_T01"
    assert ids[videos[2]]["video_id"] == "S03_ufoul_T02"
    assert ids[videos[3]] == {
        "video_id": "random_clip",
        "subject": "",
        "action": "",
        "take": "",
    }


def test_identify_uses_subjects_csv(modules, tmp_path):
    batch, _, _ = modules
    rename = _load("referee_00_rename_videos.py")
    table = tmp_path / "subjects.csv"
    table.write_text("subject_id,name\nS04,가나다\n", encoding="utf-8-sig")
    rename.SUBJECT_CSV = table
    video = Path("raw/s03 u파울-가나다 (3).mp4")
    assert batch.identify_videos([video], rename)[video]["video_id"] == "S04_ufoul_T03"


def test_unknown_action_is_not_counted_as_wrong(modules):
    batch, s17, timeline = modules
    row = batch.summarize_video("random_clip", fake_frames(), ARGS, s17, timeline)
    assert row["rule_correct"] is None
    assert "동작을 알 수 없음" in row["reasons"]
    assert "오판" not in row["reasons"]
