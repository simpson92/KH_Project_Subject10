from __future__ import annotations

# ==================================================
# 1조 7일차 테스트 — 설정 · 입력 검사 · 세로 회전 · 클래스 읽기 · Target · 로그 요약
# ==================================================
# YOLO(ultralytics · torch) 없이 돌아가도록 가짜 Pose 결과 · 가짜 모델 · 작은 MP4를 씀.
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.sequence_utils import sequence_feature_names
from src.team_realtime_utils import (
    CONFIG_PATH,
    check_model_bundle,
    ensure_portrait,
    expected_classes,
    expected_label_for,
    label_changes,
    load_json,
    option_name,
    parse_targets,
    resolve_settings,
    select_first_take,
    team_frame_feature_from_pose,
    val_clips,
    validate_runtime_config,
    warmup_seconds,
)

ROOT = Path(__file__).resolve().parents[1]
CLASSES = expected_classes()


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------
# 설정
# --------------------------------------------------
def test_team_config_is_valid():
    config = load_json(CONFIG_PATH)
    assert validate_runtime_config(config) == []
    assert config["video_subject"] == "S05"
    assert config["expect_portrait"] is True


def test_config_missing_key_is_reported():
    config = load_json(CONFIG_PATH)
    del config["confidence_threshold"]
    problems = validate_runtime_config(config)
    assert problems and "confidence_threshold" in problems[0]


def test_test_subject_is_locked():
    config = load_json(CONFIG_PATH)
    config["video_subject"] = "S02"
    assert any("Test" in p for p in validate_runtime_config(config))


def test_bad_votes_and_rotation_are_reported():
    config = load_json(CONFIG_PATH)
    config["stabilization_min_votes"] = 9
    config["landscape_rotation"] = "upside"
    problems = validate_runtime_config(config)
    assert any("min_votes" in p for p in problems)
    assert any("landscape_rotation" in p for p in problems)


def test_resolve_settings_one_change_at_a_time():
    config = load_json(CONFIG_PATH)
    base = resolve_settings(config)
    assert option_name(base) == "w5_c0.45_s5"
    assert base["stabilization_min_votes"] == 3
    # Buffer를 바꾸면 과반수 표: 3→2, 7→4 (수업 42 · 44번 규칙)
    assert resolve_settings(config, stabilization_buffer=3)["stabilization_min_votes"] == 2
    assert resolve_settings(config, stabilization_buffer=7)["stabilization_min_votes"] == 4
    assert option_name(resolve_settings(config, confidence_threshold=0.35)) == "w5_c0.35_s5"


# --------------------------------------------------
# 모델 · 클래스 (rf_training_info.json에서 읽기)
# --------------------------------------------------
class FakeModel:
    def __init__(self, classes, n_features):
        self.classes_ = np.array(classes)
        self.n_features_in_ = n_features

    def predict_proba(self, X):
        probabilities = np.full((len(X), len(self.classes_)), 0.2 / max(len(self.classes_) - 1, 1))
        probabilities[:, 0] = 0.8
        return probabilities


def _info(classes=CLASSES, steps=30):
    return {"classes": list(classes), "feature_count": steps * 38, "sequence_steps": steps}


def test_model_bundle_ok():
    model = FakeModel(CLASSES, 1140)
    assert check_model_bundle(model, sequence_feature_names(30), _info(), CLASSES) == []


def test_classes_come_from_training_info_not_code():
    # 6클래스 모델이어도 학습 정보와 맞으면 통과 (expected 미지정) → 고정값 없음
    six = ["A01", "A02", "A03", "A04", "A05", "A06"]
    assert check_model_bundle(FakeModel(six, 1140), sequence_feature_names(30), _info(six)) == []
    # 1조 16클래스와 대조하면 걸림
    problems = check_model_bundle(FakeModel(six, 1140), sequence_feature_names(30), _info(six), CLASSES)
    assert any("16클래스" in p for p in problems)


def test_model_mismatch_is_reported():
    problems = check_model_bundle(FakeModel(CLASSES[:-1], 1000), sequence_feature_names(30), _info(), CLASSES)
    assert any("모델 입력 Feature" in p for p in problems)
    assert any("모델 클래스" in p for p in problems)


def test_missing_training_info_key():
    info = _info()
    del info["sequence_steps"]
    problems = check_model_bundle(FakeModel(CLASSES, 1140), sequence_feature_names(30), info, CLASSES)
    assert any("sequence_steps" in p for p in problems)


# --------------------------------------------------
# Target (숫자키 1~6)
# --------------------------------------------------
def test_parse_targets_maps_keys_and_daily_to_n00():
    targets = parse_targets("A01, a04 ,D03,X", CLASSES)
    assert [t["key"] for t in targets] == ["1", "2", "3", "4"]
    assert [t["target_action"] for t in targets] == ["A01", "A04", "D03", "X"]
    assert [t["expected_label"] for t in targets] == ["A01", "A04", "N00", "X"]
    assert expected_label_for("D01") == "N00"


@pytest.mark.parametrize("text", ["A01,A02,A03,A04,A05,A06,A07", "A01,A01", "A99", ""])
def test_parse_targets_rejects(text):
    with pytest.raises(ValueError):
        parse_targets(text, CLASSES)


# --------------------------------------------------
# 세로 회전
# --------------------------------------------------
def test_ensure_portrait_rotates_landscape():
    landscape = np.zeros((108, 192, 3), np.uint8)
    landscape[0, 0] = 255                                       # 왼쪽 위 표시
    rotated, flag = ensure_portrait(landscape, True, "clockwise")
    assert flag and rotated.shape[:2] == (192, 108)
    assert rotated[0, -1].tolist() == [255, 255, 255]           # 시계 방향: 왼쪽 위 → 오른쪽 위
    rotated_ccw, _ = ensure_portrait(landscape, True, "counterclockwise")
    assert rotated_ccw[-1, 0].tolist() == [255, 255, 255]       # 반시계: 왼쪽 위 → 왼쪽 아래


def test_ensure_portrait_keeps_portrait_and_respects_off():
    portrait = np.zeros((192, 108, 3), np.uint8)
    assert ensure_portrait(portrait)[1] is False
    landscape = np.zeros((108, 192, 3), np.uint8)
    assert ensure_portrait(landscape, expect_portrait=False)[1] is False
    assert ensure_portrait(landscape, True, "none")[1] is False


# --------------------------------------------------
# S05 Val 영상 목록 (S02 잠금)
# --------------------------------------------------
def test_val_clips_are_s05_only_and_first_takes_are_18():
    config = load_json(CONFIG_PATH)
    clips = val_clips(config)
    assert len(clips) == 90
    assert set(clips["subject_id"]) == {"S05"}
    first, replaced = select_first_take(clips)                        # 길이 조건 없음 → 동작마다 첫 Take
    assert len(first) == 18 and replaced == []
    assert first["action_id"].tolist() == sorted(first["action_id"])
    assert set(first.loc[first["action_id"].isin(["D01", "D02", "D03"]), "train_label"]) == {"N00"}
    assert first.set_index("action_id").loc["A03", "take_id"] == "T02"   # A03은 T01이 없음


def test_short_clips_and_warmup_take_replacement():
    # S05 D01 T04 · T05 · D03 T01은 4초 미만 → 기본 Warm-up(5초 × 0.8 = 4초)이면 Prediction 0번
    config = load_json(CONFIG_PATH)
    clips = val_clips(config)
    assert warmup_seconds(config) == pytest.approx(4.0)
    short = clips[clips["duration_sec"] < warmup_seconds(config)]["clip_id"].tolist()
    assert sorted(short) == ["T01_S05_D01_T04", "T01_S05_D01_T05", "T01_S05_D03_T01"]

    chosen, replaced = select_first_take(clips, warmup_seconds(config))
    assert len(chosen) == 18
    assert (chosen["duration_sec"] >= 4.0).all()                    # 18개 모두 Prediction 1번 이상 가능
    assert replaced == [{"action_id": "D03", "from": "T01_S05_D03_T01", "from_sec": pytest.approx(3.833),
                         "to": "T01_S05_D03_T02", "to_sec": chosen.set_index("action_id").loc["D03", "duration_sec"]}]


def test_saved_video_refuses_test_clip():
    realtime = _load_script("day07_team_realtime_video")
    args = realtime.parse_args(["--clip-id", "T01_S02_A01_T01"])
    with pytest.raises(SystemExit):
        realtime.pick_clips(args, load_json(CONFIG_PATH))


# --------------------------------------------------
# 저장영상 Pipeline 1개 끝까지 (가짜 Pose · 가짜 모델 · 가로 MP4 → 회전)
# --------------------------------------------------
class _T:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return _T(self.array[index])


class _FakeResult:
    def __init__(self, keypoints):
        self.boxes = _T([0.9])
        self.boxes.conf = _T([0.9])
        self.keypoints = type("K", (), {"xy": _T(keypoints[None])})()


class _TwoPeople:
    """사람 2명: 0번 = 작은 BBox · conf 높음, 1번 = 큰 BBox · conf 낮음."""

    def __init__(self, small_points, big_points):
        self.boxes = _T([0.95, 0.60])
        self.boxes.conf = _T([0.95, 0.60])
        self.boxes.xyxy = _T([[0, 0, 50, 100], [0, 0, 400, 900]])
        self.keypoints = type("K", (), {"xy": _T(np.stack([small_points, big_points]))})()


def test_main_person_is_largest_bbox_like_training(standing_pose):
    from src.realtime_utils import frame_feature_from_pose

    small = standing_pose.astype(np.float32)
    big = small.copy()
    big[9, 1] -= 120                                                # 큰 사람만 왼손목을 듦
    result = _TwoPeople(small, big)
    team = team_frame_feature_from_pose(result, "area")             # 1조 D6
    textbook = frame_feature_from_pose(result)                      # 수업: conf 최고
    expected_big = team_frame_feature_from_pose(_FakeResult(big), "area")
    expected_small = team_frame_feature_from_pose(_FakeResult(small), "area")
    np.testing.assert_allclose(team, expected_big)                  # 큰 BBox 사람을 봄
    np.testing.assert_allclose(textbook, expected_small)            # 수업 함수는 작은(conf 높은) 사람
    assert not np.allclose(team, textbook)
    # 한 명일 때는 수업 함수와 같은 38-D (계산 자체는 같음)
    np.testing.assert_allclose(team_frame_feature_from_pose(_FakeResult(big)), frame_feature_from_pose(_FakeResult(big)))


def test_main_person_config_is_validated():
    config = load_json(CONFIG_PATH)
    assert config["main_person"] == "area"
    config["main_person"] = "tallest"
    assert any("main_person" in p for p in validate_runtime_config(config))


def test_run_clip_end_to_end(tmp_path, standing_pose):
    realtime = _load_script("day07_team_realtime_video")
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))   # 가로
    for _ in range(180):                                                                     # 6초
        writer.write(np.zeros((36, 64, 3), np.uint8))
    writer.release()

    counter = {"n": 0}

    def pose(frame):
        counter["n"] += 1
        points = standing_pose.astype(np.float32).copy()
        points[10, 1] -= counter["n"] % 20                                                 # 오른손목 위아래
        return _FakeResult(points)

    config = load_json(CONFIG_PATH)
    settings = resolve_settings(config)
    bundle = {"model": FakeModel(CLASSES, 1140), "sequence_steps": 30}
    clip = {"clip_id": "T01_S05_A01_T01", "action_id": "A01", "train_label": "A01", "video_abspath": str(video)}
    summary = realtime.run_clip(clip, settings, bundle, pose, tmp_path, False, True, "clockwise", {})

    log = pd.read_csv(summary["log"])
    assert len(log) == 180 and summary["rotated"] == 1
    assert set(log["frame_height"]) == {64} and set(log["frame_width"]) == {36}
    first_prediction = log[log["prediction_updated"] == 1]["video_time_sec"].min()
    assert first_prediction >= 4.0                                   # Warm-up: 5초 × 0.8
    assert summary["prediction_updates"] >= 5
    assert summary["final_stable"] == "A01"                          # 가짜 모델은 항상 A01 (0.8)
    assert (log.loc[log["video_time_sec"] < 3.9, "stable_label"] == "WARMING_UP").all()


# --------------------------------------------------
# 요약 (C · E)
# --------------------------------------------------
def test_label_changes():
    assert label_changes(pd.Series(["A", "A", "B", "B", "A"])) == 2
    assert label_changes(pd.Series([], dtype=str)) == 0


def _frame_log(expected, stable, conf=0.6, setting=(5.0, 0.45, 5, 3)):
    rows = []
    for index, label in enumerate(stable):
        rows.append({
            "expected_label": expected, "prediction_updated": 1, "raw_label": label,
            "threshold_label": label, "stable_label": label, "confidence": conf,
            "processing_fps": 10.0, "rotated": 0,
            "window_seconds": setting[0], "confidence_threshold": setting[1],
            "stabilization_buffer": setting[2], "stabilization_min_votes": setting[3],
        })
    rows.insert(0, {**rows[0], "prediction_updated": 0, "stable_label": "WARMING_UP"})
    return pd.DataFrame(rows)


def test_summarize_runs_compare_settings():
    summarize = _load_script("day07_team_summarize_runs")
    good = summarize.summarize_log(_frame_log("A01", ["UNKNOWN", "A01", "A01"]), "w5", "c1")
    bad = summarize.summarize_log(_frame_log("A04", ["UNKNOWN", "A06"]), "w5", "c2")
    assert good["final_stable_correct"] == 1 and good["stable_label_changes"] == 1
    assert bad["final_stable"] == "A06" and bad["final_stable_correct"] == 0
    frames = pd.DataFrame({"threshold_label": ["UNKNOWN", "A01", "A01", "UNKNOWN", "A06"],
                           "confidence": [0.6] * 5, "setting": ["w5"] * 5})
    table = summarize.compare_settings(pd.DataFrame([good, bad]), frames)
    row = table.iloc[0]
    assert row["clips"] == 2 and row["warmup_pass"] == 2 and row["final_stable_correct"] == 1
    assert row["unknown_ratio"] == pytest.approx(0.4)


def test_analyze_log_counts_segments_per_target():
    analyze = _load_script("day07_team_analyze_log")
    rows = []
    # D03 2번 수행 (segment 1 · 2): 1번은 N00으로 끝, 1번은 A11로 끝
    for segment, stable in ((1, ["N00", "N00"]), (2, ["UNKNOWN", "A11"])):
        for label in stable:
            rows.append({"target_action": "D03", "expected_label": "N00", "segment_id": segment,
                         "protocol": "with_c01", "prediction_updated": 1, "raw_label": label,
                         "threshold_label": label, "stable_label": label, "confidence": 0.5,
                         "processing_fps": 8.0, "log": "a.csv"})
    rows.append({**rows[0], "target_action": "X", "expected_label": "X", "segment_id": 3,
                 "stable_label": "A15", "raw_label": "A15"})
    table = analyze.summarize(pd.DataFrame(rows), {"D03": {"A11", "A09"}})
    d03 = table[table["target_action"] == "D03"].iloc[0]
    assert d03["segments"] == 2 and d03["final_correct"] == 1
    assert d03["top_wrong_stable"] == "A11" and d03["top_wrong_known_confusable"] == 1
    untrained = table[table["target_action"] == "X"].iloc[0]
    assert pd.isna(untrained["final_correct"]) and untrained["top_stable"] == "A15"


def test_analyze_log_reads_runtime_csv(tmp_path):
    analyze = _load_script("day07_team_analyze_log")
    log = pd.DataFrame({
        "target_action": ["", "A01", "A01"], "expected_label": ["", "A01", "A01"], "segment_id": [0, 1, 1],
        "prediction_updated": [1, 0, 1], "threshold_label": ["A02", "WARMING_UP", "A01"],
        "stable_label": ["A02", "WARMING_UP", "A01"], "raw_label": ["A02", "WARMING_UP", "A01"],
        "confidence": [0.5, 0.0, 0.7], "processing_fps": [9.0, 9.0, 9.0],
    })
    path = tmp_path / "ip_webcam_w5_c0.45_s5_with_c01_20261120_150000.csv"
    log.to_csv(path, index=False, encoding="utf-8-sig")
    usable, diagnostic = analyze.read_runtime_log(path)
    assert diagnostic == {"rows": 3, "target_rows": 2, "prediction_rows": 2, "usable_rows": 1}
    assert usable["stable_label"].tolist() == ["A01"]


def test_config_file_is_json():
    json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
