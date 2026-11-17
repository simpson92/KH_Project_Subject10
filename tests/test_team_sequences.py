from __future__ import annotations

# ==================================================
# scripts/day04_build_team_sequences.py 테스트
# ==================================================
# YOLO 대신 "정답을 아는 가짜 Pose 결과"를 넣어 확인합니다.
# - clip_id별 저장(덮어쓰기 없음) · 실제 FPS로 time_sec · 대표 인물(D6 가장 큰 BBox)
# - 38-D 열 이름이 교재 22·23번과 같아서 23번 요약 코드가 그대로 읽히는지
# - 실패 Frame도 Row로 남는지, Confidence가 없을 때 1.0으로 채우지 않는지
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load(filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("day04_build_team_sequences.py")


# ---------- 가짜 YOLO 결과 ----------
def _result(people: list[np.ndarray], boxes=None, conf=True):
    """people: Pixel 좌표 (17, 2) 배열 목록. 빈 목록이면 사람 없음."""
    if not people:
        keypoints = types.SimpleNamespace(xy=np.zeros((0, 17, 2)), conf=None)
        return types.SimpleNamespace(keypoints=keypoints, boxes=None)
    xy = np.stack(people).astype(np.float32)
    kp_conf = np.full((len(people), 17), 0.9, np.float32) if conf else None
    if boxes is None:
        boxes = [
            [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in people
        ]
    box_ns = types.SimpleNamespace(
        xyxy=np.asarray(boxes, np.float32),
        conf=np.full(len(people), 0.8, np.float32),
    )
    return types.SimpleNamespace(
        keypoints=types.SimpleNamespace(xy=xy, conf=kp_conf), boxes=box_ns
    )


class FakeModel:
    """Frame마다 같은 가짜 결과를 돌려주는 YOLO 대용."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        return [self.result]


BASE = {"clip_id": "T01_S01_A03_T01", "frame_id": 0}


def test_columns_match_textbook_22_23(mod):
    assert len(mod.FEATURE_COLUMNS) == 38
    assert mod.FEATURE_COLUMNS[:2] == ["nose_x", "nose_y"]
    assert mod.FEATURE_COLUMNS[-4:] == list(mod.ANGLE_NAMES)
    day23 = _load("23_summarize_action_sequences.py")
    assert day23.coordinate_columns() == mod.FEATURE_COLUMNS[:34]
    assert len(mod.RAW_COLUMNS) == 17 * 3


def test_no_person_keeps_row(mod):
    row = mod.frame_row(_result([]), BASE, 0.5, "area")
    assert row["pose_ok"] == 0 and row["feature_ok"] == 0
    assert row["fail_reason"] == "no_person"
    assert row["nose_x"] == ""


def test_normal_frame_matches_feature_utils(mod, standing_pose):
    from src.feature_utils import normalize_by_shoulder_width

    row = mod.frame_row(_result([standing_pose]), BASE, 0.5, "area")
    assert row["pose_ok"] == 1 and row["feature_ok"] == 1
    assert row["fail_reason"] == ""
    normalized, _center, scale = normalize_by_shoulder_width(standing_pose)
    assert row["left_wrist_y"] == pytest.approx(float(normalized[9, 1]), abs=1e-6)
    assert row["shoulder_width"] == pytest.approx(scale)
    assert (row["hip_center_x"], row["hip_center_y"]) == (300.0, 400.0)
    assert row["raw_nose_x"] == 300.0 and row["conf_nose"] == pytest.approx(0.9)
    assert row["norm_joint_low_count"] == 0


def test_main_person_is_largest_bbox(mod, standing_pose):
    small = standing_pose * 0.5  # 멀리 있는 사람 (BBox가 작음)
    result = _result([small, standing_pose])
    assert mod.frame_row(result, BASE, 0.5, "area")["person_index"] == 1
    assert mod.frame_row(result, BASE, 0.5, "first")["person_index"] == 0
    assert mod.frame_row(result, BASE, 0.5, "area")["people"] == 2


def test_zero_shoulder_width_is_feature_fail(mod, standing_pose):
    pose = standing_pose.copy()
    pose[6] = pose[5]  # 두 어깨가 같은 점 → 어깨 폭 0
    row = mod.frame_row(_result([pose]), BASE, 0.5, "area")
    assert row["pose_ok"] == 1 and row["feature_ok"] == 0
    assert row["fail_reason"] == "shoulder_width_zero"


def test_missing_confidence_is_not_one(mod, standing_pose):
    row = mod.frame_row(_result([standing_pose], conf=False), BASE, 0.5, "area")
    assert row["conf_nose"] == "" and row["mean_keypoint_conf"] == ""
    assert row["feature_ok"] == 1


# ---------- 전체 실행 (가짜 영상 + 가짜 model) ----------
def _write_video(path: Path, frames: int, fps: float) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 96))
    for i in range(frames):
        writer.write(np.full((96, 64, 3), i, np.uint8))
    writer.release()


@pytest.fixture
def team(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    rows = []
    for clip_id, frames, fps in (
        ("T01_S01_A03_T01", 6, 30.0),
        ("T01_S01_A03_T02", 5, 25.0),  # 같은 동작 → 22번이면 덮어쓰던 경우
        ("T01_S01_A05_T01", 4, 30.0),
    ):
        _write_video(video_dir / f"{clip_id}.mp4", frames, fps)
        _, subject, action, take = clip_id.split("_")
        rows.append(
            {
                "clip_id": clip_id,
                "subject_id": subject,
                "action_id": action,
                "action_label": "U파울" if action == "A03" else "푸싱",
                "take_id": take,
                "video_file": f"{clip_id}.mp4",
                "source_type": "phone",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    out = tmp_path / "sequences"
    argv = ["--manifest", str(manifest), "--video-dir", str(video_dir)]
    return argv + ["--output-dir", str(out)], out


def test_main_saves_one_csv_per_clip(mod, team, standing_pose):
    argv, out = team
    assert mod.main(argv, model=FakeModel(_result([standing_pose]))) == 0
    names = sorted(p.name for p in out.glob("*.csv"))
    assert names == [
        "T01_S01_A03_T01_sequence.csv",
        "T01_S01_A03_T02_sequence.csv",
        "T01_S01_A05_T01_sequence.csv",
    ]
    seq = pd.read_csv(out / "T01_S01_A03_T02_sequence.csv")
    assert list(seq.columns) == mod.ALL_COLUMNS
    assert len(seq) == 5 and seq["feature_ok"].all()
    assert seq["time_sec"].iloc[1] == pytest.approx(0.04)  # 25 fps → 영상 자신의 FPS


def test_filters_and_skip_existing(mod, team, standing_pose):
    argv, out = team
    model = FakeModel(_result([standing_pose]))
    mod.main(argv + ["--clip-id", "T01_S01_A03_T01"], model=model)
    assert [p.name for p in out.glob("*.csv")] == ["T01_S01_A03_T01_sequence.csv"]
    first_calls = model.calls

    mod.main(argv + ["--action-id", "A03"], model=model)  # T01은 건너뜀
    assert model.calls - first_calls == 5  # T02의 5 Frame만 새로 처리
    calls = model.calls
    mod.main(argv + ["--action-id", "A03", "--overwrite"], model=model)
    assert model.calls - calls == 11  # 6 + 5 다시 처리

    with pytest.raises(SystemExit):
        mod.main(argv + ["--clip-id", "T01_S09_A03_T01"], model=model)


def test_textbook_23_reads_team_sequences(mod, team, standing_pose, tmp_path):
    argv, out = team
    mod.main(argv, model=FakeModel(_result([standing_pose])))
    day23 = _load("23_summarize_action_sequences.py")
    summary_path = tmp_path / "summary.csv"
    old_argv = sys.argv
    sys.argv = ["23", "--sequence-dir", str(out), "--output", str(summary_path)]
    try:
        day23.main()
    finally:
        sys.argv = old_argv
    summary = pd.read_csv(summary_path)
    assert len(summary) == 3
    assert (summary["feature_ok_ratio"] == 1.0).all()