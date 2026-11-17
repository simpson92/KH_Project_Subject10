from __future__ import annotations

# ==================================================
# scripts/day04_hand_detect_check.py 테스트
# ==================================================
# MediaPipe 대신 "손이 어디 있는지 아는 가짜 detector"를 넣어 확인합니다.
# - crop이 손목을 가운데에 두는지 (화면 밖은 검은색으로 채움)
# - 손이 2개일 때 손목에 가까운 손을 고르는지, 먼 손은 "다른 손"으로 버리는지
# - 손목 conf가 낮은 Frame은 검출률 분모에서 빠지는지 (빈 conf는 1.0으로 바꾸지 않음)
# - 화면을 줄이면 crop이 작아져 검출률이 떨어지는 흐름이 표에 그대로 나오는지
# - Sequence CSV가 없을 때 STEP 4 frame_row로 손목을 구하는지 (가짜 YOLO)
import importlib.util
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
    return _load("day04_hand_detect_check.py")


class FakeDetector:
    """이미지 안의 '흰 사각형' = 손. 사각형이 min_px보다 작게 찍히면 못 찾음.

    손 0번 점(WRIST) = 사각형 아래쪽 가운데로 돌려줌.
    """

    def __init__(self, min_px=12):
        self.min_px = min_px
        self.calls = []

    def detect(self, bgr):
        self.calls.append(bgr.shape[:2])
        gray = bgr[..., 0] if bgr.ndim == 3 else bgr
        count, _labels, stats, _ = cv2.connectedComponentsWithStats((gray > 200).astype(np.uint8))
        hands = []
        for i in range(1, count):
            x, y, w, h, _area = stats[i]
            if min(w, h) < self.min_px:
                continue
            points = np.zeros((21, 2), np.float32)
            points[:] = [x + w / 2, y + h / 2]
            points[0] = [x + w / 2, y + h]  # WRIST
            hands.append(types.SimpleNamespace(points=points, score=0.9))
        return hands

    def close(self):
        pass


def _args(mod, **kwargs):
    argv = []
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, list):
            argv += [flag] + [str(v) for v in value]
        else:
            argv += [flag, str(value)]
    return mod.parse_args(argv)


def _row(left=(140, 200), right=(60, 200), conf=0.9, shoulder=40.0):
    row = {
        "raw_left_wrist_x": left[0],
        "raw_left_wrist_y": left[1],
        "raw_right_wrist_x": right[0],
        "raw_right_wrist_y": right[1],
        "conf_left_wrist": conf,
        "conf_right_wrist": conf,
        "shoulder_width": shoulder,
    }
    return {k: ("" if v is None else v) for k, v in row.items()}


def _frame_with_hands(wrists, size=24, shape=(320, 200)):
    """손목 바로 위에 size×size 흰 사각형(손)을 그린 화면."""
    frame = np.zeros((*shape, 3), np.uint8)
    for x, y in wrists:
        frame[y - size : y, x - size // 2 : x + size // 2] = 255
    return frame


# ---------- crop ----------
def test_crop_keeps_wrist_in_center_and_pads(mod):
    image = np.full((100, 100, 3), 50, np.uint8)
    crop, origin = mod.crop_square(image, np.array([5.0, 95.0]), 40)
    assert crop.shape[:2] == (40, 40)
    assert tuple(origin) == (-15.0, 75.0)
    # 화면 밖(왼쪽 15 px, 아래 15 px)은 검은색
    assert crop[:, :15].max() == 0 and crop[-15:, :].max() == 0
    assert crop[10, 30, 0] == 50


# ---------- 같은 손 찾기 ----------
def test_nearest_hand_picks_close_and_rejects_far(mod):
    near = mod.Hand(points=np.full((21, 2), 10, np.float32), score=0.8)
    far = mod.Hand(points=np.full((21, 2), 100, np.float32), score=0.9)
    hand, dist = mod.nearest_hand([far, near], np.array([12.0, 10.0]), radius=5)
    assert hand is near and dist == pytest.approx(2.0)
    hand, dist = mod.nearest_hand([far], np.array([12.0, 10.0]), radius=5)
    assert hand is None and dist > 5
    assert mod.nearest_hand([], np.array([0.0, 0.0]), 5)[0] is None


# ---------- 손목 상태 ----------
def test_wrist_status_low_missing_and_blank_conf(mod):
    _, _, status = mod.wrist_of(_row(conf=0.3), "left", 0.5)
    assert status == "low"
    row = _row()
    row["raw_right_wrist_x"] = ""
    assert mod.wrist_of(row, "right", 0.5)[2] == "missing"
    row = _row(conf="")
    _wrist, conf, status = mod.wrist_of(row, "left", 0.5)
    assert status == "ok" and np.isnan(conf)  # 빈 conf를 1.0으로 바꾸지 않음


def test_shoulder_falls_back_to_raw_points(mod):
    row = {"shoulder_width": "", "raw_left_shoulder_x": 130, "raw_left_shoulder_y": 100,
           "raw_right_shoulder_x": 70, "raw_right_shoulder_y": 100}
    assert mod.shoulder_px(row) == pytest.approx(60.0)
    assert np.isnan(mod.shoulder_px({}))


# ---------- Frame 한 장 ----------
def test_frame_records_crop_and_full(mod):
    args = _args(mod, scales=[1.0], crop_scale=1.5, match_radius=0.5)
    frame = _frame_with_hands([(140, 200), (60, 200)])
    detector = FakeDetector(min_px=12)
    records = mod.frame_records(frame, _row(), {"clip_id": "c", "frame_id": 0}, detector, args)
    assert len(records) == 4  # (crop, full) × (left, right)
    assert all(r["detected"] == 1 for r in records)
    crop = [r for r in records if r["mode"] == "crop"]
    assert all(r["crop_px"] == 60 for r in crop)  # 어깨 40 px × 1.5
    # full 방식은 화면 전체를 한 번만 검출 (좌/우 공유)
    assert detector.calls.count((320, 200)) == 1


def test_low_conf_wrist_is_not_counted(mod):
    args = _args(mod, scales=[1.0], modes=["crop"])
    detector = FakeDetector()
    records = mod.frame_records(
        _frame_with_hands([(140, 200), (60, 200)]), _row(conf=0.2), {"frame_id": 0}, detector, args
    )
    assert {r["wrist_status"] for r in records} == {"low"}
    assert all(r["detected"] == "" for r in records)
    assert detector.calls == []  # 믿을 수 없는 손목이면 검출기를 부르지 않음


def test_other_hand_in_crop_is_not_matched(mod):
    """U파울처럼 두 손이 붙어 있어도 '내 손목에 가까운 손'만 검출로 셈."""
    args = _args(mod, scales=[1.0], modes=["crop"], crop_scale=3.0)
    # 왼손만 그림. 오른손목은 왼손에서 40 px(= 어깨 폭 1.0) 떨어져 있음
    frame = _frame_with_hands([(120, 200)])
    row = _row(left=(120, 200), right=(80, 200))
    records = mod.frame_records(frame, row, {"frame_id": 0}, FakeDetector(), args)
    detected = {r["side"]: r["detected"] for r in records}
    assert detected == {"left": 1, "right": 0}
    right = next(r for r in records if r["side"] == "right")
    assert right["hands_found"] == 1 and right["match_dist"] == pytest.approx(1.0)


def test_same_hand_goes_to_closer_wrist(mod):
    """두 손목이 붙어 있고 손은 하나만 보이면 → 가까운 손목만 검출, 다른 쪽은 shared."""
    args = _args(mod, scales=[1.0], modes=["crop"])
    frame = _frame_with_hands([(100, 200)])
    row = _row(left=(100, 200), right=(110, 200))
    records = mod.frame_records(frame, row, {"frame_id": 0}, FakeDetector(), args)
    by_side = {r["side"]: r for r in records}
    assert by_side["left"]["detected"] == 1 and by_side["left"]["shared"] == 0
    assert by_side["right"]["detected"] == 0 and by_side["right"]["shared"] == 1


def test_downscale_lowers_detection(mod):
    """손 24 px → 배율 0.25면 6 px라 가짜 detector(최소 12 px)가 못 찾음."""
    args = _args(mod, scales=[1.0, 0.25], modes=["crop"])
    frame = _frame_with_hands([(140, 200), (60, 200)])
    records = mod.frame_records(frame, _row(), {"frame_id": 0}, FakeDetector(min_px=12), args)
    by_scale = {}
    for r in records:
        by_scale.setdefault(r["scale"], []).append(r["detected"])
    assert by_scale[1.0] == [1, 1] and by_scale[0.25] == [0, 0]
    assert {r["shoulder_px"] for r in records if r["scale"] == 0.25} == {10.0}


# ---------- 요약 ----------
def test_summarize_rates_use_ok_wrists_only(mod):
    base = {"clip_id": "c", "subject_id": "S01", "action_id": "A03", "action_label": "U파울",
            "mode": "crop", "scale": 1.0, "hand_score": "", "shoulder_px": 40.0}
    rows = [
        dict(base, frame_id=0, side="left", wrist_status="ok", detected=1),
        dict(base, frame_id=0, side="right", wrist_status="ok", detected=1),
        dict(base, frame_id=5, side="left", wrist_status="ok", detected=1),
        dict(base, frame_id=5, side="right", wrist_status="ok", detected=0),
        dict(base, frame_id=10, side="left", wrist_status="low", detected=""),
        dict(base, frame_id=10, side="right", wrist_status="ok", detected=1),
    ]
    summary = mod.summarize(pd.DataFrame(rows)).iloc[0]
    assert summary["sampled_frames"] == 3
    assert summary["left_wrist_ok"] == 2 and summary["left_detect_rate"] == 1.0
    assert summary["right_detect_rate"] == pytest.approx(2 / 3, abs=1e-4)
    # 양손: 둘 다 ok인 Frame 0·5 중 둘 다 찾은 건 Frame 0 하나
    assert summary["both_wrist_ok"] == 2 and summary["both_detect_rate"] == 0.5


# ---------- 전체 실행 (가짜 영상) ----------
def _write_video(path, frames, wrists):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (200, 320))
    for _ in range(frames):
        writer.write(_frame_with_hands(wrists, size=40))
    writer.release()


@pytest.fixture
def team(tmp_path):
    video_dir, seq_dir, out = tmp_path / "videos", tmp_path / "sequences", tmp_path / "tables"
    video_dir.mkdir()
    seq_dir.mkdir()
    wrists = [(140, 200), (60, 200)]
    manifest = []
    for clip_id, action, label in (("T01_S01_A03_T01", "A03", "U파울"), ("T01_S01_A05_T01", "A05", "푸싱")):
        _write_video(video_dir / f"{clip_id}.mp4", 10, wrists)
        manifest.append({"clip_id": clip_id, "subject_id": "S01", "action_id": action,
                         "action_label": label, "video_file": f"{clip_id}.mp4"})
    # 첫 clip만 Sequence CSV가 있음 → 두 번째는 YOLO(가짜)로 손목을 구해야 함
    seq = pd.DataFrame([dict(frame_id=i, **_row(shoulder=40.0)) for i in range(10)])
    seq.to_csv(seq_dir / "T01_S01_A03_T01_sequence.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(tmp_path / "manifest.csv", index=False)
    argv = ["--manifest", str(tmp_path / "manifest.csv"), "--video-dir", str(video_dir),
            "--sequence-dir", str(seq_dir), "--output-dir", str(out),
            "--stride", "5", "--scales", "1.0", "0.5"]
    return argv, out


class FakeYolo:
    """가짜 YOLO: 손목 두 개와 어깨가 정해진 위치에 있는 사람 1명."""

    def predict(self, **kwargs):
        xy = np.zeros((1, 17, 2), np.float32)
        xy[0, :, :] = [100, 150]
        xy[0, 5], xy[0, 6] = [120, 150], [80, 150]  # 어깨 폭 40
        xy[0, 11], xy[0, 12] = [115, 250], [85, 250]
        xy[0, 9], xy[0, 10] = [140, 200], [60, 200]  # 손목
        keypoints = types.SimpleNamespace(xy=xy, conf=np.full((1, 17), 0.9, np.float32))
        boxes = types.SimpleNamespace(xyxy=np.array([[50, 100, 150, 300]], np.float32),
                                      conf=np.array([0.9], np.float32))
        return [types.SimpleNamespace(keypoints=keypoints, boxes=boxes)]


def test_main_end_to_end(mod, team):
    argv, out = team
    assert mod.main(argv, detector=FakeDetector(min_px=12), yolo_model=FakeYolo()) == 0
    frames = pd.read_csv(out / "hand_detect_frames.csv", encoding="utf-8-sig")
    summary = pd.read_csv(out / "hand_detect_qa.csv", encoding="utf-8-sig")
    assert set(frames["wrist_source"]) == {"sequence", "yolo"}
    # 10 Frame / stride 5 = 2장 × 배율 2 × 방식 2 × 손 2 = 16 Row × clip 2개
    assert len(frames) == 32
    assert len(summary) == 2 * 2 * 2  # clip × 방식 × 배율
    assert (summary["both_detect_rate"] == 1.0).all()


def test_no_yolo_skips_clip_without_sequence(mod, team):
    argv, out = team
    mod.main(argv + ["--no-yolo"], detector=FakeDetector())
    frames = pd.read_csv(out / "hand_detect_frames.csv", encoding="utf-8-sig")
    assert set(frames["clip_id"]) == {"T01_S01_A03_T01"}
