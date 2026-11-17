from __future__ import annotations

# ==================================================
# scripts/day04_check_team_videos.py 테스트
# ==================================================
# 실제 영상 대신 작은 가짜 MP4를 만들어 판정(OK/WARN/FAIL)을 확인합니다.
# 조원 얼굴이 담긴 영상은 쓰지 않습니다.
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "day04_check_team_videos.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_video(path: Path, frames: int, size=(64, 96), fps=30.0) -> None:
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for i in range(frames):
        writer.write(np.full((height, width, 3), i % 255, np.uint8))
    writer.release()


def _row(clip_id, frames, width=64, height=96, **override):
    _team, subject, action, take = clip_id.split("_")
    row = {
        "clip_id": clip_id,
        "subject_id": subject,
        "action_id": action,
        "action_label": "U파울",
        "take_id": take,
        "video_file": f"{clip_id}.mp4",
        "fps": "30.0",
        "frame_count": str(frames),
        "width": str(width),
        "height": str(height),
    }
    row.update(override)
    return row


@pytest.fixture
def mod():
    return _load()


def test_orientation(mod):
    assert mod.orientation_of(1080, 1920) == "portrait"
    assert mod.orientation_of(640, 480) == "landscape"
    assert mod.orientation_of(0, 0) == ""


def test_ok_video(mod, tmp_path):
    video = tmp_path / "T01_S01_A03_T01.mp4"
    _write_video(video, 12)
    info = mod.probe_video(video)
    assert info["open_ok"] and info["decoded_frames"] == 12
    status, issues = mod.judge(_row("T01_S01_A03_T01", 12), info, {"A03", "A05"})
    assert status == "OK", issues


def test_missing_file_is_fail(mod, tmp_path):
    info = mod.probe_video(tmp_path / "T01_S01_A03_T02.mp4")
    status, issues = mod.judge(_row("T01_S01_A03_T02", 10), info, {"A03"})
    assert status == "FAIL" and "파일 없음" in issues


def test_broken_file_is_fail(mod, tmp_path):
    video = tmp_path / "T01_S01_A03_T03.mp4"
    video.write_bytes(b"not a video")
    status, _ = mod.judge(_row("T01_S01_A03_T03", 10), mod.probe_video(video), {"A03"})
    assert status == "FAIL"


def test_label_mismatch_is_warn(mod, tmp_path):
    video = tmp_path / "T01_S01_A03_T04.mp4"
    _write_video(video, 8)
    row = _row("T01_S01_A03_T04", 8, action_id="A05")
    status, issues = mod.judge(row, mod.probe_video(video), {"A03", "A05"})
    assert status == "WARN"
    assert any("action_id 불일치" in issue for issue in issues)


def test_manifest_value_mismatch_is_warn(mod, tmp_path):
    video = tmp_path / "T01_S02_A05_T01.mp4"
    _write_video(video, 8)
    row = _row("T01_S02_A05_T01", 8, width="1080", height="1920")
    status, issues = mod.judge(row, mod.probe_video(video), {"A05"})
    assert status == "WARN"
    assert any("width" in issue for issue in issues)


def test_main_writes_inventory(mod, tmp_path, monkeypatch, capsys):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    rows = []
    for clip_id, frames in (("T01_S01_A03_T01", 6), ("T01_S01_A05_T01", 7)):
        _write_video(video_dir / f"{clip_id}.mp4", frames)
        rows.append(_row(clip_id, frames))
    rows.append(_row("T01_S02_A03_T01", 5))  # 영상 없음 → FAIL
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    classes = tmp_path / "classes.csv"
    pd.DataFrame({"action_id": ["A03", "A05"]}).to_csv(classes, index=False)
    output = tmp_path / "out" / "video_inventory.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--manifest",
            str(manifest),
            "--action-classes",
            str(classes),
            "--video-dir",
            str(video_dir),
            "--output",
            str(output),
        ],
    )
    assert mod.main() == 1  # FAIL이 하나 있으므로 1
    result = pd.read_csv(output)
    assert list(result["status"]) == ["OK", "OK", "FAIL"]
    assert "이름" not in result.columns
    assert "Team01 video check: FAIL" in capsys.readouterr().out