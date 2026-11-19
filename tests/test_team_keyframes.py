from __future__ import annotations

# ==================================================
# scripts/day04_extract_team_keyframes.py 테스트
# ==================================================
# 작은 가짜 MP4로 Frame 번호 계산 · clip_id 파일명 · 모아보기 · 검수표 보호를 확인합니다.
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
    path = ROOT / "scripts" / "day04_extract_team_keyframes.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_video(path: Path, frames: int, size=(48, 80)) -> None:
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    for i in range(frames):
        writer.write(np.full((height, width, 3), (i * 7) % 255, np.uint8))
    writer.release()


@pytest.fixture
def mod():
    return _load()


def test_keyframe_indices_match_day04_21(mod):
    # 교재 21번: bend_return 85 Frame → 0 / 42 / 84
    assert mod.keyframe_indices(85) == {"start": 0, "middle": 42, "end": 84}


def test_sheet_indices(mod):
    assert mod.sheet_indices(101) == [10, 30, 50, 70, 90]
    assert mod.sheet_indices(1) == [0, 0, 0, 0, 0]


def test_read_frames_sequential(mod, tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video, 20)
    frames, read_count = mod.read_frames(video, {0, 10, 19})
    assert set(frames) == {0, 10, 19}
    assert read_count == 20


@pytest.fixture
def team_dir(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    rows = []
    for subject, action, take, frames, status in (
        ("S01", "A03", "T01", 15, "PASS"),
        ("S01", "A03", "T02", 12, "REVIEW"),
        ("S01", "A05", "T01", 18, "PASS"),
    ):
        clip_id = f"T01_{subject}_{action}_{take}"
        _write_video(video_dir / f"{clip_id}.mp4", frames)
        rows.append(
            {
                "clip_id": clip_id,
                "subject_id": subject,
                "action_id": action,
                "action_label": "U파울" if action == "A03" else "푸싱",
                "take_id": take,
                "video_file": f"{clip_id}.mp4",
                "review_status": status,
                "review_reason": "다른 사람 등장" if status == "REVIEW" else "",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return tmp_path, manifest, video_dir


def _run(mod, monkeypatch, tmp_path, manifest, video_dir, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--manifest",
            str(manifest),
            "--video-dir",
            str(video_dir),
            "--output-dir",
            str(tmp_path / "keyframes"),
            "--review-csv",
            str(tmp_path / "tables" / "keyframe_review.csv"),
            *extra,
        ],
    )
    return mod.main()


def test_main_outputs(mod, team_dir, monkeypatch):
    tmp_path, manifest, video_dir = team_dir
    assert _run(mod, monkeypatch, tmp_path, manifest, video_dir) == 0

    out = tmp_path / "keyframes"
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert len(names) == 9  # 3 clip × 3장, clip_id가 들어가 덮어쓰지 않음
    assert "T01_S01_A03_T02_middle.jpg" in names

    sheets = sorted(p.name for p in (out / "contact_sheets").glob("*.jpg"))
    assert sheets == ["contact_S01_A03.jpg", "contact_S01_A05.jpg"]
    sheet = cv2.imread(str(out / "contact_sheets" / "contact_S01_A03.jpg"))
    assert sheet.shape[1] == mod.LABEL_WIDTH + 5 * mod.THUMB_WIDTH

    review = pd.read_csv(tmp_path / "tables" / "keyframe_review.csv")
    assert list(review.columns) == mod.REVIEW_COLUMNS
    assert review.loc[0, "middle_frame"] == 7  # 15 // 2
    assert review["review_result"].isna().all()  # 사람이 채울 칸은 비어 있음


def test_review_csv_is_protected(mod, team_dir, monkeypatch):
    tmp_path, manifest, video_dir = team_dir
    _run(mod, monkeypatch, tmp_path, manifest, video_dir)
    path = tmp_path / "tables" / "keyframe_review.csv"
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    review.loc[0, "review_result"] = "PASS"
    review.to_csv(path, index=False)

    _run(mod, monkeypatch, tmp_path, manifest, video_dir)  # 다시 실행
    again = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert again.loc[0, "review_result"] == "PASS"  # 채운 내용 유지

    _run(mod, monkeypatch, tmp_path, manifest, video_dir, "--overwrite-review")
    reset = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert reset.loc[0, "review_result"] == ""


def test_missing_video_is_fail(mod, team_dir, monkeypatch):
    tmp_path, manifest, video_dir = team_dir
    (video_dir / "T01_S01_A05_T01.mp4").unlink()
    assert _run(mod, monkeypatch, tmp_path, manifest, video_dir) == 1