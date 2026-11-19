from __future__ import annotations

# ==================================================
# scripts/day04_make_clip_manifest.py 테스트
# ==================================================
# 4일차 이름 규칙 T01_S##_A##_T## · manifest 칸 · 실명 미노출 · 중복 방지

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
def mods():
    pytest.importorskip("matplotlib")
    stubbed = []
    for name in ("cv2", "torch", "ultralytics"):
        if importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            module.YOLO = None
            sys.modules[name] = module
            stubbed.append(name)
    yield (
        _load("day04_make_clip_manifest.py"),
        _load("day03_batch_video_qa.py"),
        _load("referee_00_rename_videos.py"),
    )
    for name in stubbed:
        sys.modules.pop(name, None)


ARGS = argparse.Namespace(source_type="phone", capture_set="smoke")


def no_info(_path):
    return {}


def make_videos(tmp_path, names):
    raw = tmp_path / "raw"
    raw.mkdir()
    videos = []
    for name in names:
        path = raw / name
        path.write_bytes(b"x")
        videos.append(path)
    return sorted(videos)


def test_action_classes_match_labeling_guide(mods):
    manifest, _, _ = mods
    action_map = manifest.load_action_map()
    # 02_라벨링_가이드 4장: U파울 = A03, 푸싱 = A05
    assert action_map["ufoul"] == ("A03", "U파울")
    assert action_map["pushing"][0] == "A05"
    table = pd.read_csv(manifest.ACTION_CLASSES_CSV, dtype=str)
    assert table["action_id"].tolist()[:15] == [f"A{i:02d}" for i in range(1, 16)]
    assert {"C01", "N00"} <= set(table["action_id"])


def test_clip_id_format(mods):
    manifest, _, _ = mods
    assert manifest.make_clip_id("S03", "A05", 2) == "T01_S03_A05_T02"
    assert manifest.make_clip_id("S01", "A03", 10) == "T01_S01_A03_T10"


def test_manifest_from_original_names(mods, tmp_path):
    manifest, batch, rename = mods
    videos = make_videos(
        tmp_path,
        [
            "s03 u파울-가나다 (1).mp4",
            "s05 푸싱-라마바 (10).mp4",
            "s05 푸싱-가나다  (2).mp4",
        ],
    )
    rename.SUBJECT_CSV = tmp_path / "subjects.csv"
    assert manifest.ensure_subjects_csv(videos, rename, rename.SUBJECT_CSV)
    ids = batch.identify_videos(videos, rename)
    qa = {"S02_pushing_T10": ("REVIEW", "다른 사람 등장")}

    table, mapping, warnings = manifest.build_manifest(
        videos, ids, manifest.load_action_map(), qa, ARGS, read_info=no_info
    )
    assert table["clip_id"].tolist() == [
        "T01_S01_A03_T01",
        "T01_S01_A05_T02",
        "T01_S02_A05_T10",
    ]
    # 4일차 교재 기본 칸이 앞에 그대로
    assert list(table.columns[:8]) == [
        "clip_id",
        "subject_id",
        "action_id",
        "action_label",
        "take_id",
        "video_file",
        "source_type",
        "review_status",
    ]
    assert table.loc[2, "review_status"] == "REVIEW"
    assert table.loc[0, "review_status"] == "PENDING"
    assert table.loc[0, "video_file"] == "T01_S01_A03_T01.mp4"
    # Git에 올라가는 manifest에는 실명이 없어야 함
    text = table.to_csv(index=False)
    assert "가나다" not in text and "라마바" not in text
    # 대응표(Git 제외)에만 원본 경로
    assert len(mapping) == 3
    assert warnings == []


def test_subjects_csv_is_kept_once_created(mods, tmp_path):
    manifest, _, rename = mods
    subjects = tmp_path / "subjects.csv"
    subjects.write_text("subject_id,name\nS07,가나다\n", encoding="utf-8-sig")
    videos = make_videos(tmp_path, ["s03 u파울-가나다 (1).mp4"])
    assert not manifest.ensure_subjects_csv(videos, rename, subjects)  # 덮어쓰지 않음
    assert "S07" in subjects.read_text(encoding="utf-8-sig")


def test_duplicate_clip_id_raises(mods, tmp_path):
    manifest, batch, rename = mods
    # 이름 뒤 공백만 다른 같은 사람·같은 회차 → 같은 clip_id
    videos = make_videos(
        tmp_path, ["s03 u파울-가나다 (3).mp4", "s03 u파울-가나다  (3).mp4"]
    )
    rename.SUBJECT_CSV = tmp_path / "none.csv"
    ids = batch.identify_videos(videos, rename)
    with pytest.raises(RuntimeError, match="겹칩니다"):
        manifest.build_manifest(
            videos, ids, manifest.load_action_map(), {}, ARGS, read_info=no_info
        )


def test_unknown_file_is_skipped_with_warning(mods, tmp_path):
    manifest, batch, rename = mods
    videos = make_videos(tmp_path, ["random_clip.mp4", "s03 u파울-가나다 (1).mp4"])
    rename.SUBJECT_CSV = tmp_path / "none.csv"
    ids = batch.identify_videos(videos, rename)
    table, _, warnings = manifest.build_manifest(
        videos, ids, manifest.load_action_map(), {}, ARGS, read_info=no_info
    )
    assert table["clip_id"].tolist() == ["T01_S01_A03_T01"]
    assert any("random_clip" in w for w in warnings)


def test_copy_keeps_original_and_is_rerunnable(mods, tmp_path):
    manifest, _, _ = mods
    src = tmp_path / "s03 u파울-가나다 (1).mp4"
    src.write_bytes(b"video")
    mapping = pd.DataFrame([{"source_path": str(src), "clip_id": "T01_S01_A03_T01"}])
    dest = tmp_path / "videos"

    assert manifest.copy_videos(mapping, dest, link=False) == (1, 0)
    assert (dest / "T01_S01_A03_T01.mp4").read_bytes() == b"video"
    assert src.exists()  # 원본은 그대로
    assert manifest.copy_videos(mapping, dest, link=False) == (
        0,
        1,
    )  # 다시 실행해도 안전

    (dest / "T01_S01_A03_T01.mp4").write_bytes(b"other content")
    with pytest.raises(RuntimeError):
        manifest.copy_videos(mapping, dest, link=False)
