from __future__ import annotations

# ==================================================
# scripts/day06_team_build_sequences.py 테스트 — 1조 Split → Sequence Dataset (30 Step × 38-D)
# ==================================================
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.pose_utils import COCO_KEYPOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "seq6", ROOT / "scripts" / "day06_team_build_sequences.py"
)
seq6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seq6)

LABELS = ["A01", "N00"]
SPLIT_CONFIG = {
    "train_subjects": ["S01"],
    "val_subjects": ["S05"],
    "test_subjects": ["S02"],
}


def _keypoints(n_frames: int, n_bad: int = 0) -> pd.DataFrame:
    rows = []
    for i in range(n_frames):
        ok = i >= n_bad
        row = {"frame_id": i, "pose_detected": 1 if ok else 0}
        for j, name in enumerate(COCO_KEYPOINT_NAMES):
            row[f"{name}_x_px"] = 400.0 + 10 * j + i if ok else np.nan
            row[f"{name}_y_px"] = 300.0 + 20 * j if ok else np.nan
        # 어깨 폭이 0이 되지 않게
        row["left_shoulder_x_px"] = 500.0 + i if ok else np.nan
        row["right_shoulder_x_px"] = 380.0 + i if ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _write_split(tmp: Path, name: str, subject: str, labels=LABELS, frames=40):
    rows = []
    for k, label in enumerate(labels):
        clip_id = f"T01_{subject}_{label}_T0{k + 1}"
        rel = f"kp/{subject}/{clip_id}.csv"
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        _keypoints(frames).to_csv(tmp / rel, index=False)
        rows.append(
            {
                "clip_id": clip_id,
                "subject": subject,
                "action_id": label,
                "train_label": label,
                "take": f"T0{k + 1}",
                "keypoint_csv": rel,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(tmp / "splits" / f"{name}.csv", index=False)
    return df


def _setup(tmp: Path):
    (tmp / "splits").mkdir()
    splits = {
        "train": _write_split(tmp, "train", "S01"),
        "val": _write_split(tmp, "val", "S05"),
        "test": _write_split(tmp, "test", "S02"),
    }
    return splits


def test_check_inputs_ok(tmp_path):
    splits = _setup(tmp_path)
    assert (
        seq6.check_inputs(splits, SPLIT_CONFIG, LABELS, "train_label", tmp_path) == []
    )


def test_check_inputs_finds_problems(tmp_path):
    splits = _setup(tmp_path)
    splits["val"] = splits["val"].assign(subject="S01")  # 사람 겹침 (설정과 다름)
    splits["test"] = splits["test"][splits["test"]["train_label"] == "A01"]  # N00 없음
    splits["train"].loc[0, "keypoint_csv"] = "kp/none.csv"  # 파일 없음
    errors = " | ".join(
        seq6.check_inputs(splits, SPLIT_CONFIG, LABELS, "train_label", tmp_path)
    )
    assert "val: subject" in errors
    assert "test: 없는 라벨 ['N00']" in errors
    assert "Keypoint CSV 없음" in errors


def test_build_clip_row_shape_and_label():
    item = {
        "clip_id": "c1",
        "subject": "S01",
        "action_id": "D02",
        "train_label": "N00",
        "take": "T01",
    }
    row, valid = seq6.build_clip_row(
        item,
        _keypoints(40, n_bad=5),
        steps=30,
        min_valid=10,
        label_column="train_label",
    )
    assert valid == 35
    assert row["train_label"] == "N00" and row["action_id"] == "D02"
    assert row["first_valid_frame"] == 5 and row["last_valid_frame"] == 39
    features = [
        k for k in row if k.startswith("t") and k != "take" and k != "train_label"
    ]
    assert len(features) == 30 * 38
    assert np.all(np.isfinite([row[k] for k in features]))


def test_build_clip_row_skips_short_clip():
    item = {"clip_id": "c1", "subject": "S01", "train_label": "A01"}
    row, valid = seq6.build_clip_row(
        item,
        _keypoints(12, n_bad=5),
        steps=30,
        min_valid=10,
        label_column="train_label",
    )
    assert row is None and valid == 7


def test_main_end_to_end(tmp_path):
    _setup(tmp_path)
    out = tmp_path / "out"
    summary = tmp_path / "summary.csv"
    config = tmp_path / "cfg.json"
    config.write_text(
        '{"sequence_steps": 30, "min_valid_frames": 10, "label_column": "train_label", "labels": ["A01", "N00"]}'
    )
    split_cfg = tmp_path / "split.json"
    split_cfg.write_text(
        '{"train_subjects": ["S01"], "val_subjects": ["S05"], "test_subjects": ["S02"]}'
    )
    code = seq6.main(
        [
            "--config",
            str(config),
            "--split-config",
            str(split_cfg),
            "--split-dir",
            str(tmp_path / "splits"),
            "--output-dir",
            str(out),
            "--summary",
            str(summary),
            "--root",
            str(tmp_path),
        ]
    )
    assert code == 0
    train = pd.read_csv(out / "train.csv")
    assert train.shape == (2, 8 + 1140)
    assert list(train.columns[:8]) == seq6.META_COLUMNS
    s = pd.read_csv(summary)
    assert list(s["clips"]) == [2, 2, 2] and list(s["N00"]) == [1, 1, 1]
