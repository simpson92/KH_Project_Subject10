from __future__ import annotations

# ==================================================
# scripts/day05_team_export_keypoints.py 테스트 — Sequence → 5일차 Keypoint CSV 형식
# ==================================================
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.pose_utils import COCO_KEYPOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export", ROOT / "scripts" / "day05_team_export_keypoints.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def _seq():
    rows = []
    for i, ok in enumerate((1, 0)):
        row = {"frame_id": i, "time_sec": i / 30, "image_width": 1080, "image_height": 1920,
               "people": 1 if ok else 0, "bbox_conf": 0.9 if ok else np.nan, "pose_ok": ok}
        for name in COCO_KEYPOINT_NAMES:
            row[f"raw_{name}_x"] = 540.0 if ok else np.nan
            row[f"raw_{name}_y"] = 960.0 if ok else np.nan
            row[f"conf_{name}"] = 0.95 if ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def test_columns_match_course_26_and_norm_is_image_ratio():
    item = {"clip_id": "T01_S09_D01_T01", "subject_id": "S09", "action_id": "D01",
            "action_label": "휘슬", "take_id": "T01", "train_label": "N00"}
    out = export.to_keypoint_frame(_seq(), item)
    head = ["clip_id", "frame_id", "time_sec", "subject", "action_id", "action_label", "take",
            "person_count", "selected_box_conf", "pose_detected"]
    assert list(out.columns[:10]) == head
    assert list(out.columns[10:15]) == ["nose_x_px", "nose_y_px", "nose_x_norm", "nose_y_norm", "nose_conf"]
    assert out.shape[1] == 10 + 17 * 5 + 1 and out["train_label"].eq("N00").all()
    assert np.isclose(out.loc[0, "nose_x_norm"], 0.5) and np.isclose(out.loc[0, "nose_y_norm"], 0.5)
    assert out.loc[1, "pose_detected"] == 0 and pd.isna(out.loc[1, "nose_x_px"])
