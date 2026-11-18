from __future__ import annotations

# scripts/day05_team_make_webcam_log.py 테스트 — manifest · Human QA → 03 문서 9.1 형식
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wlog", ROOT / "scripts" / "day05_team_make_webcam_log.py")
wlog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wlog)


def test_build_log_fills_known_columns_and_leaves_measurements_blank():
    manifest = pd.DataFrame([
        {"clip_id": "T01_S02_A01_T01", "subject_id": "S02", "action_id": "A01", "take_id": "T01",
         "video_file": "T01_S02_A01_T01.mp4", "capture_set": "main_1118", "review_reason": "",
         "fps": 30.0, "width": 1080, "height": 1920},
        {"clip_id": "T01_S05_D02_T03", "subject_id": "S05", "action_id": "D02", "take_id": "T03",
         "video_file": "T01_S05_D02_T03.mp4", "capture_set": "main_1117", "review_reason": "손끝 잘림",
         "fps": 30.0, "width": 1080, "height": 1920},
    ])
    review = pd.DataFrame({"clip_id": manifest["clip_id"], "human_decision": ["PASS", "EXCLUDE"],
                           "human_note": ["", "대체"]})
    log = wlog.build_log(manifest, review, {"S02"})
    assert list(log.columns) == wlog.COLUMNS
    assert log.loc[0, "date"] == "2026-11-18" and log.loc[0, "split_role"] == "test"
    assert log.loc[1, "clip_type"] == "daily" and log.loc[1, "take"] == 3
    assert log.loc[1, "result"] == "RESHOOT" and log.loc[1, "issue"] == "손끝 잘림"
    assert (log[wlog.MANUAL] == "").all().all()
