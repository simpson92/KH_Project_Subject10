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


def test_fill_manual_only_blank_cells_with_operator_map():
    log = pd.DataFrame({"person_id": ["S01", "S02"], "cam_height_m": ["", "1.5"], "cam_distance_m": ["", ""],
                        "operator": ["", ""], "note": ["", "기존"]})
    args = type("A", (), {"cam_height": "1.3", "cam_distance": "2.0", "operator": "",
                          "operator_map": "S01=S05,S02=S04", "value_note": "실측", "force": False})()
    out = wlog.fill_manual(log, args)
    assert out["cam_height_m"].tolist() == ["1.3", "1.5"]          # 이미 적힌 1.5는 그대로
    assert out["operator"].tolist() == ["S05", "S04"]
    assert out["note"].tolist() == ["높이·거리: 실측", "기존 · 높이·거리: 실측"]
