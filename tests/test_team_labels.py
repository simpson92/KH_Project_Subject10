from __future__ import annotations

# ==================================================
# src/team_label_utils.py 테스트 — 구간 라벨 자동 초안 · V-01~V-10 검증
# ==================================================
# 정규화 좌표(골반 0, 어깨 폭 1, y 아래로 +)로 만든 가짜 신호 Take:
#   차렷 30 → 손 들기 15 → C01 유지 25 → 전환 12 → 신호(양손 허리) 40 → 복귀 12 → 차렷 30
import numpy as np
import pandas as pd
import pytest

from src.team_label_utils import (
    COLUMNS,
    _merge_short,
    keep_human_rows,
    label_rows,
    segment_signal,
    validate_labels,
)

PHASES = [("rest", 30), ("raise", 15), ("c01", 25), ("trans", 12), ("sig", 40), ("ret", 12), ("rest", 30)]
REST = {"right_wrist": (-0.6, 0.0), "left_wrist": (0.6, 0.0)}
C01 = {"right_wrist": (-0.4, -3.8), "left_wrist": (0.6, 0.0)}
SIG = {"right_wrist": (-0.7, -0.6), "left_wrist": (0.7, -0.6)}   # A01 양손 허리


def _fake_take(seed=0):
    rng = np.random.default_rng(seed)
    pose = []
    for name, length in PHASES:
        for i in range(length):
            f = (i + 1) / length
            if name == "rest":
                p = REST
            elif name == "c01":
                p = C01
            elif name == "sig":
                p = SIG
            else:
                a, b = {"raise": (REST, C01), "trans": (C01, SIG), "ret": (SIG, REST)}[name]
                p = {k: tuple(np.add(np.multiply(a[k], 1 - f), np.multiply(b[k], f))) for k in a}
            pose.append(p)
    rows = []
    for t, p in enumerate(pose):
        row = {"frame_id": t}
        for k, (x, y) in p.items():
            row[f"{k}_x"], row[f"{k}_y"] = x + rng.normal(0, 0.01), y + rng.normal(0, 0.01)
        row["right_ankle_x"], row["right_ankle_y"] = -0.3, 2.0
        row["left_ankle_x"], row["left_ankle_y"] = 0.3, 2.0
        for h in ["nose_y", "left_eye_y", "right_eye_y", "left_ear_y", "right_ear_y"]:
            row[h] = -2.3
        rows.append(row)
    return pd.DataFrame(rows)


def test_segment_finds_seven_phases():
    df = _fake_take()
    seg = segment_signal(df, "A01")
    c_s, c_e = seg["c01"]
    s_s, s_e = seg["signal"]
    assert 25 <= seg["rest_end"] <= 32              # 차렷 끝 ≈ 29
    assert 38 <= c_s <= 50 and 66 <= c_e <= 74        # C01 ≈ 45~69 (유지 구간 ± 손 올라가는 끝부분)
    assert 78 <= s_s <= 86 and 118 <= s_e <= 130      # 신호 ≈ 82~121 (정지형은 자세가 0.4 안이면 신호)
    assert 125 <= seg["rest_start"] <= 136            # 끝 차렷 ≈ 134


def test_rows_cover_clip_without_gaps_and_pass_validation():
    df = _fake_take()
    rows = pd.DataFrame(label_rows("T01_S09_A01_T01", "S09", "A01", len(df), segment_signal(df, "A01")))
    labels = rows["label"].tolist()
    assert labels[:5] == ["N00", "X", "C01", "X", "A01"] and labels[-1] == "N00"
    assert rows["start_frame"].iloc[0] == 0 and rows["end_frame"].iloc[-1] == len(df) - 1
    errors, warnings = validate_labels(rows, {"T01_S09_A01_T01": len(df)})
    assert errors == [] and warnings == []
    assert rows["qa_status"].eq("PASS").all() and rows["labeler"].eq("auto").all()


def test_daily_clip_is_whole_n00():
    rows = label_rows("T01_S09_D02_T01", "S09", "D02", 150, None, action_label="팔짱")
    assert len(rows) == 1 and rows[0]["label"] == "N00" and rows[0]["end_frame"] == 149
    assert "D02" in rows[0]["note"]


def test_missing_c01_becomes_review_x():
    df = _fake_take()
    df["right_wrist_y"] = 0.0                         # 손을 안 듦
    rows = label_rows("T01_S09_A01_T01", "S09", "A01", len(df), segment_signal(df, "A01"))
    assert rows[0]["label"] == "X" and rows[0]["qa_status"] == "REVIEW"


def test_merge_short_gives_short_x_to_n00():
    segs = _merge_short([["N00", 0, 3], ["X", 4, 20], ["A05", 21, 60], ["X", 61, 63], ["N00", 64, 99]])
    assert segs == [["X", 0, 20], ["A05", 21, 60], ["N00", 61, 99]]


def test_validate_catches_gap_overlap_and_bad_label():
    base = dict(source="webcam", person_id="S09", fps=30, labeler="auto", qa_status="PASS", note="")
    rows = pd.DataFrame([
        {**base, "clip_id": "T01_S09_A02_T01", "label": "N00", "start_frame": 0, "end_frame": 29},
        {**base, "clip_id": "T01_S09_A02_T01", "label": "A99", "start_frame": 35, "end_frame": 80},
        {**base, "clip_id": "T01_S09_A02_T01", "label": "N00", "start_frame": 70, "end_frame": 99},
    ])
    rows["start_sec"] = (rows["start_frame"] / 30).round(3)
    rows["end_sec"] = (rows["end_frame"] / 30).round(3)
    errors, _ = validate_labels(rows[COLUMNS])
    text = " ".join(errors)
    assert "V-02" in text and "V-03" in text and "V-04" in text


def test_for_training_rejects_review():
    rows = pd.DataFrame(label_rows("T01_S09_A01_T01", "S09", "A01", 100, None))
    errors, _ = validate_labels(rows, for_training=True)
    assert any("V-10" in e for e in errors)


def test_rerun_keeps_human_fixed_clip():
    df = _fake_take()
    auto = pd.DataFrame(label_rows("T01_S09_A01_T01", "S09", "A01", len(df), segment_signal(df, "A01")))
    human = auto.copy()
    human.loc[4, "end_frame"] = 125
    human["labeler"] = "S05"
    other = pd.DataFrame(label_rows("T01_S09_D01_T01", "S09", "D01", 90, None))
    merged = keep_human_rows(pd.concat([auto, other]), human)
    assert merged.loc[merged["clip_id"] == "T01_S09_A01_T01", "labeler"].eq("S05").all()
    assert (merged["clip_id"] == "T01_S09_D01_T01").sum() == 1
