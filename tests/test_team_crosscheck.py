from __future__ import annotations

# ==================================================
# src/team_crosscheck_utils.py 테스트 — 교차 검수 표본 · 배정 · 비교 · 반영
# ==================================================

import pandas as pd

from src.team_crosscheck_utils import (
    apply_fixes,
    assign_reviewers,
    auto_boundaries,
    blank_review_sheet,
    compare,
    pick_sample,
    validate_review_sheet,
)
from src.team_label_utils import label_rows, validate_labels

SUBJ = ["S01", "S02", "S03", "S04", "S05"]
ACTS = [f"A{i:02d}" for i in range(1, 16)] + ["D01", "D02", "D03"]


def _labels():
    seg = {"rest_end": 19, "rest_start": 120, "c01": (30, 50), "signal": (60, 100)}
    rows = []
    for s in SUBJ:
        for a in ACTS:
            for t in range(1, 6):
                clip = f"T01_{s}_{a}_T{t:02d}"
                rows += label_rows(clip, s, a, 140, None if a.startswith("D") else seg)
    return pd.DataFrame(rows)


def test_sample_is_10_percent_and_reviewers_never_check_own_video():
    labels = _labels()
    sample = pick_sample(labels, ["T01_S03_A07_T02"], n_total=45)
    assert len(sample) == 45 and "T01_S03_A07_T02" in set(sample["clip_id"])
    assert sample["action_id"].isin([f"A{i:02d}" for i in range(1, 16)]).sum() >= 30
    assigned = assign_reviewers(sample)
    assert (assigned["reviewer"] != assigned["subject_id"]).all()
    assert assigned.groupby("reviewer").size().tolist() == [9] * 5


def test_blank_sheet_hides_auto_boundaries():
    sample = assign_reviewers(pick_sample(_labels(), [], n_total=45))
    sheet = blank_review_sheet(sample, {"A01": "블로킹"})
    assert not any(c in sheet.columns for c in ["c01_start", "sig_start"])
    assert (sheet["review_c01_start"] == "").all()


def test_compare_flags_boundary_diff_over_0_2s():
    labels = _labels()
    review = pd.DataFrame([
        {"reviewer": "S02", "clip_id": "T01_S01_A01_T01", "action_id": "A01", "frames": 140,
         "review_c01_start": 31, "review_c01_end": 50, "review_sig_start": 61, "review_sig_end": 101, "verdict": "OK"},
        {"reviewer": "S03", "clip_id": "T01_S01_A02_T01", "action_id": "A02", "frames": 140,
         "review_c01_start": 40, "review_c01_end": 60, "review_sig_start": 70, "review_sig_end": 110, "verdict": "FIX"},
    ])
    res = compare(review, labels).set_index("clip_id")
    assert res.loc["T01_S01_A01_T01", "mean_diff_sec"] < 0.2 and not res.loc["T01_S01_A01_T01", "over_0_2s"]
    assert res.loc["T01_S01_A02_T01", "over_0_2s"]


def test_apply_fix_rebuilds_segments_and_passes_validation():
    labels = _labels()
    review = pd.DataFrame([{"reviewer": "S03", "clip_id": "T01_S01_A02_T01", "action_id": "A02", "frames": 140,
                            "review_c01_start": 35, "review_c01_end": 55, "review_sig_start": 66,
                            "review_sig_end": 110, "verdict": "FIX"}])
    assert validate_review_sheet(review) == []
    fixed, changed = apply_fixes(labels, review)
    b = auto_boundaries(fixed).set_index("clip_id").loc["T01_S01_A02_T01"]
    assert (b["c01_start"], b["c01_end"], b["sig_start"], b["sig_end"]) == (35, 55, 66, 110)
    g = fixed[fixed["clip_id"] == "T01_S01_A02_T01"]
    assert g["labeler"].eq("S03").all() and changed == ["T01_S01_A02_T01: FIX by S03"]
    errors, _ = validate_labels(fixed)
    assert errors == []


def test_review_sheet_errors():
    bad = pd.DataFrame([
        {"clip_id": "a", "action_id": "A01", "frames": 140, "verdict": "", "review_c01_start": ""},
        {"clip_id": "b", "action_id": "A01", "frames": 140, "verdict": "FIX", "review_c01_start": 50,
         "review_c01_end": 40, "review_sig_start": 60, "review_sig_end": 90},
    ])
    errors = " ".join(validate_review_sheet(bad))
    assert "a: verdict" in errors and "b: Frame 순서" in errors
