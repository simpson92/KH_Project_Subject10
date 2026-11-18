from __future__ import annotations

# ==================================================
# src/team_dataset_utils.py 테스트 (1조 ②단계: QA → Human QA → Approved → Split → Leakage)
# ==================================================
# 가짜 Sequence 표와 manifest로 확인합니다. 영상 · YOLO · data/ 폴더가 없어도 돌아갑니다.
import pandas as pd
import pytest

from src.pose_utils import COCO_KEYPOINT_NAMES
from src.team_dataset_utils import (
    ACTION_CRITICAL_KEYPOINTS,
    EXPECTED_ACTIONS,
    build_approved,
    build_qa_table,
    check_leakage,
    inspect_sequence,
    merge_previous_review,
    split_by_subject,
    train_label,
    validate_review,
)


def _sequence(n=10, conf=0.9, wrist=0.9, people=1, pose_ok=1):
    rows = []
    for i in range(n):
        row = {"frame_id": i, "pose_ok": pose_ok, "people": people}
        for name in COCO_KEYPOINT_NAMES:
            row[f"conf_{name}"] = conf
        row["conf_right_wrist"] = wrist
        rows.append(row)
    return pd.DataFrame(rows)


def _manifest(subjects=("S01", "S02", "S03"), actions=EXPECTED_ACTIONS, takes=1):
    rows = []
    for s in subjects:
        for a in actions:
            for t in range(1, takes + 1):
                clip = f"T01_{s}_{a}_T{t:02d}"
                rows.append({"clip_id": clip, "subject_id": s, "action_id": a, "action_label": a,
                             "take_id": f"T{t:02d}", "capture_set": "main", "video_file": f"{clip}.mp4",
                             "review_status": "PASS", "review_reason": ""})
    return pd.DataFrame(rows)


# ---------- [01] 동작 정의 ----------
def test_all_18_actions_have_critical_keypoints():
    assert len(EXPECTED_ACTIONS) == 18
    for names in ACTION_CRITICAL_KEYPOINTS.values():
        assert set(names) <= set(COCO_KEYPOINT_NAMES)


def test_daily_actions_map_to_n00():
    assert train_label("D02") == "N00"
    assert train_label("A15") == "A15"


# ---------- [02] 자동 QA ----------
def test_clean_clip_is_auto_pass():
    qa = inspect_sequence(_sequence(), "A01")
    assert qa["qa_status"] == "AUTO_PASS" and qa["qa_reason"] == ""


def test_low_wrist_conf_triggers_check():
    qa = inspect_sequence(_sequence(wrist=0.1), "A09")  # 6개 중 1개 관절이 항상 낮음 → 16.7%
    assert qa["low_critical_ratio"] == pytest.approx(1 / 6, abs=1e-3)
    qa = inspect_sequence(_sequence(wrist=0.1, conf=0.3), "A09")
    assert qa["qa_status"] == "CHECK" and "low_critical" in qa["qa_reason"]


def test_two_people_and_missing_pose_trigger_check():
    assert "single_person" in inspect_sequence(_sequence(people=2), "A02")["qa_reason"]
    assert "pose_ratio" in inspect_sequence(_sequence(pose_ok=0), "A02")["qa_reason"]


def test_unknown_action_raises():
    with pytest.raises(RuntimeError):
        inspect_sequence(_sequence(), "A99")


def test_qa_table_marks_missing_and_manifest_review(tmp_path):
    manifest = _manifest(subjects=("S01",), actions=["A01", "A02"])
    manifest.loc[1, "review_status"] = "REVIEW"
    for clip in manifest["clip_id"]:
        _sequence().to_csv(tmp_path / f"{clip}_sequence.csv", index=False)
    (tmp_path / "T01_S01_A01_T01_sequence.csv").unlink()
    qa = build_qa_table(manifest, tmp_path)
    assert qa.loc[0, "qa_status"] == "NO_SEQUENCE"
    assert qa.loc[1, "qa_status"] == "CHECK" and qa.loc[1, "qa_reason"] == "manifest_review"


# ---------- [03] Human QA ----------
def test_rerun_keeps_previous_human_decision():
    qa = pd.DataFrame({"clip_id": ["a", "b"], "qa_status": ["AUTO_PASS", "CHECK"]})
    prev = pd.DataFrame({"clip_id": ["a"], "human_decision": ["EXCLUDE"], "human_note": ["손 잘림"]})
    review = merge_previous_review(qa, prev)
    assert review["human_decision"].tolist() == ["EXCLUDE", "UNREVIEWED"]
    assert review.loc[0, "human_note"] == "손 잘림"


def test_validate_review_rules():
    df = pd.DataFrame({
        "clip_id": ["a", "b", "c", "d"],
        "qa_status": ["AUTO_PASS", "CHECK", "AUTO_PASS", "NO_SEQUENCE"],
        "human_decision": ["PASS", "PASS", "maybe", "PASS"],
        "human_note": ["", "", "", ""],
    })
    errors = " ".join(validate_review(df))
    assert "human_note 없이 PASS: b" in errors
    assert "PASS / RESHOOT / EXCLUDE 가 아닌 값: c" in errors
    assert "NO_SEQUENCE" in errors
    assert not any("NO_SEQUENCE" in e for e in validate_review(df, allow_no_sequence=True))


def test_approved_keeps_only_pass_with_sequence():
    df = pd.DataFrame({"clip_id": list("abcd"),
                       "qa_status": ["AUTO_PASS", "AUTO_PASS", "CHECK", "NO_SEQUENCE"],
                       "human_decision": ["PASS", "RESHOOT", "EXCLUDE", "PASS"]})
    assert build_approved(df)["clip_id"].tolist() == ["a"]


# ---------- [04] Split · Leakage ----------
CONFIG = {"train_subjects": ["S01"], "val_subjects": ["S02"], "test_subjects": ["S03"]}


def test_subject_split_and_no_leakage():
    approved = _manifest()
    approved["train_label"] = approved["action_id"].map(train_label)
    splits = split_by_subject(approved, CONFIG)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [18, 18, 18]
    assert check_leakage(splits) == []


def test_split_fails_on_overlap_missing_action_or_unassigned():
    approved = _manifest()
    with pytest.raises(RuntimeError, match="중복"):
        split_by_subject(approved, {**CONFIG, "val_subjects": ["S01"]})
    with pytest.raises(RuntimeError, match="없는 Action: A05"):
        split_by_subject(approved[~((approved.subject_id == "S02") & (approved.action_id == "A05"))], CONFIG)
    with pytest.raises(RuntimeError, match="배정되지 않은"):
        split_by_subject(_manifest(subjects=("S01", "S02", "S03", "S04")), CONFIG)


def test_leakage_detects_same_clip_in_two_splits():
    approved = _manifest()
    splits = split_by_subject(approved, CONFIG)
    splits["test"] = pd.concat([splits["test"], splits["train"].head(1)])
    errors = " ".join(check_leakage(splits))
    assert "Subject Leakage" in errors and "Clip Leakage" in errors
