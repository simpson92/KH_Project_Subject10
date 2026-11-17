from __future__ import annotations

# ==================================================
# scripts/day04_team_sequence_qa.py 테스트
# ==================================================
# STEP 4 코드(frame_row)로 "정답을 아는 가짜 Sequence"를 만들고,
# 일부 Frame을 일부러 망가뜨려 QA 값과 REVIEW 제안이 맞게 나오는지 확인합니다.
import importlib.util
import types
from pathlib import Path

import pandas as pd
import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load(filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qa():
    pytest.importorskip("cv2")
    return _load("day04_team_sequence_qa.py")


@pytest.fixture(scope="module")
def seq_mod():
    pytest.importorskip("cv2")
    return _load("day04_build_team_sequences.py")


def _result(poses):
    xy = np.stack(poses).astype(np.float32)
    boxes = [
        [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in poses
    ]
    return types.SimpleNamespace(
        keypoints=types.SimpleNamespace(
            xy=xy, conf=np.full((len(poses), 17), 0.9, np.float32)
        ),
        boxes=types.SimpleNamespace(
            xyxy=np.asarray(boxes, np.float32),
            conf=np.full(len(poses), 0.8, np.float32),
        ),
    )


def make_sequence(seq_mod, pose, clip_id="T01_S01_A03_T01", frames=30):
    """손목이 조금씩 움직이는 정상 Sequence."""
    _, subject, action, take = clip_id.split("_")
    rows = []
    for i in range(frames):
        p = pose.copy()
        p[10, 1] -= 2 * i  # 오른손목이 위로
        base = {
            "clip_id": clip_id,
            "subject_id": subject,
            "action_id": action,
            "action_label": "U파울",
            "take_id": take,
            "frame_id": i,
            "time_sec": round(i / 30, 4),
        }
        rows.append(seq_mod.frame_row(_result([p]), base, 0.5, "area"))
    return pd.DataFrame(rows, columns=seq_mod.ALL_COLUMNS)


def _args(qa, *extra):
    return qa.parse_args(list(extra))


def test_adjacent_change_matches_textbook_23(qa, seq_mod, standing_pose):
    df = make_sequence(seq_mod, standing_pose)
    day23 = _load("23_summarize_action_sequences.py")
    expected = day23.mean_adjacent_feature_change(df, day23.coordinate_columns())
    assert qa.adjacent_changes(df).mean() == pytest.approx(expected)


def test_longest_run(qa):
    mask = pd.Series([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert qa.longest_run(mask) == 3
    assert qa.longest_run(pd.Series([], dtype=bool)) == 0


def test_clean_clip_is_pass(qa, seq_mod, standing_pose):
    row = qa.summarize_clip(make_sequence(seq_mod, standing_pose), _args(qa))
    row.update(manifest_status="PASS", keyframe_review_result="")
    assert row["essential_low_ratio"] == 0 and row["max_consecutive_bad"] == 0
    assert row["shoulder_collapse_frames"] == 0
    assert qa.suggest(row, _args(qa)) == ("PASS", [])


def test_low_wrist_run_is_review(qa, seq_mod, standing_pose):
    df = make_sequence(seq_mod, standing_pose)
    df.loc[10:15, "conf_left_wrist"] = 0.2  # 6 Frame 연속 왼손목 저신뢰
    row = qa.summarize_clip(df, _args(qa))
    row.update(manifest_status="PASS", keyframe_review_result="")
    assert row["essential_low_ratio"] == pytest.approx(6 / 30, abs=1e-4)
    assert row["worst_essential_joint"] == "left_wrist"
    assert row["max_consecutive_bad"] == 6
    status, reasons = qa.suggest(row, _args(qa))
    assert status == "REVIEW"
    assert any("left_wrist" in r for r in reasons)
    assert any("6개 연속" in r for r in reasons)
    # 기준을 바꾸면 결과도 바뀜
    loose = _args(qa, "--low-ratio", "0.5", "--max-bad-run", "10")
    assert qa.suggest(row, loose)[0] == "PASS"


def test_low_knee_is_not_essential(qa, seq_mod, standing_pose):
    df = make_sequence(seq_mod, standing_pose)
    df["conf_left_knee"] = 0.1  # 무릎은 필수 관절이 아님
    assert qa.summarize_clip(df, _args(qa))["essential_low_ratio"] == 0


def test_multi_person_and_collapse(qa, seq_mod, standing_pose):
    df = make_sequence(seq_mod, standing_pose)
    df.loc[0:4, "people"] = 2
    df.loc[20, "shoulder_width"] = 10.0  # 중앙값 100의 절반 미만
    row = qa.summarize_clip(df, _args(qa))
    row.update(manifest_status="PASS", keyframe_review_result="")
    assert row["multi_person_ratio"] == pytest.approx(5 / 30, abs=1e-4)
    assert row["shoulder_collapse_frames"] == 1
    reasons = qa.suggest(row, _args(qa))[1]
    assert any("여러 사람" in r for r in reasons)
    assert any("어깨 폭 붕괴" in r for r in reasons)


def test_manifest_and_human_review_flags(qa, seq_mod, standing_pose):
    row = qa.summarize_clip(make_sequence(seq_mod, standing_pose), _args(qa))
    row.update(manifest_status="REVIEW", keyframe_review_result="")
    assert qa.suggest(row, _args(qa))[0] == "REVIEW"
    row.update(manifest_status="PASS", keyframe_review_result="수정")
    assert qa.suggest(row, _args(qa))[0] == "REVIEW"


def test_main_writes_table(qa, seq_mod, standing_pose, tmp_path, capsys):
    seq_dir = tmp_path / "sequences"
    seq_dir.mkdir()
    for clip_id in ("T01_S01_A03_T01", "T01_S01_A05_T01"):
        make_sequence(seq_mod, standing_pose, clip_id).to_csv(
            seq_dir / f"{clip_id}_sequence.csv", index=False
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "clip_id": ["T01_S01_A03_T01", "T01_S01_A05_T01", "T01_S02_A03_T01"],
            "review_status": ["PASS", "REVIEW", "PASS"],
        }
    ).to_csv(manifest, index=False)
    output = tmp_path / "qa.csv"
    qa.main(
        [
            "--sequence-dir",
            str(seq_dir),
            "--manifest",
            str(manifest),
            "--keyframe-review",
            str(tmp_path / "none.csv"),
            "--output",
            str(output),
        ]
    )
    table = pd.read_csv(output)
    assert list(table.columns) == qa.OUTPUT_COLUMNS
    assert list(table["suggested_status"]) == ["PASS", "REVIEW"]
    out = capsys.readouterr().out
    assert "Sequence 없는 clip 1개" in out and "T01_S02_A03_T01" in out