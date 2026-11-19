from __future__ import annotations

# ==================================================
# scripts/day04_pose_conf_report.py 테스트
# ==================================================
# 가짜 Sequence CSV 2개로 확인합니다.
# - pose_ok = 0 Frame은 관절 평균에서 빠지고, clip의 pose_ok 비율에만 들어가는지
# - 손목 conf가 기준보다 낮은 Frame이 많으면 CHECK가 되는지
# - Sequence CSV가 없는 clip은 missing으로 남고, Markdown 요약이 만들어지는지
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    path = ROOT / "scripts" / "day04_pose_conf_report.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sequence(mod, n, conf=0.9, wrist=0.9, pose_ok_last=True):
    rows = []
    for i in range(n):
        row = {"frame_id": i, "pose_ok": 1}
        for name in mod.COCO_KEYPOINT_NAMES:
            row[f"conf_{name}"] = conf
        row["conf_left_wrist"] = wrist
        rows.append(row)
    if not pose_ok_last:
        rows[-1]["pose_ok"] = 0
        for name in mod.COCO_KEYPOINT_NAMES:
            rows[-1][f"conf_{name}"] = None
    return pd.DataFrame(rows)


@pytest.fixture()
def workdir(tmp_path, mod):
    seq = tmp_path / "seq"
    seq.mkdir()
    _sequence(mod, 10, pose_ok_last=False).to_csv(seq / "T01_S01_A01_T01_sequence.csv", index=False)
    _sequence(mod, 10, wrist=0.3).to_csv(seq / "T01_S05_A02_T01_sequence.csv", index=False)
    manifest = pd.DataFrame(
        {
            "clip_id": ["T01_S01_A01_T01", "T01_S05_A02_T01", "T01_S05_A03_T01"],
            "subject_id": ["S01", "S05", "S05"],
            "action_id": ["A01", "A02", "A03"],
        }
    )
    manifest.to_csv(tmp_path / "m.csv", index=False)
    return tmp_path


def test_tables_and_status(mod, workdir):
    args = mod.parse_args(["--manifest", str(workdir / "m.csv"), "--sequence-dir", str(workdir / "seq")])
    manifest = pd.read_csv(workdir / "m.csv")
    frames, missing = mod.load_sequences(manifest, workdir / "seq")
    assert missing == ["T01_S05_A03_T01"]

    joints = mod.joint_table(frames, args.kp_conf).set_index("joint")
    # pose_ok = 0 인 1 Frame은 빠짐 → 9 + 10
    assert joints.loc["nose", "frames"] == 19
    assert joints.loc["left_wrist", "min"] == pytest.approx(0.3)
    assert joints.loc["left_wrist", "low_ratio"] == pytest.approx(10 / 19, abs=1e-4)
    assert joints.loc["left_wrist", "group"] == "팔"

    clips = mod.clip_table(frames, args).set_index("clip_id")
    assert clips.loc["T01_S01_A01_T01", "pose_ok_ratio"] == pytest.approx(0.9)
    assert clips.loc["T01_S01_A01_T01", "status"] == "PASS"  # 0.9 = 기준과 같으면 통과
    assert clips.loc["T01_S05_A02_T01", "status"] == "CHECK"
    assert clips.loc["T01_S05_A02_T01", "weakest_joint"] == "left_wrist"


def test_main_writes_files(mod, workdir):
    out = workdir / "tables"
    md = workdir / "summary.md"
    code = mod.main(
        ["--manifest", str(workdir / "m.csv"), "--sequence-dir", str(workdir / "seq"),
         "--output-dir", str(out), "--summary", str(md)]
    )
    assert code == 0
    for name in ["pose_conf_joints.csv", "pose_conf_clips.csv", "pose_conf_actions.csv"]:
        assert (out / name).exists()
    text = md.read_text(encoding="utf-8")
    assert "관절별" in text and "CHECK 1" in text and "Sequence CSV 없음 1개" in text


def test_no_sequences_returns_1(mod, tmp_path):
    pd.DataFrame({"clip_id": ["T01_S01_A01_T01"], "subject_id": ["S01"], "action_id": ["A01"]}).to_csv(
        tmp_path / "m.csv", index=False
    )
    assert mod.main(["--manifest", str(tmp_path / "m.csv"), "--sequence-dir", str(tmp_path), "--summary", ""]) == 1
