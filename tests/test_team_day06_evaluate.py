from __future__ import annotations

# ==================================================
# scripts/day06_team_evaluate.py 테스트 — Val 평가 · Test 잠금(--final) · 예상 혼동 쌍 표시
# ==================================================
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "eval6", ROOT / "scripts" / "day06_team_evaluate.py"
)
eval6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval6)

LABELS = ["A02", "A03", "N00"]
COLS = [f"f{j}" for j in range(4)]


def _seq(rows):
    """rows = [(clip_id, action_id, train_label, center)]"""
    data = []
    for clip_id, action_id, label, center in rows:
        row = {
            "clip_id": clip_id,
            "subject": "S05",
            "action_id": action_id,
            "train_label": label,
            "take": "T01",
            "valid_frames": 100,
            "first_valid_frame": 0,
            "last_valid_frame": 99,
        }
        row.update({c: float(center) for c in COLS})
        data.append(row)
    return pd.DataFrame(data)


def _setup(tmp: Path):
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = np.array([[0.0] * 4, [1.0] * 4, [2.0] * 4] * 3)
    y = np.array(LABELS * 3)
    model.fit(X, y)
    art = tmp / "art"
    art.mkdir()
    joblib.dump(model, art / "rf_team_baseline.joblib")
    (art / "rf_feature_columns.json").write_text(json.dumps(COLS))
    seq = tmp / "seq"
    seq.mkdir()
    # A02 1개를 A03 자리(1.0)에 둬서 일부러 틀리게 함 → A02↔A03은 예상 혼동 쌍
    val = _seq(
        [
            ("c1", "A02", "A02", 0),
            ("c2", "A02", "A02", 1),
            ("c3", "A03", "A03", 1),
            ("c4", "D01", "N00", 2),
        ]
    )
    val.to_csv(seq / "val.csv", index=False)
    val.to_csv(seq / "test.csv", index=False)
    splits = tmp / "splits"
    splits.mkdir()
    val[["clip_id"]].assign(video_path="v.mp4", keypoint_csv="k.csv").to_csv(
        splits / "val.csv", index=False
    )
    cfg = tmp / "cfg.json"
    cfg.write_text(json.dumps({"labels": LABELS}))
    classes = tmp / "classes.csv"
    classes.write_text(
        "action_id,confusable\nA02,A03\nA03,A02\nN00,\nD01,A11\n", encoding="utf-8"
    )
    return [
        "--config",
        str(cfg),
        "--classes-csv",
        str(classes),
        "--seq-dir",
        str(seq),
        "--split-dir",
        str(splits),
        "--artifact-dir",
        str(art),
        "--output-dir",
        str(tmp / "out"),
    ]


def test_load_confusable_is_two_way(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("action_id,confusable\nA08,A09 A12\nA09,\n", encoding="utf-8")
    pairs = eval6.load_confusable(p)
    assert pairs["A08"] == {"A09", "A12"} and "A08" in pairs["A12"]


def test_val_outputs_and_known_confusable(tmp_path):
    base = _setup(tmp_path)
    assert eval6.main(["--split", "val", *base]) == 0
    out = tmp_path / "out"
    pred = pd.read_csv(out / "val_predictions.csv")
    assert list(pred.columns) == [
        "clip_id",
        "subject",
        "action_id",
        "true_label",
        "pred_label",
        "confidence",
        "correct",
    ]
    wrong = pd.read_csv(out / "val_misclassified.csv")
    assert list(wrong["clip_id"]) == ["c2"]
    assert wrong.loc[0, "pred_label"] == "A03" and wrong.loc[0, "known_confusable"] == 1
    assert wrong.loc[0, "video_path"] == "v.mp4"
    text = (out / "val_metrics.txt").read_text(encoding="utf-8")
    assert "Accuracy: 0.7500" in text and "Correct: 3 / 4" in text
    per_class = pd.read_csv(out / "val_per_class.csv")
    assert list(per_class["label"]) == LABELS
    assert (out / "val_confusion_matrix.png").exists()


def test_test_split_needs_final(tmp_path):
    base = _setup(tmp_path)
    assert eval6.main(["--split", "test", *base]) == 1
    assert not (tmp_path / "out" / "test_metrics.txt").exists()
    assert eval6.main(["--split", "test", "--final", *base]) == 0
    assert (tmp_path / "out" / "test_metrics.txt").exists()


def test_feature_mismatch_stops(tmp_path):
    base = _setup(tmp_path)
    (tmp_path / "art" / "rf_feature_columns.json").write_text(
        json.dumps([*COLS, "extra"])
    )
    assert eval6.main(["--split", "val", *base]) == 1
