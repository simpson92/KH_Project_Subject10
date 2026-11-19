from __future__ import annotations

# ==================================================
# scripts/day06_team_compare.py 테스트 — 두 설정의 Val 결과 비교 (지표 · 클래스별 Recall · 바뀐 clip · seed 반복)
# ==================================================
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cmp6", ROOT / "scripts" / "day06_team_compare.py"
)
cmp6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp6)

LABELS = ["A01", "N00"]


def _pred(correct):
    true = ["A01", "A01", "N00", "N00"]
    pred = [t if c else ("N00" if t == "A01" else "A01") for t, c in zip(true, correct)]
    return pd.DataFrame(
        {
            "clip_id": ["c1", "c2", "c3", "c4"],
            "subject": "S05",
            "action_id": true,
            "true_label": true,
            "pred_label": pred,
            "confidence": 0.5,
            "correct": correct,
        }
    )


def _seq(n_per=3, dim=2):
    rng = np.random.default_rng(0)
    rows = []
    for k, label in enumerate(LABELS):
        for i in range(n_per):
            row = {c: "" for c in cmp6.META_COLUMNS}
            row.update({"clip_id": f"{label}{i}", "train_label": label})
            row.update({f"f{j}": k + rng.normal(0, 0.05) for j in range(dim)})
            rows.append(row)
    return pd.DataFrame(rows)


def _setup(tmp: Path):
    rep, seq = tmp / "rep", tmp / "seq"
    (rep / "e01").mkdir(parents=True)
    (seq / "e01").mkdir(parents=True)
    _pred([1, 1, 0, 1]).to_csv(rep / "val_predictions.csv", index=False)
    _pred([1, 0, 1, 1]).to_csv(rep / "e01" / "val_predictions.csv", index=False)
    for d in (seq, seq / "e01"):
        _seq().to_csv(d / "train.csv", index=False)
        _seq().to_csv(d / "val.csv", index=False)
    base = {
        "sequence_steps": 1,
        "labels": LABELS,
        "n_estimators": 5,
        "max_depth": None,
        "class_weight": "balanced",
    }
    (tmp / "a.json").write_text(json.dumps(base))
    (tmp / "b.json").write_text(
        json.dumps({**base, "run_name": "e01", "feature_set": "team48"})
    )
    return [
        "--configs",
        str(tmp / "a.json"),
        str(tmp / "b.json"),
        "--report-dir",
        str(rep),
        "--seq-dir",
        str(seq),
    ]


def test_changed_clips():
    clips = cmp6.changed_clips(
        _pred([1, 1, 0, 1]), _pred([1, 0, 1, 1]), "baseline", "e01"
    )
    assert dict(zip(clips["clip_id"], clips["change"])) == {
        "c3": "고쳐짐",
        "c2": "새로 틀림",
    }


def test_main_writes_three_files(tmp_path):
    args = _setup(tmp_path)
    assert cmp6.main([*args, "--seeds", "2"]) == 0
    out = tmp_path / "rep" / "compare"
    m = pd.read_csv(out / "baseline_vs_e01_metrics.csv")
    assert list(m["run"]) == ["baseline", "e01"] and list(m["val_wrong"]) == [1, 1]
    assert list(m["features"]) == [38, 48] and set(m["seeds"]) == {2}
    assert m["acc_mean"].between(0, 1).all()
    pc = pd.read_csv(out / "baseline_vs_e01_per_class.csv")
    assert list(pc["diff"]) == [-0.5, 0.5]
    assert len(pd.read_csv(out / "baseline_vs_e01_clips.csv")) == 2


def test_missing_predictions_stops(tmp_path):
    args = _setup(tmp_path)
    (tmp_path / "rep" / "e01" / "val_predictions.csv").unlink()
    assert cmp6.main([*args, "--seeds", "0"]) == 1
