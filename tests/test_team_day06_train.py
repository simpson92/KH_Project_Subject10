from __future__ import annotations

# ==================================================
# scripts/day06_team_train_rf.py 테스트 — 1조 RF 학습 (정답은 X에 없음 · 16클래스 · 저장 파일 3개)
# ==================================================
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train6", ROOT / "scripts" / "day06_team_train_rf.py"
)
train6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train6)

STEPS = 2  # 테스트는 2 Step × 38 = 76 Feature


def _train_df(labels=("A01", "N00"), per_label=3, steps=STEPS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k, label in enumerate(labels):
        for i in range(per_label):
            row = {
                "clip_id": f"{label}_{i}",
                "subject": "S01",
                "action_id": label,
                "train_label": label,
                "take": f"T0{i + 1}",
                "valid_frames": 100,
                "first_valid_frame": 0,
                "last_valid_frame": 99,
            }
            row.update(
                {f"f{j}": float(k) + rng.normal(0, 0.1) for j in range(steps * 38)}
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_xy_excludes_meta_and_label():
    X, y, cols = train6.split_xy(_train_df())
    assert X.shape == (6, STEPS * 38)
    assert "train_label" not in cols and "clip_id" not in cols
    assert list(y[:3]) == ["A01", "A01", "A01"]


def test_check_train_finds_problems():
    df = _train_df(labels=("A01",))
    df.loc[0, "f0"] = np.nan
    errors = " | ".join(train6.check_train(df, ["A01", "N00"], STEPS))
    assert "없는 라벨 ['N00']" in errors
    assert "NaN" in errors
    assert "Feature 수" in " ".join(train6.check_train(_train_df(), ["A01", "N00"], 30))


def test_main_saves_model_and_info(tmp_path):
    csv = tmp_path / "train.csv"
    _train_df().to_csv(csv, index=False)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "sequence_steps": STEPS,
                "labels": ["A01", "N00"],
                "random_state": 42,
                "n_estimators": 10,
                "max_depth": None,
                "class_weight": "balanced",
            }
        )
    )
    art = tmp_path / "art"
    assert (
        train6.main(
            ["--config", str(cfg), "--train-csv", str(csv), "--artifact-dir", str(art)]
        )
        == 0
    )
    model = joblib.load(art / "rf_team_baseline.joblib")
    assert list(model.classes_) == ["A01", "N00"]
    assert model.class_weight == "balanced"
    cols = json.loads((art / "rf_feature_columns.json").read_text(encoding="utf-8"))
    assert len(cols) == STEPS * 38
    info = json.loads((art / "rf_training_info.json").read_text(encoding="utf-8"))
    assert info["train_clips"] == 6 and info["class_counts"] == {"A01": 3, "N00": 3}


def test_main_fails_without_train_csv(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"sequence_steps": STEPS, "labels": ["A01"]}))
    assert (
        train6.main(["--config", str(cfg), "--train-csv", str(tmp_path / "none.csv")])
        == 1
    )
