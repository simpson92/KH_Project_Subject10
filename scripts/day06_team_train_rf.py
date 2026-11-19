from __future__ import annotations

# ==================================================
# day06_team_train_rf.py   (6일차 실습 35번 → 1조 버전)
# 1조 Train(S01 · S03 · S04, 270 clip) → Random Forest 학습 · 저장
# ==================================================
# 왜?  수업 35번은 6행 · 6클래스가 코드에 고정되어 있어 1조 270행 · 16클래스에 못 씀.
#      X = 1,140개 숫자 (t00_nose_x … t29_right_knee_angle)
#      y = train_label (A01~A15 · N00)  ← 정답은 X에 넣지 않음 (정답 누출 방지)
#      N00이 신호보다 3배 많아서 class_weight="balanced" (결정 기록 C10)
#      평가는 하지 않음 → day06_team_evaluate.py (Val S05)
#
# 입력: data/team_projects/team01/main/day06/sequences/train.csv   (day06_team_build_sequences.py 결과)
#       configs/team01_day06_baseline.json
# 출력: artifacts/team01/day06/rf_team_baseline.joblib    (모델, Git 제외)
#       artifacts/team01/day06/rf_feature_columns.json     (Feature 이름 · 순서, git add -f)
#       artifacts/team01/day06/rf_training_info.json       (설정 · 클래스별 개수, git add -f)
#
# 실행
#   uv run python scripts/day06_team_train_rf.py                                             # 38-D 기본
#   uv run python scripts/day06_team_train_rf.py --config configs/team01_day06_e01_48d.json  # E-01 48-D
#   (설정에 run_name이 있으면 입력 sequences/<run_name>/ · 출력 artifacts/team01/day06/<run_name>/)
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KST = timezone(timedelta(hours=9))
SEQ_DIR = ROOT / "data" / "team_projects" / "team01" / "main" / "day06" / "sequences"
ART_DIR = ROOT / "artifacts" / "team01" / "day06"
META_COLUMNS = [
    "clip_id",
    "subject",
    "action_id",
    "train_label",
    "take",
    "valid_frames",
    "first_valid_frame",
    "last_valid_frame",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "team01_day06_baseline.json")
    )
    parser.add_argument("--train-csv", default=str(SEQ_DIR / "train.csv"))
    parser.add_argument("--artifact-dir", default=str(ART_DIR))
    return parser.parse_args(argv)


def split_xy(df: pd.DataFrame, label_column: str = "train_label"):
    feature_columns = [c for c in df.columns if c not in META_COLUMNS]
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[label_column].astype(str).to_numpy()
    return X, y, feature_columns


FEATURE_DIMS = {"base38": 38, "team48": 48}


def check_train(df: pd.DataFrame, labels: list, steps: int, dim: int = 38) -> list:
    errors = []
    feature_columns = [c for c in df.columns if c not in META_COLUMNS]
    if len(feature_columns) != steps * dim:
        errors.append(f"Feature 수 {len(feature_columns)} ≠ {steps} × {dim}")
    missing = sorted(set(labels) - set(df["train_label"].astype(str)))
    if missing:
        errors.append(f"Train에 없는 라벨 {missing}")
    if df[feature_columns].isna().any().any():
        errors.append("Feature에 빈 값(NaN)이 있음")
    return errors


def make_model(config: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=config.get("max_depth"),
        random_state=int(config["random_state"]),
        class_weight=config.get("class_weight"),
        n_jobs=-1,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    steps = int(config["sequence_steps"])
    labels = list(config["labels"])
    feature_set = config.get("feature_set", "base38")
    dim = FEATURE_DIMS[feature_set]
    run = config.get("run_name", "")

    train_path = Path(args.train_csv)
    art_dir = Path(args.artifact_dir)
    if run:  # 실험은 하위 폴더 (기본 38-D 결과는 그대로)
        train_path = train_path.parent / run / train_path.name
        art_dir = art_dir / run
    if not train_path.exists():
        print(f"[ERROR] 없음: {train_path} (day06_team_build_sequences.py 먼저 실행)")
        return 1
    train = pd.read_csv(train_path, encoding="utf-8-sig")

    errors = check_train(train, labels, steps, dim)
    if errors:
        print("[ERROR] Train 검사 실패")
        for e in errors:
            print("  -", e)
        return 1

    X_train, y_train, feature_columns = split_xy(train)
    counts = (
        train["train_label"].astype(str).value_counts().reindex(labels, fill_value=0)
    )
    print(
        f"Train: {'·'.join(sorted(train['subject'].unique()))} / {len(train)} clips / {len(labels)} classes"
    )
    print(
        f"X_train: {X_train.shape}  ({dim}-D {feature_set} · 정답 train_label은 X에 없음)"
    )
    print("클래스별 clip 수: " + " ".join(f"{k}={v}" for k, v in counts.items()))

    model = make_model(config)
    model.fit(X_train, y_train)

    art_dir.mkdir(parents=True, exist_ok=True)
    model_path = art_dir / "rf_team_baseline.joblib"
    joblib.dump(model, model_path)
    with open(art_dir / "rf_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    info = {
        "trained_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "train_csv": str(train_path.relative_to(ROOT))
        if train_path.is_relative_to(ROOT)
        else str(train_path),
        "train_subjects": sorted(train["subject"].unique().tolist()),
        "train_clips": len(train),
        "run_name": run or "baseline",
        "feature_set": feature_set,
        "feature_count": len(feature_columns),
        "sequence_steps": steps,
        "classes": [str(c) for c in model.classes_],
        "class_counts": {k: int(v) for k, v in counts.items()},
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": int(config["n_estimators"]),
            "max_depth": config.get("max_depth"),
            "class_weight": config.get("class_weight"),
            "random_state": int(config["random_state"]),
        },
    }
    with open(art_dir / "rf_training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    train_acc = float((model.predict(X_train) == y_train).mean())
    print()
    print(f"Train Accuracy (참고용, 본 데이터라 높게 나옴): {train_acc:.4f}")
    print(f"Saved: {model_path}")
    print(f"Saved: {art_dir / 'rf_feature_columns.json'}")
    print(f"Saved: {art_dir / 'rf_training_info.json'}")
    print("1조 RF Baseline 학습: DONE (평가는 day06_team_evaluate.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
