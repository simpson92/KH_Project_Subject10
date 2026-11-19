from __future__ import annotations

# ==================================================
# day06_team_evaluate.py   (6일차 실습 36 · 37 · 40번 → 1조 버전)
# 학습한 RF로 Validation(S05) 또는 마지막 Test(S02) 평가
# ==================================================
# 왜?  수업 36 · 37 · 40번은 6 clip · 6클래스가 고정. 1조는 90 clip · 16클래스.
#      Accuracy · Macro F1 · 클래스별 Precision/Recall · 혼동행렬 · 오분류 목록을 한 번에 만듦.
#      오분류에는 원본 영상 · Keypoint 경로(수업 37번)와 "설계 때 예상한 혼동 쌍인지"를 붙임
#      (configs/team01_action_classes.csv의 confusable 열, 예: A02 홀딩 ↔ A03 U파울).
#      Test(S02)는 결정 기록대로 맨 마지막 1번만 → --split test는 --final을 같이 줘야 실행됨.
#
# 입력: data/team_projects/team01/main/day06/sequences/{val,test}.csv
#       artifacts/team01/day06/rf_team_baseline.joblib · rf_feature_columns.json
#       reports/team_projects/team01/main/dataset/splits/{val,test}.csv  (video_path · keypoint_csv)
# 출력: reports/team_projects/team01/main/day06/{split}_metrics.txt
#       reports/team_projects/team01/main/day06/{split}_predictions.csv   (true_label · pred_label · confidence · correct)
#       reports/team_projects/team01/main/day06/{split}_per_class.csv     (클래스별 precision · recall · f1 · support)
#       reports/team_projects/team01/main/day06/{split}_misclassified.csv (틀린 clip + 영상 경로 + 예상 혼동 여부)
#       reports/team_projects/team01/main/day06/{split}_confusion_matrix.png (png = Git 제외)
#
# 실행
#   uv run python scripts/day06_team_evaluate.py                       # Validation S05
#   uv run python scripts/day06_team_evaluate.py --split test --final  # 마지막 Test S02 (1번만)
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"
SEQ_DIR = ROOT / "data" / "team_projects" / "team01" / "main" / "day06" / "sequences"
ART_DIR = ROOT / "artifacts" / "team01" / "day06"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Test(S02) 평가는 마지막 1번만 — 이 옵션이 있어야 실행",
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "team01_day06_baseline.json")
    )
    parser.add_argument(
        "--classes-csv", default=str(ROOT / "configs" / "team01_action_classes.csv")
    )
    parser.add_argument("--seq-dir", default=str(SEQ_DIR))
    parser.add_argument("--split-dir", default=str(MAIN_DIR / "dataset" / "splits"))
    parser.add_argument("--artifact-dir", default=str(ART_DIR))
    parser.add_argument("--output-dir", default=str(MAIN_DIR / "day06"))
    return parser.parse_args(argv)


def load_confusable(path) -> dict:
    """action_id → 설계 때 헷갈릴 거라 적어 둔 action_id 집합 (양방향)."""
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    pairs = {}
    for row in df.to_dict("records"):
        for other in str(row.get("confusable", "")).split():
            pairs.setdefault(row["action_id"], set()).add(other)
            pairs.setdefault(other, set()).add(row["action_id"])
    return pairs


def make_predictions(df: pd.DataFrame, model, feature_columns: list) -> pd.DataFrame:
    X = df[feature_columns].to_numpy(dtype=float)
    pred = model.predict(X)
    proba = model.predict_proba(X).max(axis=1)
    return pd.DataFrame(
        {
            "clip_id": df["clip_id"],
            "subject": df["subject"],
            "action_id": df["action_id"],
            "true_label": df["train_label"].astype(str),
            "pred_label": pred,
            "confidence": proba.round(4),
            "correct": (df["train_label"].astype(str).to_numpy() == pred).astype(int),
        }
    )


def make_misclassified(
    pred: pd.DataFrame, split_df: pd.DataFrame, confusable: dict
) -> pd.DataFrame:
    wrong = pred[pred["correct"] == 0].copy()
    # 일상 동작(D01~D03)은 action_id 기준으로 예상 혼동 쌍을 찾음 (예: D02 팔짱 ↔ A15 교체)
    wrong["known_confusable"] = [
        int(p in confusable.get(a, set()) or p in confusable.get(t, set()))
        for a, t, p in zip(wrong["action_id"], wrong["true_label"], wrong["pred_label"])
    ]
    keep = [
        c for c in ("clip_id", "video_path", "keypoint_csv") if c in split_df.columns
    ]
    return wrong.merge(split_df[keep], on="clip_id", how="left")


def per_class_table(pred: pd.DataFrame, labels: list) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        pred["true_label"], pred["pred_label"], labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "label": labels,
            "precision": p.round(4),
            "recall": r.round(4),
            "f1": f.round(4),
            "support": s,
        }
    )


def save_confusion(pred: pd.DataFrame, labels: list, title: str, path: Path) -> None:
    matrix = confusion_matrix(pred["true_label"], pred["pred_label"], labels=labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.split == "test" and not args.final:
        print("[STOP] Test(S02)는 최종 조건이 정해진 뒤 마지막에 1번만 평가합니다.")
        print(
            "       정말 마지막이면: uv run python scripts/day06_team_evaluate.py --split test --final"
        )
        return 1

    with open(args.config, encoding="utf-8") as f:
        labels = list(json.load(f)["labels"])
    art_dir = Path(args.artifact_dir)
    model_path = art_dir / "rf_team_baseline.joblib"
    seq_path = Path(args.seq_dir) / f"{args.split}.csv"
    for path in (model_path, art_dir / "rf_feature_columns.json", seq_path):
        if not path.exists():
            print(f"[ERROR] 없음: {path}")
            return 1

    model = joblib.load(model_path)
    with open(art_dir / "rf_feature_columns.json", encoding="utf-8") as f:
        feature_columns = json.load(f)
    df = pd.read_csv(seq_path, encoding="utf-8-sig")
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print(
            f"[ERROR] Feature 열이 학습 때와 다름 (없는 열 {len(missing)}개, 예: {missing[:3]})"
        )
        return 1

    pred = make_predictions(df, model, feature_columns)
    acc = accuracy_score(pred["true_label"], pred["pred_label"])
    macro_f1 = f1_score(
        pred["true_label"],
        pred["pred_label"],
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        pred["true_label"], pred["pred_label"], labels=labels, digits=4, zero_division=0
    )

    split_csv = Path(args.split_dir) / f"{args.split}.csv"
    split_df = (
        pd.read_csv(split_csv, encoding="utf-8-sig", keep_default_na=False)
        if split_csv.exists()
        else df[["clip_id"]]
    )
    wrong = make_misclassified(pred, split_df, load_confusable(args.classes_csv))
    per_class = per_class_table(pred, labels)

    name = "Validation" if args.split == "val" else "Test"
    subjects = "·".join(sorted(df["subject"].unique()))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(
        out_dir / f"{args.split}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    per_class.to_csv(
        out_dir / f"{args.split}_per_class.csv", index=False, encoding="utf-8-sig"
    )
    wrong.to_csv(
        out_dir / f"{args.split}_misclassified.csv", index=False, encoding="utf-8-sig"
    )
    save_confusion(
        pred,
        labels,
        f"Team01 RF Baseline - {name} ({subjects})",
        out_dir / f"{args.split}_confusion_matrix.png",
    )
    lines = [
        f"1조 RF Baseline — {name} ({subjects}, {len(pred)} clips)",
        f"Model: {model_path.name} · Feature {len(feature_columns)}",
        f"Accuracy: {acc:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        f"Correct: {int(pred['correct'].sum())} / {len(pred)}",
        f"Misclassified: {len(wrong)} (설계 때 예상한 혼동 쌍 {int(wrong['known_confusable'].sum())}개)",
        "",
        report,
    ]
    (out_dir / f"{args.split}_metrics.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n".join(lines[:6]))
    print()
    worst = per_class[per_class["recall"] < 1].sort_values(["recall", "f1"])
    if len(worst):
        names = " · ".join(f"{r.label} {r.recall:.2f}" for r in worst.itertuples())
        print("Recall 1.0 미만 클래스: " + names)
    if len(wrong):
        pairs = (
            wrong.groupby(["true_label", "pred_label"])
            .size()
            .sort_values(ascending=False)
            .head(5)
        )
        print(
            "자주 틀린 쌍 (정답 → 예측): "
            + " · ".join(f"{t}→{p} {n}" for (t, p), n in pairs.items())
        )
    print()
    print(
        f"Saved: {out_dir}/{args.split}_metrics.txt · _predictions.csv · _per_class.csv · _misclassified.csv · _confusion_matrix.png"
    )
    print(f"1조 {name} 평가: DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
