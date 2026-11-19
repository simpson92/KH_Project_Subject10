from __future__ import annotations

# ==================================================
# day06_team_compare.py   (6일차 실습 38 · 39번 → 1조 버전)
# 설정 파일 2개(예: 38-D 기본 ↔ E-01 48-D)의 Validation 결과 비교
# ==================================================
# 왜?  한 번에 하나만 바꾼 실험(수업 39번 원칙)을 같은 Val S05로 비교해서 F1 결정 근거를 남김.
#      ① 이미 만든 val_predictions.csv 비교: Accuracy · Macro F1 · 클래스별 Recall · 바뀐 clip(고쳐짐 / 새로 틀림)
#      ② seed 여러 개 재학습: Val 90개라 clip 1개 = 0.011 → random_state 하나만 보면 우연일 수 있어서
#         seed 0 ~ N-1로 다시 학습해 평균 · 최소 · 최대를 함께 봄 (Test는 쓰지 않음)
#
# 입력: day06_team_build_sequences · train_rf · evaluate 를 각 설정으로 먼저 실행한 결과
# 출력: reports/team_projects/team01/main/day06/compare/<A>_vs_<B>_metrics.csv     (①② 요약)
#       reports/team_projects/team01/main/day06/compare/<A>_vs_<B>_per_class.csv   (클래스별 Recall)
#       reports/team_projects/team01/main/day06/compare/<A>_vs_<B>_clips.csv       (결과가 바뀐 clip)
#
# 실행
#   uv run python scripts/day06_team_compare.py \
#     --configs configs/team01_day06_baseline.json configs/team01_day06_e01_48d.json
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"
SEQ_DIR = ROOT / "data" / "team_projects" / "team01" / "main" / "day06" / "sequences"
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
        "--configs", nargs=2, required=True, help="비교할 설정 2개 (기준 → 실험)"
    )
    parser.add_argument(
        "--seeds", type=int, default=10, help="재학습 seed 수 (0이면 재학습 안 함)"
    )
    parser.add_argument("--report-dir", default=str(MAIN_DIR / "day06"))
    parser.add_argument("--seq-dir", default=str(SEQ_DIR))
    return parser.parse_args(argv)


def load_run(config_path, report_dir: Path, seq_dir: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    run = config.get("run_name", "")
    return {
        "name": run or "baseline",
        "config": config,
        "report_dir": report_dir / run if run else report_dir,
        "seq_dir": seq_dir / run if run else seq_dir,
    }


def prediction_metrics(pred: pd.DataFrame, labels: list) -> dict:
    return {
        "accuracy": accuracy_score(pred["true_label"], pred["pred_label"]),
        "macro_f1": f1_score(
            pred["true_label"],
            pred["pred_label"],
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "wrong": int((pred["correct"] == 0).sum()),
    }


def per_class_recall(pred: pd.DataFrame, labels: list) -> np.ndarray:
    return recall_score(
        pred["true_label"],
        pred["pred_label"],
        labels=labels,
        average=None,
        zero_division=0,
    )


def changed_clips(
    a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str
) -> pd.DataFrame:
    m = a.merge(b, on=["clip_id", "true_label"], suffixes=(f"_{name_a}", f"_{name_b}"))
    m = m[m[f"correct_{name_a}"] != m[f"correct_{name_b}"]].copy()
    m["change"] = np.where(m[f"correct_{name_b}"] == 1, "고쳐짐", "새로 틀림")
    keep = [
        "clip_id",
        "true_label",
        f"pred_label_{name_a}",
        f"pred_label_{name_b}",
        "change",
    ]
    return m[keep].sort_values(["change", "clip_id"]).reset_index(drop=True)


def seed_scores(run: dict, seeds: int) -> list:
    cfg = run["config"]
    train = pd.read_csv(run["seq_dir"] / "train.csv", encoding="utf-8-sig")
    val = pd.read_csv(run["seq_dir"] / "val.csv", encoding="utf-8-sig")
    cols = [c for c in train.columns if c not in META_COLUMNS]
    labels = list(cfg["labels"])
    scores = []
    for seed in range(seeds):
        model = RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=cfg.get("max_depth"),
            class_weight=cfg.get("class_weight"),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train[cols].to_numpy(dtype=float), train["train_label"].astype(str))
        pred = model.predict(val[cols].to_numpy(dtype=float))
        y = val["train_label"].astype(str)
        scores.append(
            (
                accuracy_score(y, pred),
                f1_score(y, pred, labels=labels, average="macro", zero_division=0),
            )
        )
    return scores


def main(argv=None) -> int:
    args = parse_args(argv)
    runs = [
        load_run(c, Path(args.report_dir), Path(args.seq_dir)) for c in args.configs
    ]
    a, b = runs
    labels = list(a["config"]["labels"])
    preds = []
    for run in runs:
        path = run["report_dir"] / "val_predictions.csv"
        if not path.exists():
            print(
                f"[ERROR] 없음: {path} (day06_team_evaluate.py --config ... 먼저 실행)"
            )
            return 1
        preds.append(pd.read_csv(path, encoding="utf-8-sig"))

    rows = []
    for run, pred in zip(runs, preds):
        m = prediction_metrics(pred, labels)
        cfg = run["config"]
        row = {
            "run": run["name"],
            "feature_set": cfg.get("feature_set", "base38"),
            "features": int(cfg["sequence_steps"])
            * (48 if cfg.get("feature_set") == "team48" else 38),
            "val_accuracy": round(m["accuracy"], 4),
            "val_macro_f1": round(m["macro_f1"], 4),
            "val_wrong": m["wrong"],
        }
        if args.seeds > 0:
            scores = np.array(seed_scores(run, args.seeds))
            row.update(
                {
                    "seeds": args.seeds,
                    "acc_mean": round(scores[:, 0].mean(), 4),
                    "acc_min": round(scores[:, 0].min(), 4),
                    "acc_max": round(scores[:, 0].max(), 4),
                    "f1_mean": round(scores[:, 1].mean(), 4),
                    "f1_min": round(scores[:, 1].min(), 4),
                    "f1_max": round(scores[:, 1].max(), 4),
                }
            )
        rows.append(row)
    metrics = pd.DataFrame(rows)
    per_class = pd.DataFrame({"label": labels})
    for run, pred in zip(runs, preds):
        per_class[f"recall_{run['name']}"] = per_class_recall(pred, labels).round(4)
    per_class["diff"] = (
        per_class[f"recall_{b['name']}"] - per_class[f"recall_{a['name']}"]
    ).round(4)
    clips = changed_clips(preds[0], preds[1], a["name"], b["name"])

    out_dir = Path(args.report_dir) / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a['name']}_vs_{b['name']}"
    metrics.to_csv(out_dir / f"{stem}_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(
        out_dir / f"{stem}_per_class.csv", index=False, encoding="utf-8-sig"
    )
    clips.to_csv(out_dir / f"{stem}_clips.csv", index=False, encoding="utf-8-sig")

    print(f"비교: {a['name']} → {b['name']} (Validation S05)")
    for r in rows:
        line = f"  {r['run']:10s} {r['features']:>5} Feature · Acc {r['val_accuracy']:.4f} · Macro F1 {r['val_macro_f1']:.4f} · 오답 {r['val_wrong']}"
        if args.seeds > 0:
            line += f"  | seed {args.seeds}개 평균 Acc {r['acc_mean']:.4f} ({r['acc_min']:.4f}~{r['acc_max']:.4f}) · F1 {r['f1_mean']:.4f}"
        print(line)
    moved = per_class[per_class["diff"] != 0]
    if len(moved):
        print(
            "  Recall 바뀐 클래스: "
            + " · ".join(f"{r.label} {r.diff:+.2f}" for r in moved.itertuples())
        )
    fixed = int((clips["change"] == "고쳐짐").sum())
    broken = int((clips["change"] == "새로 틀림").sum())
    print(f"  바뀐 clip: 고쳐짐 {fixed} · 새로 틀림 {broken}")
    print()
    print(f"Saved: {out_dir}/{stem}_metrics.csv · _per_class.csv · _clips.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
