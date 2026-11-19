from __future__ import annotations

# ==================================================
# day05_team_signal_feature_check.py
# G-01 ~ G-10이 설계(04 문서 5.2)대로 동작을 가르는지 — 라벨 구간별 평균표
# ==================================================
# labels.csv의 신호(A##) · C01 구간과 일상 동작(D##) clip 전체에서 G-01~G-10 평균을 냄
# → "이 Feature가 정말 A15와 팔짱을 가르나?" 를 6일차 E-01(38-D ↔ 48-D) 전에 눈으로 확인
#
# 출력: reports/team_projects/team01/main/features/signal_feature_by_label.csv
# 실행: uv run python scripts/day05_team_signal_feature_check.py
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import SIGNAL_FEATURE_NAMES, add_signal_features

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(MAIN_DIR / "labels" / "team01_labels.csv"))
    parser.add_argument("--sequence-dir", default=str(MAIN_DIR / "sequences"))
    parser.add_argument("--output", default=str(MAIN_DIR / "features" / "signal_feature_by_label.csv"))
    return parser.parse_args(argv)


def segment_means(labels: pd.DataFrame, sequence_dir: Path) -> pd.DataFrame:
    rows = []
    for clip, g in labels.groupby("clip_id"):
        path = sequence_dir / f"{clip}_sequence.csv"
        if not path.exists() or (g["qa_status"] == "EXCLUDE").any():
            continue
        seq = add_signal_features(pd.read_csv(path))
        action = clip.split("_")[2]
        parts = []
        if action.startswith("D"):
            parts.append((f"N00({action})", seq))
        for _, r in g[g["label"].str.match(r"^(A\d\d|C01)$")].iterrows():
            parts.append((r["label"], seq.iloc[int(r["start_frame"]):int(r["end_frame"]) + 1]))
        for name, part in parts:
            row = {"label": name, "clip_id": clip, "subject_id": clip.split("_")[1]}
            row.update(part[SIGNAL_FEATURE_NAMES].mean().to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    labels = pd.read_csv(args.labels, keep_default_na=False)
    per_clip = segment_means(labels, Path(args.sequence_dir))
    table = per_clip.groupby("label")[SIGNAL_FEATURE_NAMES].mean().round(2)
    table.insert(0, "clips", per_clip.groupby("label").size())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(table.to_string())
    print(f"\nSaved: {out}")
    return table


if __name__ == "__main__":
    main()
