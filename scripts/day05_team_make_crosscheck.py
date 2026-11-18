from __future__ import annotations

# ==================================================
# day05_team_make_crosscheck.py
# 구간 라벨 교차 검수 표본 45개(10%) 고르기 + 검수자 5명 배정 (02_라벨링_가이드 8장 5단계)
# ==================================================
# 고르는 순서: 자동 라벨 REVIEW → 검증 경고(V-07 · V-08) → 신호 15개 × 2 → 일상 동작 3 → 헷갈리기 쉬운 동작
# 배정: 사람마다 9개, **자기 영상은 자기가 보지 않음**
# 출력 (reports/team_projects/team01/main/labels/crosscheck/)
#   crosscheck_plan.csv      : 45개 전체 · 고른 이유 · 검수자
#   crosscheck_S01.csv ~ S05 : 검수자별 빈 검수표 (자동 경계는 안 넣음 = 보고 따라 적지 않게)
#
# 실행
#   uv run python scripts/day05_team_make_crosscheck.py
#   uv run python scripts/day05_team_label_sheet.py --blind --step 3 \
#       --clips-from reports/team_projects/team01/main/labels/crosscheck/crosscheck_plan.csv
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_crosscheck_utils import assign_reviewers, blank_review_sheet, pick_sample
from src.team_label_utils import validate_labels

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(MAIN_DIR / "labels" / "team01_labels.csv"))
    parser.add_argument("--manifest", default=str(ROOT / "configs" / "team01_main_manifest.csv"))
    parser.add_argument("--output-dir", default=str(MAIN_DIR / "labels" / "crosscheck"))
    parser.add_argument("--n", type=int, default=45, help="표본 수 (전체의 10%)")
    parser.add_argument("--overwrite", action="store_true", help="이미 적은 검수표가 있어도 새로 만듦")
    return parser.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.glob("crosscheck_S*.csv")) and not args.overwrite:
        print(f"[STOP] 검수표가 이미 있음: {out} (다시 만들려면 --overwrite — 적은 내용이 지워짐)")
        return pd.DataFrame()
    labels = pd.read_csv(args.labels, keep_default_na=False)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False)
    action_labels = dict(zip(manifest["action_id"], manifest["action_label"]))
    _, warnings = validate_labels(labels)
    warn_clips = sorted({m.group(0) for w in warnings for m in [re.search(r"T01_S\d\d_[AD]\d\d_T\d\d", w)] if m})

    sample = pick_sample(labels, warn_clips, n_total=args.n)
    assigned = assign_reviewers(sample)
    out.mkdir(parents=True, exist_ok=True)
    plan = assigned[["reviewer", "clip_id", "subject_id", "action_id", "frames", "why"]].copy()
    plan["action_label"] = plan["action_id"].map(action_labels)
    plan.to_csv(out / "crosscheck_plan.csv", index=False, encoding="utf-8-sig")
    sheets = blank_review_sheet(assigned, action_labels)
    for reviewer, g in sheets.groupby("reviewer"):
        g.to_csv(out / f"crosscheck_{reviewer}.csv", index=False, encoding="utf-8-sig")

    print(f"표본 {len(plan)}개 / 전체 {labels['clip_id'].nunique()}개")
    print(plan["why"].value_counts().to_string())
    print()
    print(pd.crosstab(plan["reviewer"], plan["subject_id"]).to_string())
    print(f"\nSaved: {out}")
    return plan


if __name__ == "__main__":
    main()
