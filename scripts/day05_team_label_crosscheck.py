from __future__ import annotations

# ==================================================
# day05_team_label_crosscheck.py
# 교차 검수 결과 비교 — 검수자가 적은 경계 ↔ 자동 라벨 (02_라벨링_가이드 8장 5단계)
# ==================================================
# 판정: 경계 4개(C01 시작 · 끝, 신호 시작 · 끝) 차이 평균이 **0.2초 넘으면** 1단계 기준 다시 맞추기
# 출력: reports/team_projects/team01/main/labels/crosscheck/crosscheck_result.csv
# --apply : verdict = FIX 인 clip을 검수자 경계로 labels.csv에 반영 (labeler = 검수자 S##),
#           OK는 qa_status = PASS, RESHOOT는 RESHOOT
#
# 실행
#   uv run python scripts/day05_team_label_crosscheck.py            # 비교만
#   uv run python scripts/day05_team_label_crosscheck.py --apply    # 반영까지
#   uv run python scripts/day05_team_validate_labels.py             # 반영 후 V-01~V-10
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_crosscheck_utils import THRESHOLD_SEC, apply_fixes, compare, validate_review_sheet

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(MAIN_DIR / "labels" / "team01_labels.csv"))
    parser.add_argument("--dir", default=str(MAIN_DIR / "labels" / "crosscheck"))
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    folder = Path(args.dir)
    files = sorted(folder.glob("crosscheck_S*.csv"))
    if not files:
        print(f"[ERROR] 검수표 없음: {folder}")
        return 1
    review = pd.concat([pd.read_csv(f, encoding="utf-8-sig", keep_default_na=False) for f in files],
                       ignore_index=True)
    done = review[review["verdict"].astype(str).str.strip() != ""]
    print(f"검수표 {len(files)}개 · {len(review)}행 · 적은 것 {len(done)}행")
    errors = validate_review_sheet(done)
    for e in errors:
        print("[FIX ME]", e)

    labels = pd.read_csv(args.labels, keep_default_na=False)
    result = compare(done, labels)
    result.to_csv(folder / "crosscheck_result.csv", index=False, encoding="utf-8-sig")
    measured = result.dropna(subset=["mean_diff_sec"])
    if not measured.empty:
        overall = measured["mean_diff_sec"].mean()
        print(f"\n경계 차이 평균 {overall:.3f}초 (기준 {THRESHOLD_SEC}초) · clip {len(measured)}개")
        print(measured.groupby("reviewer")["mean_diff_sec"].mean().round(3).to_string())
        over = measured[measured["over_0_2s"]]
        if len(over):
            print(f"\n[0.2초 초과 {len(over)}개]")
            print(over[["reviewer", "clip_id", "mean_diff_sec", "verdict"]].to_string(index=False))
        print("\n판정:", "OK — 자동 라벨 기준 유지" if overall <= THRESHOLD_SEC
              else "기준 다시 맞추기 (02 문서 8장 1단계)")
    print(f"Saved: {folder / 'crosscheck_result.csv'}")

    if args.apply:
        if errors:
            print("\n[STOP] 검수표 오류부터 고치세요 — 반영하지 않음")
            return 1
        fixed, changed = apply_fixes(labels, done)
        fixed.to_csv(args.labels, index=False, encoding="utf-8", float_format="%.3f")
        print(f"\nlabels.csv 반영: {len(changed)}건")
        for c in changed:
            print("  ", c)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
