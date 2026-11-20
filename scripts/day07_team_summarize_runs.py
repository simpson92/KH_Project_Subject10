from __future__ import annotations

# ==================================================
# day07_team_summarize_runs.py — 3단계 C. 설정 비교표 (수업 43번 → 1조 버전)
# ==================================================
# 왜?  Confidence · Window · Buffer를 "한 번에 하나만" 바꿔 돌린 저장영상 결과를
#      한 장의 표로 모아야 어떤 설정이 나은지 비교할 수 있습니다.
#
# 수업 43번과 다른 점
#   - 입력: saved_video/<설정 이름>/<clip_id>.csv (설정마다 폴더 하나)
#   - 출력 2개
#       runtime_runs.csv       : 설정 × clip 한 줄씩 (수업 43번 표와 같은 열 + clip_id · 최종 Stable)
#       runtime_comparison.csv : 설정 한 줄씩 (여러 clip 평균) ← 추천 설정을 고를 때 보는 표
#   - 계산은 수업과 같음: Prediction이 갱신된 Frame(prediction_updated=1)만 사용
#
# 열 뜻
#   warmup_pass          : Prediction이 1번 이상 나온 clip 수 (Warm-up 통과)
#   unknown_ratio        : Prediction 중 Confidence < 기준이라 UNKNOWN이 된 비율
#   stable_label_changes : clip 안에서 Stable Label이 바뀐 횟수 (clip 평균) — 흔들림
#   stable_match_ratio   : Stable = 정답인 Prediction 비율 (clip 평균) — Accuracy 아님 (참고)
#   final_stable_correct : 마지막 Stable = 정답인 clip 수 — Accuracy 아님 (참고)
#
# 입력: reports/team_projects/team01/main/day07/saved_video/*/*.csv
# 출력: reports/team_projects/team01/main/day07/runtime_runs.csv · runtime_comparison.csv  (요약표 = 커밋)
#
# 실행
#   uv run python scripts/day07_team_summarize_runs.py
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_realtime_utils import (
    CONFIG_PATH,
    DAY07_DIR,
    SAVED_VIDEO_DIR,
    label_changes,
    load_json,
    option_name,
    resolve_settings,
)

SETTING_COLUMNS = ["window_seconds", "confidence_threshold", "stabilization_buffer", "stabilization_min_votes"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(SAVED_VIDEO_DIR))
    parser.add_argument("--output-dir", default=str(DAY07_DIR))
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args(argv)


def summarize_log(df: pd.DataFrame, setting: str, clip_id: str) -> dict:
    """Frame 로그 1개 → 1줄 (수업 43번 계산 + 1조 열)."""
    predictions = df[df["prediction_updated"] == 1]
    expected = str(df["expected_label"].fillna("").iloc[0])
    fps = df.loc[df["processing_fps"] > 0, "processing_fps"]
    row = {
        "setting": setting,
        "clip_id": clip_id,
        "expected_label": expected,
        **{column: df[column].iloc[0] for column in SETTING_COLUMNS},
        "frames": len(df),
        "rotated": int(df["rotated"].max()) if "rotated" in df.columns else 0,
        "prediction_updates": len(predictions),
        "unknown_ratio": None,
        "raw_label_changes": 0,
        "stable_label_changes": 0,
        "stable_match_ratio": None,
        "final_stable": "WARMING_UP",
        "final_stable_correct": 0,
        "mean_confidence": None,
        "mean_processing_fps": round(float(fps.mean()), 3) if not fps.empty else 0.0,
    }
    if predictions.empty:
        return row
    final_stable = str(predictions["stable_label"].iloc[-1])
    row.update({
        "unknown_ratio": round(float((predictions["threshold_label"] == "UNKNOWN").mean()), 4),
        "raw_label_changes": label_changes(predictions["raw_label"]),
        "stable_label_changes": label_changes(predictions["stable_label"]),
        "stable_match_ratio": round(float((predictions["stable_label"] == expected).mean()), 4),
        "final_stable": final_stable,
        "final_stable_correct": int(final_stable == expected),
        "mean_confidence": round(float(predictions["confidence"].mean()), 4),
    })
    return row


def compare_settings(runs: pd.DataFrame, prediction_frames: pd.DataFrame) -> pd.DataFrame:
    """설정마다 1줄. unknown_ratio는 Prediction 전체 기준, 나머지는 clip 평균."""
    rows = []
    for setting, group in runs.groupby("setting", sort=False):
        preds = prediction_frames[prediction_frames["setting"] == setting]
        warm = group[group["prediction_updates"] > 0]
        rows.append({
            "setting": setting,
            **{column: group[column].iloc[0] for column in SETTING_COLUMNS},
            "clips": len(group),
            "warmup_pass": len(warm),
            "prediction_updates": int(group["prediction_updates"].sum()),
            "unknown_ratio": round(float((preds["threshold_label"] == "UNKNOWN").mean()), 4) if len(preds) else None,
            "stable_label_changes": round(float(warm["stable_label_changes"].mean()), 3) if len(warm) else None,
            "stable_match_ratio": round(float(warm["stable_match_ratio"].mean()), 4) if len(warm) else None,
            "final_stable_correct": int(group["final_stable_correct"].sum()),
            "mean_confidence": round(float(preds["confidence"].mean()), 4) if len(preds) else None,
            "mean_processing_fps": round(float(group["mean_processing_fps"].mean()), 2),
        })
    table = pd.DataFrame(rows)
    return table.sort_values(SETTING_COLUMNS).reset_index(drop=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_dir = input_dir if input_dir.is_absolute() else ROOT / input_dir
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir

    files = sorted(input_dir.glob("*/*.csv"))
    if not files:
        raise SystemExit(f"[STOP] 저장영상 실행 CSV가 없음: {input_dir} — day07_team_realtime_video.py 먼저")

    runs, frames = [], []
    for path in files:
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        if "prediction_updated" not in df.columns or df.empty:
            continue
        for column in ("prediction_updated", "processing_fps", "confidence", *SETTING_COLUMNS):
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        setting = path.parent.name
        runs.append(summarize_log(df, setting, path.stem))
        prediction_rows = df[df["prediction_updated"] == 1][["threshold_label", "confidence"]].copy()
        prediction_rows["setting"] = setting
        frames.append(prediction_rows)
    if not runs:
        raise SystemExit("[STOP] 요약할 로그가 없음")

    runs_table = pd.DataFrame(runs)
    comparison = compare_settings(runs_table, pd.concat(frames, ignore_index=True))

    # 기본 설정(설정 파일 값) 표시
    baseline = option_name(resolve_settings(load_json(Path(args.config))))
    comparison.insert(1, "baseline", (comparison["setting"] == baseline).astype(int))

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runtime_runs.csv"
    comparison_path = output_dir / "runtime_comparison.csv"
    runs_table.to_csv(runs_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(comparison.drop(columns=SETTING_COLUMNS).to_string(index=False))
    clip_counts = comparison["clips"].unique().tolist()
    print()
    if len(clip_counts) > 1:
        print(f"[주의] 설정마다 clip 수가 다름 {clip_counts} → 같은 clip 묶음(예: --first-takes)으로 맞춰서 비교")
    print("한 번에 하나만 바꿈: ① Confidence 0.35 / 0.45 / 0.55 → ② Window 5 / 3초 → ③ Buffer 3 / 5 / 7")
    print("stable_match_ratio · final_stable_correct는 Accuracy가 아님 (참고 지표)")
    print(f"Saved: {runs_path.relative_to(ROOT) if runs_path.is_relative_to(ROOT) else runs_path}")
    print(f"Saved: {comparison_path.relative_to(ROOT) if comparison_path.is_relative_to(ROOT) else comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
