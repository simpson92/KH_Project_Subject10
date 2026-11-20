from __future__ import annotations

# ==================================================
# day07_team_check_inputs.py — 3단계 A. 입력 검사 (수업 41번 → 1조 버전)
# ==================================================
# 왜?  실시간에 연결하기 전에, 6일차 1조 모델 · Feature 목록 · 학습 정보 · 7일차 설정 · S05 영상이
#      서로 맞는지 먼저 확인합니다. 하나라도 어긋나면 실시간 결과 전체가 무의미해지기 때문.
#
# 수업 41번과 다른 점
#   - 모델: artifacts/team01/day06/rf_team_baseline.joblib (경로는 설정 파일에서)
#   - 클래스: A001~A006 고정 대신 rf_training_info.json의 classes (16개, baseline 설정과 대조)
#   - 영상: S02 6개 대신 splits/val.csv의 S05 90개 (S02 = Test는 설정 단계에서 막음)
#   - 세로 확인: 첫 Frame 높이 > 너비 (A5: 1080×1920)
#
# 완료 기준: 클래스 16 · Feature 1,140 · Step 30 · S05 Val 90개 · 모두 OK
#
# 입력: configs/team01_day07_realtime.json · artifacts/team01/day06/ (모델 · json 2개)
#       reports/team_projects/team01/main/dataset/splits/val.csv · data/.../rgb/videos/T01_S05_*.mp4
# 출력: reports/team_projects/team01/main/day07/tables/day07_team_input_check.csv  (Git 제외)
#
# 실행
#   uv run python scripts/day07_team_check_inputs.py
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_realtime_utils import (
    CONFIG_PATH,
    DAY07_DIR,
    load_model_bundle,
    load_runtime_config,
    val_clips,
)

OUTPUT_PATH = DAY07_DIR / "tables" / "day07_team_input_check.csv"
PROJECT_CONFIG_PATH = ROOT / "configs" / "project.json"
EXPECTED_VAL_CLIPS = 90


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args(argv)


def read_video_info(path: Path) -> dict:
    """FPS · Frame 수 · 길이 · 첫 Frame 크기 (수업 41번 + 세로 확인)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"open_ok": 0, "fps": 0.0, "frames": 0, "duration_sec": 0.0, "width": 0, "height": 0}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = capture.read()
    capture.release()
    if not success or frame is None or fps <= 0 or frames <= 0:
        return {"open_ok": 0, "fps": fps, "frames": frames, "duration_sec": 0.0, "width": 0, "height": 0}
    height, width = frame.shape[:2]
    return {
        "open_ok": 1,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_sec": round(frames / fps, 3),
        "width": int(width),
        "height": int(height),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_runtime_config(Path(args.config))           # 설정 검사 + S02 잠금
    if not PROJECT_CONFIG_PATH.exists():
        raise FileNotFoundError(PROJECT_CONFIG_PATH)

    bundle = load_model_bundle(config)                         # 모델 · Feature · 학습 정보 · 16클래스 검사
    steps = bundle["sequence_steps"]
    feature_count = len(bundle["feature_columns"])

    clips = val_clips(config)
    problems = []
    if len(clips) != EXPECTED_VAL_CLIPS:
        problems.append(f"{config['video_subject']} Val clip {len(clips)}개 (기대 {EXPECTED_VAL_CLIPS})")

    required_sec = float(config["window_seconds"]) * float(config["min_window_fill_ratio"])
    rows = []
    for _, clip in clips.iterrows():
        path = Path(clip["video_abspath"])
        info = read_video_info(path) if path.exists() else {
            "open_ok": 0, "fps": 0.0, "frames": 0, "duration_sec": 0.0, "width": 0, "height": 0}
        portrait = int(info["height"] > info["width"]) if info["open_ok"] else 0
        rows.append({
            "clip_id": clip["clip_id"],
            "action_id": clip["action_id"],
            "train_label": clip["train_label"],
            "video_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "exists": int(path.exists()),
            **info,
            "portrait": portrait,
            "required_warmup_sec": round(required_sec, 3),
            "runtime_ready": int(info["duration_sec"] >= required_sec),
        })
    table = pd.DataFrame(rows)

    missing = table[table["exists"] == 0]["clip_id"].tolist()
    unreadable = table[(table["exists"] == 1) & (table["open_ok"] == 0)]["clip_id"].tolist()
    landscape = table[(table["open_ok"] == 1) & (table["portrait"] == 0)]["clip_id"].tolist()
    if missing:
        problems.append(f"영상 없음 {len(missing)}개 (예: {missing[:3]}) — 드라이브 영상 폴더에서 받기")
    if unreadable:
        problems.append(f"영상 열기 실패 {len(unreadable)}개 (예: {unreadable[:3]})")
    if landscape and config["expect_portrait"]:
        problems.append(f"세로가 아닌 영상 {len(landscape)}개 (예: {landscape[:3]})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    ready = int(table["runtime_ready"].sum())
    readable = table[table["open_ok"] == 1]
    print(f"Model          : {bundle['paths']['model'].relative_to(ROOT)}")
    print(f"Sequence steps : {steps}")
    print(f"Feature count  : {feature_count:,}  ({steps} × 38)")
    print(f"Action classes : {len(bundle['classes'])}  {bundle['classes']}")
    print(f"Runtime config : window {config['window_seconds']}s · confidence {config['confidence_threshold']}"
          f" · buffer {config['stabilization_buffer']}/{config['stabilization_min_votes']}")
    print(f"{config['video_subject']} Val videos  : {len(table) - len(missing)} / {len(table)}"
          f"  (세로 {int(readable['portrait'].sum())} / {len(readable)})")
    if len(readable):
        print(f"Video size     : {sorted(set(zip(readable['width'], readable['height'])))}"
              f" · FPS {sorted(readable['fps'].unique().tolist())}")
        print(f"Duration       : {readable['duration_sec'].min():.1f} ~ {readable['duration_sec'].max():.1f}s"
              f" (중앙 {readable['duration_sec'].median():.1f}s)")
    print(f"Runtime-ready  : {ready} / {len(table)}  (영상 길이 ≥ Warm-up {required_sec:.1f}s)")
    if ready < len(table) and not missing:
        print("  → Warm-up보다 짧은 영상은 Prediction이 안 나올 수 있음 (오류 아님). 아래 CSV의 runtime_ready=0 확인")
    print()
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")

    if problems:
        for problem in problems:
            print("[ERROR]", problem)
        print("Day07 team input data: NOT READY")
        return 1
    print("Day07 team input data: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
