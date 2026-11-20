from __future__ import annotations

# ==================================================
# day07_team_realtime_video.py — 3단계 B. 저장영상을 Stream처럼 (수업 42번 → 1조 버전)
# ==================================================
# 왜?  IP Webcam에 바로 연결하면 "카메라 문제인지 코드 문제인지" 구분이 안 됩니다.
#      먼저 S05 Val 저장영상을 한 Frame씩 읽어 실시간과 똑같은 Pipeline을 통과시켜 봅니다.
#        Frame → YOLO Pose → 38-D Feature → Sliding Window(최근 N초) → 30 Step → 1조 RF
#              → Confidence 기준(낮으면 UNKNOWN) → 최근 Prediction 다수결(Stable)
#      계산은 수업 src/realtime_utils.py 그대로 (6일차 학습과 같은 Feature 정의).
#
# 수업 42번과 다른 점
#   - 모델 · 클래스: 1조 16클래스 (rf_training_info.json에서 읽음)
#   - 영상: --clip-id로 S05 Val clip을 고름 (정답 = splits/val.csv의 train_label, D01~D03 → N00)
#           S02(Test) · 다른 Subject clip은 실행 거부
#   - 세로: 첫 Frame이 가로면 회전 (landscape_rotation)
#   - 대표 인물: 가장 큰 BBox (D6 — 학습 데이터와 같은 기준, team_frame_feature_from_pose)
#   - 여러 clip을 한 번에: --first-takes (동작마다 Warm-up보다 긴 첫 Take = 18개), --all (90개)
#     S05 D03 T01(3.8초)은 Warm-up 4초보다 짧아 Prediction 0번 → D03 T02로 바꿔 씀 (바꾼 내역 출력)
#   - 결과: reports/team_projects/team01/main/day07/saved_video/<설정>/<clip_id>.csv · .mp4 (Git 제외)
#
# 실행
#   uv run python scripts/day07_team_realtime_video.py --clip-id T01_S05_A06_T01
#   uv run python scripts/day07_team_realtime_video.py --clip-id T01_S05_A06_T01 --confidence-threshold 0.35
#   uv run python scripts/day07_team_realtime_video.py --first-takes --no-video          # 18개 · MP4 생략
#   uv run python scripts/day07_team_realtime_video.py --first-takes --window-seconds 3  # 설정 비교 (C)
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime_utils import (
    FPSMeter,
    PredictionStabilizer,
    TimedFeatureBuffer,
    predict_from_window,
    validate_runtime_values,
)
from src.team_realtime_utils import (
    CONFIG_PATH,
    SAVED_VIDEO_DIR,
    display_label,
    english_names,
    ensure_portrait,
    load_json,
    load_model_bundle,
    load_pose_backend,
    load_runtime_config,
    option_name,
    resolve_settings,
    select_first_take,
    team_frame_feature_from_pose,
    val_clips,
    warmup_seconds,
)

PROJECT_CONFIG_PATH = ROOT / "configs" / "project.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    pick = parser.add_mutually_exclusive_group(required=True)
    pick.add_argument("--clip-id", action="append", help="S05 Val clip (여러 번 줄 수 있음)")
    pick.add_argument("--first-takes", action="store_true", help="동작마다 Warm-up보다 긴 첫 Take 1개 = 18개")
    pick.add_argument("--all", action="store_true", help="S05 Val 90개 전부")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--confidence-threshold", "--confidence", dest="confidence", type=float, default=None)
    parser.add_argument("--stabilization-buffer", type=int, default=None)
    parser.add_argument("--no-video", action="store_true", help="결과 MP4를 만들지 않음 (빠름)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(SAVED_VIDEO_DIR))
    return parser.parse_args(argv)


def pick_clips(args, config: dict) -> pd.DataFrame:
    clips = val_clips(config)
    if args.first_takes:
        # 설정을 바꿔 비교해도 clip 묶음이 같도록 기본 설정의 Warm-up(5초 × 0.8)으로 고름
        chosen, replaced = select_first_take(clips, warmup_seconds(config))
        for item in replaced:
            print(f"[Take 변경] {item['action_id']}: {item['from']} ({item['from_sec']:.2f}s < Warm-up "
                  f"{warmup_seconds(config):.1f}s) → {item['to']} ({item['to_sec']:.2f}s)")
        return chosen
    if args.all:
        return clips
    unknown = [clip_id for clip_id in args.clip_id if clip_id not in set(clips["clip_id"])]
    if unknown:
        raise SystemExit(
            f"[STOP] {config['video_subject']} Val clip이 아님: {unknown}\n"
            f"       설정 조정은 Val({config['video_subject']})로만 합니다. S02(Test)는 마지막에 1번 (D7)."
        )
    return clips[clips["clip_id"].isin(args.clip_id)].reset_index(drop=True)


def draw_status(frame, lines: list[str]) -> None:
    """세로 1080 폭에 맞춘 상태 표시 (왼쪽 위 검은 상자)."""
    scale = max(frame.shape[1] / 1080, 0.4)
    line_gap = int(46 * scale)
    box_height = int(24 * scale) + line_gap * len(lines)
    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 10 + box_height), (0, 0, 0), -1)
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (24, int(10 + line_gap * (index + 1))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (255, 255, 255), max(int(2 * scale), 1), cv2.LINE_AA)


def run_clip(clip: dict, settings: dict, bundle: dict, pose_predict, output_dir: Path,
             write_video: bool, expect_portrait: bool, rotation: str, names: dict,
             main_person: str = "area") -> dict:
    """Clip 1개를 Stream처럼 처리하고 Frame 로그 CSV(+ MP4)를 저장. 요약 dict를 돌려줌."""
    source = Path(clip["video_abspath"])
    if not source.exists():
        raise FileNotFoundError(f"{source} — 드라이브 영상 폴더에서 받기 (가이드 0단계 표)")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_log = output_dir / f"{clip['clip_id']}.csv"
    output_video = output_dir / f"{clip['clip_id']}.mp4"

    feature_buffer = TimedFeatureBuffer(settings["window_seconds"])
    stabilizer = PredictionStabilizer(buffer_size=settings["stabilization_buffer"],
                                      min_votes=settings["stabilization_min_votes"])
    fps_meter = FPSMeter()
    expected_label = str(clip["train_label"])
    last_prediction_time = -1e9
    raw_label = threshold_label = stable_label = "WARMING_UP"
    confidence = 0.0
    writer = None
    log_rows = []
    frame_id = 0
    process_start = time.perf_counter()

    while True:
        success, frame = capture.read()
        if not success:
            break
        frame, rotated = ensure_portrait(frame, expect_portrait, rotation)
        height, width = frame.shape[:2]
        video_time = frame_id / source_fps                # 저장영상은 영상 시간 기준 (수업 42번과 같음)
        loop_start = time.perf_counter()

        result = pose_predict(frame)
        feature = team_frame_feature_from_pose(result, main_person)   # 38-D 또는 None (D6: 가장 큰 BBox)
        feature_buffer.append(video_time, feature)
        ready = feature_buffer.is_ready(
            min_window_fill_ratio=settings["min_window_fill_ratio"],
            min_valid_feature_ratio=settings["min_valid_feature_ratio"],
            min_valid_features=settings["min_valid_features"],
        )
        prediction_updated = 0
        if ready and (video_time - last_prediction_time) >= settings["prediction_interval_sec"]:
            threshold_label, raw_label, confidence, _ = predict_from_window(
                model=bundle["model"],
                feature_buffer=feature_buffer,
                sequence_steps=bundle["sequence_steps"],
                confidence_threshold=settings["confidence_threshold"],
            )
            stable_label = stabilizer.update(threshold_label)
            last_prediction_time = video_time
            prediction_updated = 1
        elif not ready:
            raw_label = threshold_label = stable_label = "WARMING_UP"
            confidence = 0.0
        processing_fps = fps_meter.update(time.perf_counter())

        if write_video:
            annotated = result.plot() if hasattr(result, "plot") else frame.copy()
            draw_status(annotated, [
                f"Expected: {display_label(expected_label, names)}  ({clip['action_id']})",
                f"Raw: {display_label(raw_label, names)}",
                f"Threshold: {display_label(threshold_label, names)}",
                f"Stable: {display_label(stable_label, names)}",
                f"Conf {confidence:.2f}  Window {settings['window_seconds']:g}s  FPS {processing_fps:.1f}",
            ])
            if writer is None:
                writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                         source_fps, (annotated.shape[1], annotated.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"결과 영상을 저장할 수 없음: {output_video}")
            writer.write(annotated)

        log_rows.append({
            "clip_id": clip["clip_id"],
            "action_id": clip["action_id"],
            "frame_id": frame_id,
            "video_time_sec": round(video_time, 4),
            "expected_label": expected_label,
            "frame_width": width,
            "frame_height": height,
            "rotated": int(rotated),
            "feature_ok": int(feature is not None),
            "window_seconds": settings["window_seconds"],
            "confidence_threshold": settings["confidence_threshold"],
            "stabilization_buffer": settings["stabilization_buffer"],
            "stabilization_min_votes": settings["stabilization_min_votes"],
            "buffer_duration_sec": round(feature_buffer.duration(), 4),
            "valid_feature_ratio": round(feature_buffer.valid_ratio(), 4),
            "raw_label": raw_label,
            "threshold_label": threshold_label,
            "stable_label": stable_label,
            "confidence": round(confidence, 4),
            "prediction_updated": prediction_updated,
            "processing_fps": round(processing_fps, 3),
            "frame_processing_sec": round(time.perf_counter() - loop_start, 5),
        })
        frame_id += 1

    capture.release()
    if writer is not None:
        writer.release()
    if not log_rows:
        raise RuntimeError(f"처리한 Frame이 없음: {source}")

    with output_log.open("w", newline="", encoding="utf-8-sig") as file:
        writer_csv = csv.DictWriter(file, fieldnames=list(log_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(log_rows)

    predictions = [row for row in log_rows if row["prediction_updated"] == 1]
    elapsed = time.perf_counter() - process_start
    return {
        "clip_id": clip["clip_id"],
        "expected_label": expected_label,
        "frames": frame_id,
        "duration_sec": round(frame_id / source_fps, 2),
        "frame_size": f"{log_rows[0]['frame_width']}x{log_rows[0]['frame_height']}",
        "rotated": log_rows[0]["rotated"],
        "prediction_updates": len(predictions),
        "final_stable": predictions[-1]["stable_label"] if predictions else "WARMING_UP",
        "final_raw": predictions[-1]["raw_label"] if predictions else "WARMING_UP",
        "final_confidence": predictions[-1]["confidence"] if predictions else 0.0,
        "overall_fps": round(frame_id / elapsed, 2) if elapsed > 0 else 0.0,
        "log": output_log,
        "video": output_video if write_video else None,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_runtime_config(Path(args.config))
    settings = resolve_settings(config, args.window_seconds, args.confidence, args.stabilization_buffer)
    validate_runtime_values(
        window_seconds=settings["window_seconds"],
        confidence_threshold=settings["confidence_threshold"],
        stabilization_buffer=settings["stabilization_buffer"],
        stabilization_min_votes=settings["stabilization_min_votes"],
    )
    clips = pick_clips(args, config)
    bundle = load_model_bundle(config)
    pose_predict = load_pose_backend(load_json(PROJECT_CONFIG_PATH))
    names = english_names()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    run_dir = output_dir / option_name(settings)

    print(f"Setting: {option_name(settings)} (min votes {settings['stabilization_min_votes']}) · {len(clips)} clips"
          f" · 대표 인물 {config.get('main_person', 'area')}")
    summaries = []
    for index, clip in enumerate(clips.to_dict("records"), start=1):
        summary = run_clip(clip, settings, bundle, pose_predict, run_dir, not args.no_video,
                           bool(config["expect_portrait"]), config.get("landscape_rotation", "clockwise"), names,
                           config.get("main_person", "area"))
        summaries.append(summary)
        mark = "✓" if summary["final_stable"] == summary["expected_label"] else " "
        print(f"[{index:2d}/{len(clips)}] {summary['clip_id']:17s} {summary['duration_sec']:4.1f}s "
              f"{summary['frame_size']}{' (rotated)' if summary['rotated'] else ''} · "
              f"predictions {summary['prediction_updates']:3d} · expected {summary['expected_label']} → "
              f"stable {summary['final_stable']:10s} {mark} (conf {summary['final_confidence']:.2f}) · "
              f"FPS {summary['overall_fps']:.1f}")

    table = pd.DataFrame(summaries)
    warm = int((table["prediction_updates"] > 0).sum())
    print()
    print(f"Warm-up 통과 (Prediction ≥ 1번): {warm} / {len(table)}")
    if warm < len(table):
        short = table[table["prediction_updates"] == 0]["clip_id"].tolist()
        print(f"  Prediction 없음: {short} → 영상이 Warm-up({settings['window_seconds'] * settings['min_window_fill_ratio']:.1f}s)"
              " 보다 짧거나 사람이 잘 안 잡힘. --window-seconds 3으로도 비교")
    print(f"마지막 Stable = 정답: {int((table['final_stable'] == table['expected_label']).sum())} / {len(table)}"
          "  (참고 — Accuracy 아님)")
    print(f"평균 Processing FPS: {table['overall_fps'].mean():.1f}")
    print(f"Saved: {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}/  (CSV{' · MP4' if not args.no_video else ''})")
    print("다음: uv run python scripts/day07_team_summarize_runs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
