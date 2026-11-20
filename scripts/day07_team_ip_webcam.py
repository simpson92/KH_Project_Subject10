from __future__ import annotations

# ==================================================
# day07_team_ip_webcam.py — 3단계 D. IP Webcam 실시간 (수업 44번 → 1조 버전)
# ==================================================
# 왜?  저장영상(B)에서 확인한 Pipeline의 "입력만" 스마트폰 IP Webcam으로 바꿔,
#      끝없이 들어오는 Frame에서도 6일차 1조 모델이 학습 때와 같은 조건으로 동작하는지 봅니다.
#      IP Webcam 영상은 새 학습 데이터가 아님 → 결과를 보고 모델을 다시 학습하지 않음 (교재 7장).
#
# 수업 44번과 다른 점
#   - 클래스 16개 → 숫자키 1~6에 다 못 넣으므로 --targets로 6개씩 (3번 나눠 실행)
#       예: --targets A01,A02,A03,A04,A05,A06 → 1=A01 … 6=A06
#       D01~D03(일상 동작)은 정답 N00으로 기록, X = 학습 안 한 동작 (정답 없음)
#   - 세로: 첫 Frame의 높이 > 너비를 먼저 확인. 가로면 회전하고 로그에 rotated=1
#           (가로 그대로 들어오면 결과 전부 무효 — 가이드 4쪽)
#   - --protocol with_c01 / no_c01: 학습 clip 순서(N00 → C01 → 신호 → N00)대로 했는지 기록
#   - 실행 조건(URL · 해상도 · 설정 · Target)을 같은 이름의 _meta.json에 저장
#   - 화면: 1920 높이는 모니터에 안 들어가서 표시만 축소 (YOLO에는 원본)
#
# 키: 1~6 Target · 0 Target 해제 · R Window 리셋 · Q 종료 (카메라 창을 클릭해야 키가 들어감)
#
# 출력: reports/team_projects/team01/main/day07/ip_webcam/ip_webcam_<설정>_<protocol>_<시각>.csv · _meta.json
#       (Frame 로그 = Git 제외)
#
# 실행
#   uv run python scripts/day07_team_ip_webcam.py --url "http://<폰 IP>:8080/video" --targets A01,A02,A03,A04,A05,A06
#   uv run python scripts/day07_team_ip_webcam.py --url ... --targets A07,A08,A09,A10,A11,A12
#   uv run python scripts/day07_team_ip_webcam.py --url ... --targets A13,A14,A15,D01,D02,D03
#   uv run python scripts/day07_team_ip_webcam.py --url ... --targets A04,A07,D03,X --protocol no_c01
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime_utils import (
    FPSMeter,
    PredictionStabilizer,
    TimedFeatureBuffer,
    frame_feature_from_pose,
    predict_from_window,
    validate_runtime_values,
)
from src.team_realtime_utils import (
    CONFIG_PATH,
    IP_WEBCAM_DIR,
    display_label,
    english_names,
    ensure_portrait,
    fit_height,
    load_json,
    load_model_bundle,
    load_pose_backend,
    load_runtime_config,
    option_name,
    parse_targets,
    resolve_settings,
)

PROJECT_CONFIG_PATH = ROOT / "configs" / "project.json"
WINDOW_NAME = "Team01 IP Webcam Referee Signal"
PANEL_WIDTH = 460


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="예: http://192.168.0.10:8080/video")
    parser.add_argument("--targets", required=True, help="숫자키 1~6에 놓을 동작 (최대 6개, 예: A01,A02,A03,A04,A05,A06)")
    parser.add_argument("--protocol", choices=["with_c01", "no_c01"], default="with_c01",
                        help="with_c01: N00 → C01 → 신호 → N00 (학습과 같은 순서) · no_c01: 신호만")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--confidence-threshold", "--confidence", dest="confidence", type=float, default=None)
    parser.add_argument("--stabilization-buffer", type=int, default=None)
    parser.add_argument("--keep-buffer-on-target", action="store_true",
                        help="Target을 바꿔도 Sliding Window 유지 (연속 동작 전환 관찰용)")
    parser.add_argument("--display-height", type=int, default=None, help="화면 표시 높이 (기본: 설정 파일)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(IP_WEBCAM_DIR))
    return parser.parse_args(argv)


def read_frame_with_retry(capture, max_retries: int = 10, retry_delay_sec: float = 0.05):
    """MJPEG Stream의 일시적인 read 실패는 짧게 재시도 (수업 44번 그대로)."""
    for attempt in range(1, max_retries + 1):
        success, frame = capture.read()
        if success and frame is not None:
            return True, frame
        if attempt < max_retries:
            time.sleep(retry_delay_sec)
    return False, None


def draw_panel(display, lines: list[tuple[str, tuple]], controls: list[str], panel_x: int) -> None:
    cv2.putText(display, "RUNTIME STATUS", (panel_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y = 74
    for text, color in lines:
        cv2.putText(display, text, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 30
    y += 8
    cv2.line(display, (panel_x - 10, y), (panel_x + PANEL_WIDTH - 30, y), (120, 120, 120), 1)
    y += 26
    for text in controls:
        cv2.putText(display, text, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += 24


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
    bundle = load_model_bundle(config)
    targets = parse_targets(args.targets, bundle["classes"])
    key_to_target = {ord(item["key"]): item for item in targets}
    names = english_names()
    expect_portrait = bool(config["expect_portrait"])
    rotation = config.get("landscape_rotation", "clockwise")
    display_height = int(args.display_height or config.get("display_height", 900))
    keep_buffer = bool(args.keep_buffer_on_target)

    pose_predict = load_pose_backend(load_json(PROJECT_CONFIG_PATH))
    capture = cv2.VideoCapture(args.url)
    if not capture.isOpened():
        raise RuntimeError("IP Webcam Stream을 열 수 없음 — 가이드 2쪽 ①~⑥ 순서로 확인 (같은 Wi-Fi · 서버 시작 · /video)")
    success, frame = read_frame_with_retry(capture)
    if not success:
        capture.release()
        raise RuntimeError("IP Webcam 첫 Frame을 읽지 못함")

    # 세로 확인 (가장 먼저)
    raw_height, raw_width = frame.shape[:2]
    frame, rotated = ensure_portrait(frame, expect_portrait, rotation)
    height, width = frame.shape[:2]
    orientation = "PORTRAIT" if raw_height > raw_width else "LANDSCAPE"
    print(f"Connected: {raw_width} x {raw_height} ({orientation})"
          + (f" → {rotation} 회전 → {width} x {height}" if rotated else ""))
    if expect_portrait and height <= width:
        print("[WARNING] 세로가 아님 — IP Webcam 앱에서 「세로 보기」를 켜세요. 가로 결과는 무효 (A5)")
    elif rotated:
        print("[INFO] 가로 Stream을 회전해서 씀. 사람이 거꾸로 보이면 설정 landscape_rotation을 counterclockwise로")

    print()
    for item in targets:
        extra = f" (정답 {item['expected_label']})" if item["expected_label"] != item["target_action"] else ""
        print(f"{item['key']} {display_label(item['target_action'], names)}{extra}")
    print("0 target clear · R buffer reset · Q quit")
    print(f"Protocol: {args.protocol} · Setting: {option_name(settings)} · Target mode: {'KEEP' if keep_buffer else 'RESET'}")

    preview = fit_height(frame, display_height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, preview.shape[1] + PANEL_WIDTH, max(preview.shape[0], 560))

    feature_buffer = TimedFeatureBuffer(settings["window_seconds"])
    stabilizer = PredictionStabilizer(buffer_size=settings["stabilization_buffer"],
                                      min_votes=settings["stabilization_min_votes"])
    fps_meter = FPSMeter()

    def reset_runtime():
        feature_buffer.clear()
        stabilizer.clear()
        return -1e9, "WARMING_UP", "WARMING_UP", "WARMING_UP", 0.0

    last_prediction_time, raw_label, threshold_label, stable_label, confidence = reset_runtime()
    target = {"target_action": "", "expected_label": ""}
    session_start = time.perf_counter()
    started_at = datetime.now()
    log_rows = []
    frame_id = 0
    segment_id = 0

    while True:
        elapsed = time.perf_counter() - session_start
        result = pose_predict(frame)
        feature = frame_feature_from_pose(result)
        feature_buffer.append(elapsed, feature)
        ready = feature_buffer.is_ready(
            min_window_fill_ratio=settings["min_window_fill_ratio"],
            min_valid_feature_ratio=settings["min_valid_feature_ratio"],
            min_valid_features=settings["min_valid_features"],
        )
        prediction_updated = 0
        if ready and (elapsed - last_prediction_time) >= settings["prediction_interval_sec"]:
            threshold_label, raw_label, confidence, _ = predict_from_window(
                model=bundle["model"],
                feature_buffer=feature_buffer,
                sequence_steps=bundle["sequence_steps"],
                confidence_threshold=settings["confidence_threshold"],
            )
            stable_label = stabilizer.update(threshold_label)
            last_prediction_time = elapsed
            prediction_updated = 1
        elif not ready:
            raw_label = threshold_label = stable_label = "WARMING_UP"
            confidence = 0.0
        processing_fps = fps_meter.update(time.perf_counter())

        # 화면: 축소한 카메라 + 오른쪽 상태 패널
        annotated = fit_height(result.plot(), display_height)
        display = cv2.copyMakeBorder(annotated, 0, max(560 - annotated.shape[0], 0), 0, PANEL_WIDTH,
                                     cv2.BORDER_CONSTANT, value=(25, 25, 25))
        match = (stable_label == target["expected_label"]) and bool(target["expected_label"])
        stable_color = (80, 220, 80) if match else (255, 255, 255)
        target_text = display_label(target["target_action"], names) if target["target_action"] else "-"
        draw_panel(display, [
            (f"Target: {target_text}", (0, 220, 255)),
            (f"Raw: {display_label(raw_label, names)}", (255, 255, 255)),
            (f"Threshold: {display_label(threshold_label, names)}", (255, 255, 255)),
            (f"Stable: {display_label(stable_label, names)}", stable_color),
            (f"Confidence: {confidence:.2f} (>= {settings['confidence_threshold']:g})", (255, 255, 255)),
            (f"Window: {feature_buffer.duration():.1f} / {settings['window_seconds']:g}s", (255, 255, 255)),
            (f"FPS: {processing_fps:.1f}   Frame: {width}x{height}{' R' if rotated else ''}", (255, 255, 255)),
            (f"Protocol: {args.protocol}   Mode: {'KEEP' if keep_buffer else 'RESET'}", (200, 200, 200)),
        ], [f"{item['key']}: {item['target_action']}" for item in targets] + ["0: clear  R: reset  Q: quit"],
            annotated.shape[1] + 20)
        cv2.imshow(WINDOW_NAME, display)

        log_rows.append({
            "frame_id": frame_id,
            "elapsed_sec": round(elapsed, 4),
            "target_action": target["target_action"],
            "expected_label": target["expected_label"],
            "segment_id": segment_id,
            "protocol": args.protocol,
            "frame_width": width,
            "frame_height": height,
            "rotated": int(rotated),
            "feature_ok": int(feature is not None),
            "window_seconds": settings["window_seconds"],
            "confidence_threshold": settings["confidence_threshold"],
            "stabilization_buffer": settings["stabilization_buffer"],
            "stabilization_min_votes": settings["stabilization_min_votes"],
            "keep_buffer_on_target": int(keep_buffer),
            "buffer_duration_sec": round(feature_buffer.duration(), 4),
            "valid_feature_ratio": round(feature_buffer.valid_ratio(), 4),
            "raw_label": raw_label,
            "threshold_label": threshold_label,
            "stable_label": stable_label,
            "confidence": round(confidence, 4),
            "prediction_updated": prediction_updated,
            "processing_fps": round(processing_fps, 3),
        })

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in key_to_target:
            target = key_to_target[key]
            segment_id += 1
            if not keep_buffer:
                last_prediction_time, raw_label, threshold_label, stable_label, confidence = reset_runtime()
            print(f"Target: {target['target_action']} (정답 {target['expected_label']}) | segment {segment_id}")
        elif key == ord("0"):
            target = {"target_action": "", "expected_label": ""}
            segment_id += 1
            last_prediction_time, raw_label, threshold_label, stable_label, confidence = reset_runtime()
            print("Target cleared")
        elif key in (ord("r"), ord("R")):
            last_prediction_time, raw_label, threshold_label, stable_label, confidence = reset_runtime()
            if target["target_action"]:
                segment_id += 1
            print("Buffers reset")

        success, frame = read_frame_with_retry(capture)
        if not success:
            print("IP Webcam Frame을 여러 번 재시도했지만 읽지 못함 → 저장하고 종료")
            break
        frame, rotated = ensure_portrait(frame, expect_portrait, rotation)
        height, width = frame.shape[:2]
        frame_id += 1

    capture.release()
    cv2.destroyAllWindows()
    if not log_rows:
        raise RuntimeError("저장할 Runtime 로그가 없음")

    output_dir = Path(args.output_dir)
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ip_webcam_{option_name(settings)}_{args.protocol}_{started_at.strftime('%Y%m%d_%H%M%S')}"
    output_log = output_dir / f"{stem}.csv"
    with output_log.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    fps_values = [row["processing_fps"] for row in log_rows if row["processing_fps"] > 0]
    mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    parsed = urlparse(args.url)
    meta = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "duration_sec": round(log_rows[-1]["elapsed_sec"], 1),
        "url_host": parsed.hostname,
        "url_port": parsed.port,
        "url_path": parsed.path,
        "first_frame_raw": f"{raw_width}x{raw_height}",
        "orientation": orientation,
        "rotated": bool(rotated),
        "frames": len(log_rows),
        "mean_processing_fps": round(mean_fps, 2),
        "protocol": args.protocol,
        "targets": targets,
        "settings": settings,
        "keep_buffer_on_target": keep_buffer,
        "model": str(bundle["paths"]["model"].relative_to(ROOT)),
        "classes": bundle["classes"],
        "sequence_steps": bundle["sequence_steps"],
    }
    (output_dir / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Frames: {len(log_rows)} · Mean processing FPS: {mean_fps:.2f}")
    print(f"Saved: {output_log}")
    print(f"Meta : {output_dir / (stem + '_meta.json')}")
    print("다음: uv run python scripts/day07_team_analyze_log.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
