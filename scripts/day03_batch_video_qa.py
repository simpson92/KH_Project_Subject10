from __future__ import annotations

# ==================================================
# day03_batch_video_qa.py
# 스모크 영상 전체(100개)에 3일차 Pipeline을 한 번에 적용하고
# 영상별 QA 요약표(video_qa.csv)를 만듭니다.
# ==================================================
# 영상 1개마다
#   18번  Frame → 38-D Feature  (D6: 가장 큰 BBox의 사람)
#   17번  1조 U파울 규칙을 Frame마다 판정
#   튐    앞 Frame 대비 |Δ팔꿈치 각도| ≥ 40°
# 를 실행하고, 영상 1개 = 요약 1행으로 정리합니다.
#
# 이 결과로 할 수 있는 것
#   - feature_design.md 9.3 "다음에 틀리는 경우를 찾을 영상"을 100개 전체에서 찾기
#   - 사람(S01~S05)별로 규칙이 맞는지 → 특정 사람에게만 틀리면 Feature 문제
#   - 4일차 sequence_qa의 PASS / REVIEW 후보 미리 만들기
#
# 실행
#   python scripts/day03_batch_video_qa.py                 # data/referee/smoke/raw 전체
#   python scripts/day03_batch_video_qa.py --limit 4       # 먼저 4개만
#   python scripts/day03_batch_video_qa.py --skip-existing # YOLO 다시 안 돌리고 요약만
#
# 결과 (Git 제외 경로)
#   reports/day03/tables/<영상>_frame_features.csv   (18번과 같은 형식)
#   reports/day03/tables/video_qa.csv                (영상별 요약)

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.referee_utils import RAW_DIR, VIDEO_EXTENSIONS, parse_name  # noqa: E402

TABLE_DIR = ROOT / "reports" / "day03" / "tables"


# ==================================================
# [01] 기존 스크립트의 함수 재사용
# ==================================================
# 18 · 17 · 타임라인 코드를 복사하지 않고 그대로 불러옵니다.
# → 판정 기준이 바뀌면 이 스크립트도 자동으로 같은 기준을 씀
def load_script(filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==================================================
# [02] 실행 옵션
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(RAW_DIR))
    parser.add_argument("--kp-conf", type=float, default=0.50)
    parser.add_argument("--main-person", default="area")
    parser.add_argument("--max-wrist-dist", type=float, default=1.0)
    parser.add_argument("--spike-deg", type=float, default=40.0)
    parser.add_argument(
        "--min-run-sec",
        type=float,
        default=0.3,
        help="U파울 True가 이 시간 이상 연속이면 영상 전체를 U파울로 판정",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="앞에서부터 N개만 (0 = 전체)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Frame CSV가 이미 있으면 YOLO를 다시 돌리지 않음",
    )
    parser.add_argument("--output", default="video_qa.csv")
    return parser.parse_args()


# ==================================================
# [03] 영상 1개 → Frame CSV (18번 로직)
# ==================================================
def extract_frames(video: Path, model, config, device, args, s18) -> pd.DataFrame:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    names = s18.make_feature_names()
    rows = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )[0]
        rows.append(
            s18.frame_to_row(
                result,
                frame_id,
                (frame_id - 1) / fps,
                args.kp_conf,
                names,
                args.main_person,
            )
        )
    cap.release()
    return pd.DataFrame(rows)


# ==================================================
# [04] Frame CSV → 영상 요약 1행
# ==================================================
def longest_true_run(flags: list[bool]) -> int:
    """연속으로 True인 가장 긴 구간의 Frame 수."""
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def summarize_video(stem: str, frames: pd.DataFrame, args, s17, timeline) -> dict:
    meta = parse_name(stem) or {"subject": "", "action": "", "take": ""}
    total = len(frames)
    valid = frames[frames["valid_pose"] == 1].reset_index(drop=True)
    n_valid = len(valid)
    fps = (total - 1) / frames["time_sec"].iloc[-1] if total > 1 else 30.0

    row = {
        "video": stem,
        **meta,
        "frames": total,
        "valid_ratio": round(n_valid / total, 3) if total else 0.0,
        "multi_person_frames": int((frames.get("people", 0) > 1).sum())
        if "people" in frames
        else "",
        "mean_kp_conf": round(float(valid["mean_keypoint_conf"].mean()), 3)
        if n_valid
        else "",
        "low_kp_frames": int((valid["low_keypoint_count"] > 0).sum())
        if n_valid
        else "",
    }
    if n_valid == 0:
        row.update(status="REVIEW", reasons="유효 Frame 없음")
        return row

    # --- 어깨 폭 안정성 (D2 근거) ---
    sw = valid["shoulder_width"]
    row["shoulder_min_ratio"] = round(float(sw.min() / sw.median()), 3)
    row["shoulder_max_ratio"] = round(float(sw.max() / sw.median()), 3)

    # --- 손목 − 코 (U파울의 핵심 Feature) ---
    left = valid["left_wrist_y"] - valid["nose_y"]
    right = valid["right_wrist_y"] - valid["nose_y"]
    both_above = (left < 0) & (right < 0)
    one_above = (left < 0) ^ (right < 0)
    # 두 손목 중 "더 낮은 쪽"이 코보다 얼마나 위인가 → 음수일수록 확실한 양손 위
    row["best_both_wrist_minus_nose"] = round(float(np.maximum(left, right).min()), 3)
    row["both_above_frames"] = int(both_above.sum())
    row["one_hand_above_frames"] = int(one_above.sum())

    # --- 17번 1조 규칙 (Frame마다 → 영상 판정) ---
    flags = []
    for _, f in valid.iterrows():
        r = s17.judge_ufoul(
            nose_y=float(f["nose_y"]),
            left_wrist=(float(f["left_wrist_x"]), float(f["left_wrist_y"])),
            right_wrist=(float(f["right_wrist_x"]), float(f["right_wrist_y"])),
            max_wrist_dist=args.max_wrist_dist,
        )
        flags.append(bool(r["ufoul"]))
    run = longest_true_run(flags)
    row["ufoul_true_frames"] = int(sum(flags))
    row["longest_true_run_sec"] = round(run / fps, 2)
    row["rule_pred"] = "ufoul" if run / fps >= args.min_run_sec else "other"
    expected = "ufoul" if meta.get("action") == "ufoul" else "other"
    row["rule_correct"] = row["rule_pred"] == expected

    # --- 팔꿈치 각도 튐 ---
    spikes = 0
    for column in ("left_elbow_angle", "right_elbow_angle"):
        spikes += len(timeline.detect_spikes(valid, column, args.spike_deg))
    row["elbow_spikes"] = spikes

    # --- 상태 판정 (4일차 PASS / REVIEW 후보) ---
    reasons = []
    if row["valid_ratio"] < 0.95:
        reasons.append("유효 Frame < 95%")
    if row["multi_person_frames"]:
        reasons.append("다른 사람 등장")
    if not row["rule_correct"]:
        reasons.append("17번 규칙 오판")
    if row["shoulder_min_ratio"] < 0.7:
        reasons.append("어깨 폭 30% 이상 줄어듦")
    row["status"] = "REVIEW" if reasons else "PASS"
    row["reasons"] = " · ".join(reasons)
    return row


# ==================================================
# [05] 실행
# ==================================================
def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    videos = sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"영상이 없습니다: {source_dir}")

    s18 = load_script("18_video_frame_features.py")
    s17 = load_script("17_referee_rule.py")
    timeline = load_script("day03_plot_feature_timeline.py")

    model = config = device = None
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    for i, video in enumerate(videos, start=1):
        frame_csv = TABLE_DIR / f"{video.stem}_frame_features.csv"
        if args.skip_existing and frame_csv.exists():
            frames = pd.read_csv(frame_csv)
            source = "기존 CSV"
        else:
            if model is None:  # 필요할 때만 모델 로드
                from ultralytics import YOLO

                from src.common import load_config, resolve_device

                config = load_config()
                device = resolve_device(config["device"])
                model = YOLO(config["pose_model"])
            frames = extract_frames(video, model, config, device, args, s18)
            frames.to_csv(frame_csv, index=False, encoding="utf-8-sig")
            source = "YOLO"

        row = summarize_video(video.stem, frames, args, s17, timeline)
        summaries.append(row)
        print(
            f"[{i:3d}/{len(videos)}] {video.stem:22s} {source:6s} "
            f"valid {row['valid_ratio']:.0%}  rule={row.get('rule_pred', '-'):5s} "
            f"{row['status']}  {row['reasons']}"
        )

    qa = pd.DataFrame(summaries)
    output = TABLE_DIR / args.output
    qa.to_csv(output, index=False, encoding="utf-8-sig")

    # 사람 × 동작별 요약: 특정 사람에게만 규칙이 틀리는지
    print()
    if "rule_correct" in qa:
        table = qa.pivot_table(
            index="subject",
            columns="action",
            values="rule_correct",
            aggfunc="mean",
        )
        print("17번 규칙 정답률 (사람 × 동작)")
        print(table.round(2).to_string())
        print()
    n_pass = int((qa["status"] == "PASS").sum())
    print(f"PASS {n_pass} / REVIEW {len(qa) - n_pass}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
