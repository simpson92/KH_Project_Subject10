from __future__ import annotations

# ==================================================
# day04_build_team_sequences.py
# 1조 영상 → clip_id별 Sequence CSV — 4일차 팀 확장 STEP 4
# ==================================================
# 교재 22_build_action_sequences.py의 계산(Frame → YOLO Pose → 정규화 → 각도 → 38-D)을
# 그대로 쓰고, 팀 확장 문서 STEP 4에서 요구한 부분만 바꿉니다. (22번 원본은 그대로)
#
# 22번과 달라지는 점
#   1. 저장 이름  : {action_label}_sequence.csv → {clip_id}_sequence.csv (덮어쓰기 방지)
#   2. 입력 목록  : action_classes.csv → configs/team01_clip_manifest.csv
#   3. 대표 인물  : BBox Confidence 최고 → 가장 큰 BBox (1조 D6, --main-person으로 변경 가능)
#   4. 추가 기록  : fps · 화면 크기 · 사람 수 · BBox · 실패 이유 ·
#                   정규화 기준 관절(어깨·골반) 저신뢰 개수 · 원본 Keypoint x·y·conf (17 × 3)
#   5. Confidence가 없으면 1.0으로 채우지 않고 빈칸으로 둡니다 (STEP 4 주의 문구)
#
# 22번과 같은 점
#   - 38-D 열 이름·순서 (nose_x … right_ankle_y, 각도 4개) → 교재 23·24번이 이 CSV를 그대로 읽음
#   - feature_ok 기준 (정규화 성공 + 38개 + NaN/Inf 없음). 저신뢰 관절은 표시만 하고 걸러내지 않음
#   - 실패한 Frame도 지우지 않고 Row로 남김
#
# 실행 순서 (팀 확장 문서 §6)
#   python scripts/day04_build_team_sequences.py --clip-id T01_S01_A03_T01                 # ① 1개
#   python scripts/day04_build_team_sequences.py --clip-id T01_S01_A03_T01 T01_S01_A05_T01 # ② 두 동작
#   python scripts/day04_build_team_sequences.py --subject-id S01 --action-id A03          # ③ 한 동작 10회
#   python scripts/day04_build_team_sequences.py                                           # ④ 100개 전체
#   이미 만든 CSV는 건너뜁니다. 다시 만들려면 --overwrite
#
# 출력
#   reports/team_projects/team01/sequences/{clip_id}_sequence.csv  (Git 제외)
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_utils import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    normalize_by_shoulder_width,
    selected_joint_angles,
)
from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
    MAIN_PERSON_MODES,
    select_main_person,
)

MANIFEST_CSV = ROOT / "configs" / "team01_clip_manifest.csv"
VIDEO_DIR = ROOT / "data" / "team_projects" / "team01" / "rgb" / "videos"
OUTPUT_DIR = ROOT / "reports" / "team_projects" / "team01" / "sequences"

# 정규화 기준이 되는 관절 (Hip Center · Shoulder Width 계산에 쓰임)
NORM_JOINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
ANGLE_NAMES = (
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
)

META_COLUMNS = [
    "clip_id",
    "subject_id",
    "action_id",
    "action_label",
    "take_id",
    "source_type",
    "source",
    "frame_id",
    "time_sec",
    "fps",
    "image_width",
    "image_height",
    "people",
    "person_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_conf",
    "pose_ok",
    "feature_ok",
    "fail_reason",
    "mean_keypoint_conf",
    "low_keypoint_count",
    "norm_joint_low_count",
    "hip_center_x",
    "hip_center_y",
    "shoulder_width",
]


# ==================================================
# [01] 열 이름 (38-D는 22번과 같은 이름·순서)
# ==================================================
def feature_names() -> list[str]:
    names = []
    for name in COCO_KEYPOINT_NAMES:
        names += [f"{name}_x", f"{name}_y"]
    return names + list(ANGLE_NAMES)


def raw_names() -> list[str]:
    names = []
    for name in COCO_KEYPOINT_NAMES:
        names += [f"raw_{name}_x", f"raw_{name}_y", f"conf_{name}"]
    return names


FEATURE_COLUMNS = feature_names()
RAW_COLUMNS = raw_names()
ALL_COLUMNS = META_COLUMNS + FEATURE_COLUMNS + RAW_COLUMNS


# ==================================================
# [02] 실행 옵션
# ==================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--video-dir", default=str(VIDEO_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--clip-id", nargs="+", default=None, help="이 clip만 처리")
    parser.add_argument("--subject-id", nargs="+", default=None)
    parser.add_argument("--action-id", nargs="+", default=None)
    parser.add_argument(
        "--limit", type=int, default=0, help="0이면 전체, 양수이면 앞에서 N개"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="0이면 끝까지 (시험용)"
    )
    parser.add_argument("--kp-conf", type=float, default=0.50)
    parser.add_argument(
        "--main-person",
        choices=MAIN_PERSON_MODES,
        default="area",
        help="area = 가장 큰 BBox (1조 D6) · conf = 교재 22번 방식",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def select_clips(manifest: pd.DataFrame, args) -> pd.DataFrame:
    selected = manifest
    if args.clip_id:
        unknown = sorted(set(args.clip_id) - set(manifest["clip_id"]))
        if unknown:
            raise SystemExit(f"manifest에 없는 clip_id: {unknown}")
        selected = selected[selected["clip_id"].isin(args.clip_id)]
    if args.subject_id:
        selected = selected[selected["subject_id"].isin(args.subject_id)]
    if args.action_id:
        selected = selected[selected["action_id"].isin(args.action_id)]
    if args.limit > 0:
        selected = selected.head(args.limit)
    return selected


# ==================================================
# [03] YOLO 결과 한 장 → CSV 한 Row
# ==================================================
def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def frame_row(result, base: dict, kp_conf: float, main_person: str) -> dict:
    """22번 process_clip 안의 Frame 처리와 같은 순서로 한 Row를 만듭니다."""
    row = {column: "" for column in ALL_COLUMNS}
    row.update(base)
    row["pose_ok"] = 0
    row["feature_ok"] = 0
    row["people"] = 0

    index = select_main_person(result, main_person)
    if index is None:
        row["fail_reason"] = "no_person"
        return row

    keypoints = _to_numpy(result.keypoints.xy[index]).astype(np.float32)
    conf_all = getattr(result.keypoints, "conf", None)
    confidence = None if conf_all is None else _to_numpy(conf_all[index])

    row["people"] = len(result.keypoints.xy)
    row["person_index"] = index
    row["pose_ok"] = 1

    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > index:
        x1, y1, x2, y2 = _to_numpy(boxes.xyxy[index]).tolist()
        row.update(
            bbox_x1=round(x1, 1),
            bbox_y1=round(y1, 1),
            bbox_x2=round(x2, 1),
            bbox_y2=round(y2, 1),
        )
        if boxes.conf is not None:
            row["bbox_conf"] = round(float(_to_numpy(boxes.conf[index])), 4)

    # 원본 Pixel 좌표와 관절별 Confidence (Confidence가 없으면 빈칸 — 1.0으로 채우지 않음)
    for k, name in enumerate(COCO_KEYPOINT_NAMES):
        row[f"raw_{name}_x"] = round(float(keypoints[k, 0]), 2)
        row[f"raw_{name}_y"] = round(float(keypoints[k, 1]), 2)
        if confidence is not None:
            row[f"conf_{name}"] = round(float(confidence[k]), 4)

    if confidence is not None:
        row["mean_keypoint_conf"] = round(float(np.mean(confidence)), 4)
        row["low_keypoint_count"] = int(np.sum(confidence < kp_conf))
        row["norm_joint_low_count"] = int(
            sum(confidence[j] < kp_conf for j in NORM_JOINTS)
        )

    try:
        normalized, center, scale = normalize_by_shoulder_width(keypoints)
    except ValueError:
        row["fail_reason"] = "shoulder_width_zero"
        return row

    angles = selected_joint_angles(keypoints)
    values = [float(v) for point in normalized for v in point[:2]]
    values += [float(angles[name]) for name in ANGLE_NAMES]

    if len(values) != len(FEATURE_COLUMNS):
        row["fail_reason"] = "feature_count"
        return row
    if not np.all(np.isfinite(values)):
        row["fail_reason"] = "nan_or_inf"
        return row

    row.update({name: round(value, 6) for name, value in zip(FEATURE_COLUMNS, values)})
    row["feature_ok"] = 1
    row["hip_center_x"] = round(float(center[0]), 4)
    row["hip_center_y"] = round(float(center[1]), 4)
    row["shoulder_width"] = round(float(scale), 4)
    return row


# ==================================================
# [04] clip 한 개 처리
# ==================================================
def process_clip(model, clip: dict, video_dir: Path, settings: dict) -> pd.DataFrame:
    video_path = video_dir / clip["video_file"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"FPS를 확인할 수 없습니다: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    frame_id = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        result = model.predict(
            source=frame,
            imgsz=settings["image_size"],
            conf=settings["person_confidence"],
            device=settings["device"],
            verbose=False,
        )[0]
        base = {
            "clip_id": clip["clip_id"],
            "subject_id": clip["subject_id"],
            "action_id": clip["action_id"],
            "action_label": clip["action_label"],
            "take_id": clip["take_id"],
            "source_type": clip.get("source_type", ""),
            "source": clip["video_file"],
            "frame_id": frame_id,
            # 영상마다 실제 FPS로 나눔 (STEP 2: 29.597 · 29.149 fps clip 있음)
            "time_sec": round(frame_id / fps, 4),
            "fps": round(fps, 3),
            "image_width": width,
            "image_height": height,
        }
        rows.append(
            frame_row(result, base, settings["kp_conf"], settings["main_person"])
        )
        frame_id += 1
        if settings["max_frames"] and frame_id >= settings["max_frames"]:
            break
    capture.release()

    if not rows:
        raise RuntimeError(f"처리된 Frame이 없습니다: {video_path}")
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


# ==================================================
# [05] 실행
# ==================================================
def main(argv=None, model=None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.kp_conf <= 1.0:
        raise ValueError("--kp-conf는 0~1 범위여야 합니다.")

    manifest_path = Path(args.manifest)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not video_dir.exists():
        raise FileNotFoundError(
            f"{video_dir}\n먼저 python scripts/day04_make_clip_manifest.py --apply --link"
        )

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    clips = select_clips(manifest, args)
    if clips.empty:
        raise SystemExit("처리할 clip이 없습니다. 옵션을 확인하세요.")
    output_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    skipped = 0
    for clip in clips.to_dict("records"):
        if (output_dir / f"{clip['clip_id']}_sequence.csv").exists() and not (
            args.overwrite
        ):
            skipped += 1
        else:
            todo.append(clip)
    print(f"대상 {len(clips)}개 · 이미 있음(건너뜀) {skipped}개 · 처리 {len(todo)}개")
    print(f"대표 인물: {args.main_person} · kp-conf: {args.kp_conf}")
    if not todo:
        print("Team01 sequences: 새로 만들 것 없음 (--overwrite로 다시 만들 수 있음)")
        return 0

    # YOLO·torch는 실제로 처리할 때만 불러옵니다 (테스트에서는 가짜 model 사용)
    if model is None:
        from ultralytics import YOLO

        from src.common import load_config, resolve_device

        config = load_config()
        model = YOLO(config["pose_model"])
        device = resolve_device(config["device"])
    else:
        config = {"image_size": 640, "person_confidence": 0.25}
        device = "cpu"

    settings = {
        "image_size": config["image_size"],
        "person_confidence": config["person_confidence"],
        "device": device,
        "kp_conf": args.kp_conf,
        "main_person": args.main_person,
        "max_frames": args.max_frames,
    }

    summary = []
    for count, clip in enumerate(todo, start=1):
        sequence = process_clip(model, clip, video_dir, settings)
        output_path = output_dir / f"{clip['clip_id']}_sequence.csv"
        sequence.to_csv(output_path, index=False, encoding="utf-8-sig")
        pose_ok = int(sequence["pose_ok"].sum())
        feature_ok = int(sequence["feature_ok"].sum())
        multi = int((sequence["people"] > 1).sum())
        summary.append((clip["clip_id"], len(sequence), pose_ok, feature_ok, multi))
        print(
            f"[SAVED {count}/{len(todo)}] {clip['clip_id']} "
            f"rows={len(sequence)} pose_ok={pose_ok} feature_ok={feature_ok} "
            f"multi_person_frames={multi}"
        )

    table = pd.DataFrame(
        summary, columns=["clip_id", "rows", "pose_ok", "feature_ok", "multi_person"]
    )
    print()
    print(
        f"합계: rows {table['rows'].sum()} · pose_ok {table['pose_ok'].sum()} · "
        f"feature_ok {table['feature_ok'].sum()} · 여러 사람 Frame {table['multi_person'].sum()}"
    )
    incomplete = table[table["feature_ok"] < table["rows"]]
    if not incomplete.empty:
        print("feature_ok가 100%가 아닌 clip (STEP 5 QA에서 확인)")
        print(incomplete.to_string(index=False))
    print(f"Saved: {output_dir}")
    print("Team01 sequences: READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())