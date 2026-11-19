from __future__ import annotations

# ==================================================
# day06_team_build_sequences.py   (6일차 실습 33 · 34번 → 1조 버전)
# 1조 Split(train · val · test)의 Keypoint CSV → Sequence Dataset (clip 1개 = 1행)
# ==================================================
# 왜?  수업 33 · 34번은 18 clip · 6동작 · S01~S03이 코드에 고정되어 있어 1조 450 clip에 못 씀.
#      Frame Feature · Resampling 계산은 수업과 똑같이 src/sequence_utils.py를 그대로 사용.
#        Frame 1개 = 38-D (어깨 폭 정규화 좌표 34 + 팔꿈치 · 무릎 각도 4)
#        clip 1개 = 30 Step × 38 = 1,140개 숫자  (frame_id 기준 Resampling)
#      정답 = train_label (A01~A15 · N00, 일상 동작 D01~D03 → N00)
#
# 입력: reports/team_projects/team01/main/dataset/splits/{train,val,test}.csv  (keypoint_csv 열)
#       data/team_projects/team01/main/keypoints/<S##>/<clip_id>.csv          (day05_team_export_keypoints.py 결과)
#       configs/team01_day06_baseline.json · configs/team01_subject_split.json
# 출력: data/team_projects/team01/main/day06/sequences/{train,val,test}.csv    (data/ = Git 제외)
#       data/team_projects/team01/main/day06/sequences/{split}_skipped.csv     (유효 Frame 부족 clip, 있을 때만)
#       reports/team_projects/team01/main/day06/sequence_summary.csv           (split별 clip 수 · 라벨 수)
#
# 실험 (설정 파일만 바꿔서 같은 흐름 재사용)
#   feature_set = "base38" (기본) | "team48" (38-D + 1조 신호 Feature G-01~G-10, src/team_feature_utils.py)
#   run_name 이 있으면 출력이 하위 폴더로 감: sequences/<run_name>/ · reports/.../day06/<run_name>/
#
# 실행
#   uv run python scripts/day06_team_build_sequences.py                                             # 38-D 기본
#   uv run python scripts/day06_team_build_sequences.py --config configs/team01_day06_e01_48d.json  # E-01 48-D
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import COCO_KEYPOINT_NAMES
from src.sequence_utils import (
    FRAME_FEATURE_NAMES,
    build_frame_feature_matrix,
    flatten_sequence,
    frame_feature_from_row,
    resample_sequence,
)
from src.team_feature_utils import SIGNAL_FEATURE_NAMES, signal_features

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"
SPLITS = ("train", "val", "test")
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
        "--config", default=str(ROOT / "configs" / "team01_day06_baseline.json")
    )
    parser.add_argument(
        "--split-config", default=str(ROOT / "configs" / "team01_subject_split.json")
    )
    parser.add_argument("--split-dir", default=str(MAIN_DIR / "dataset" / "splits"))
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "data" / "team_projects" / "team01" / "main" / "day06" / "sequences"
        ),
    )
    parser.add_argument(
        "--summary", default=str(MAIN_DIR / "day06" / "sequence_summary.csv")
    )
    parser.add_argument(
        "--root", default=str(ROOT), help="keypoint_csv 경로의 기준 폴더"
    )
    return parser.parse_args(argv)


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


FEATURE_SETS = {
    "base38": FRAME_FEATURE_NAMES,
    "team48": FRAME_FEATURE_NAMES + SIGNAL_FEATURE_NAMES,
}


def frame_names(feature_set: str) -> list:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set은 {list(FEATURE_SETS)} 중 하나: {feature_set}")
    return list(FEATURE_SETS[feature_set])


def feature_names(steps: int, feature_set: str = "base38") -> list:
    return [
        f"t{t:02d}_{name}" for t in range(steps) for name in frame_names(feature_set)
    ]


def run_paths(args, config: dict):
    """run_name이 있으면 하위 폴더 (기본 38-D 결과 경로는 그대로)."""
    run = config.get("run_name", "")
    out_dir = Path(args.output_dir) / run if run else Path(args.output_dir)
    summary = Path(args.summary)
    if run:
        summary = summary.parent / run / summary.name
    return out_dir, summary


def team48_frame_matrix(keypoints: pd.DataFrame):
    """38-D(수업 함수 그대로) 뒤에 G-01~G-10을 붙인 48-D. 48개 중 하나라도 계산 못 하면 그 Frame은 제외."""
    features, frame_ids = [], []
    for _, row in keypoints.sort_values("frame_id").iterrows():
        base = frame_feature_from_row(row)
        if base is None:
            continue
        points = np.asarray(base[:34], dtype=float).reshape(17, 2)
        conf_cols = [f"{n}_conf" for n in COCO_KEYPOINT_NAMES]
        conf = (
            row[conf_cols].to_numpy(dtype=float)
            if all(c in row.index for c in conf_cols)
            else None
        )
        g = signal_features(points, conf)
        if not np.all(np.isfinite(g)):
            continue
        features.append(np.concatenate([np.asarray(base, dtype=float), g]))
        frame_ids.append(int(row["frame_id"]))
    if not features:
        return np.empty((0, 48)), np.asarray([], dtype=np.int32)
    return np.stack(features), np.asarray(frame_ids, dtype=np.int32)


# --------------------------------------------------
# 입력 검사 (수업 33번 역할)
# --------------------------------------------------
def check_inputs(
    splits: dict, split_config: dict, labels: list, label_column: str, root: Path
) -> list:
    errors = []
    seen_clips = {}
    for name in SPLITS:
        df = splits[name]
        missing_cols = [
            c
            for c in ("clip_id", "subject", label_column, "keypoint_csv")
            if c not in df.columns
        ]
        if missing_cols:
            errors.append(f"{name}: 열 없음 {missing_cols}")
            continue

        expected = set(split_config[f"{name}_subjects"])
        actual = set(df["subject"])
        if actual != expected:
            errors.append(f"{name}: subject {sorted(actual)} ≠ 설정 {sorted(expected)}")

        missing_labels = sorted(set(labels) - set(df[label_column]))
        unknown_labels = sorted(set(df[label_column]) - set(labels))
        if missing_labels:
            errors.append(f"{name}: 없는 라벨 {missing_labels}")
        if unknown_labels:
            errors.append(f"{name}: 설정에 없는 라벨 {unknown_labels}")

        for clip_id in df["clip_id"]:
            if clip_id in seen_clips:
                errors.append(f"clip 겹침: {clip_id} ({seen_clips[clip_id]} · {name})")
            seen_clips[clip_id] = name

        for path in df["keypoint_csv"]:
            if not (root / path).exists():
                errors.append(
                    f"{name}: Keypoint CSV 없음 {path} (day05_team_export_keypoints.py 먼저 실행)"
                )
                break
    return errors


# --------------------------------------------------
# clip → 1행 (수업 34번 역할)
# --------------------------------------------------
def build_clip_row(
    item: dict,
    keypoints: pd.DataFrame,
    steps: int,
    min_valid: int,
    label_column: str,
    feature_set: str = "base38",
):
    if feature_set == "team48":
        features, frame_ids = team48_frame_matrix(keypoints)
    else:
        features, frame_ids = build_frame_feature_matrix(keypoints)
    valid = len(frame_ids)
    if valid < min_valid:
        return None, valid
    sequence = resample_sequence(features, frame_ids, target_steps=steps)
    row = {
        "clip_id": item["clip_id"],
        "subject": item["subject"],
        "action_id": item.get("action_id", ""),
        "train_label": item[label_column],
        "take": item.get("take", ""),
        "valid_frames": valid,
        "first_valid_frame": int(frame_ids[0]),
        "last_valid_frame": int(frame_ids[-1]),
    }
    row.update(dict(zip(feature_names(steps, feature_set), flatten_sequence(sequence))))
    return row, valid


def build_split(
    df: pd.DataFrame,
    root: Path,
    steps: int,
    min_valid: int,
    label_column: str,
    feature_set: str = "base38",
):
    rows, skipped = [], []
    for item in df.to_dict("records"):
        keypoints = pd.read_csv(root / item["keypoint_csv"], encoding="utf-8-sig")
        row, valid = build_clip_row(
            item, keypoints, steps, min_valid, label_column, feature_set
        )
        if row is None:
            skipped.append(
                {
                    "clip_id": item["clip_id"],
                    "train_label": item[label_column],
                    "valid_frames": valid,
                    "reason": f"유효 Frame {valid} < {min_valid}",
                }
            )
        else:
            rows.append(row)
    columns = META_COLUMNS + feature_names(steps, feature_set)
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(skipped)


def make_summary(
    built: dict, skipped: dict, labels: list, steps: int, dim: int = 38
) -> pd.DataFrame:
    rows = []
    for name in SPLITS:
        df = built[name]
        counts = df["train_label"].value_counts()
        row = {
            "split": name,
            "subjects": "|".join(sorted(df["subject"].unique())),
            "clips": len(df),
            "skipped": len(skipped[name]),
            "features": steps * dim,
        }
        row.update({label: int(counts.get(label, 0)) for label in labels})
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_json(args.config)
    split_config = load_json(args.split_config)
    steps = int(config["sequence_steps"])
    min_valid = int(config["min_valid_frames"])
    label_column = config["label_column"]
    labels = list(config["labels"])
    feature_set = config.get("feature_set", "base38")
    dim = len(frame_names(feature_set))
    root = Path(args.root)

    splits = {}
    for name in SPLITS:
        path = Path(args.split_dir) / f"{name}.csv"
        if not path.exists():
            print(f"[ERROR] 없음: {path}")
            return 1
        splits[name] = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)

    errors = check_inputs(splits, split_config, labels, label_column, root)
    if errors:
        print("[ERROR] 입력 검사 실패")
        for e in errors:
            print("  -", e)
        return 1
    for name in SPLITS:
        print(
            f"{name:5s}: {'·'.join(split_config[f'{name}_subjects'])} / {len(splits[name])} clips"
        )
    print("입력 검사: OK (subject · 라벨 16개 · clip 겹침 없음 · Keypoint CSV 있음)")
    print()

    out_dir, summary_path = run_paths(args, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    built, skipped = {}, {}
    for name in SPLITS:
        built[name], skipped[name] = build_split(
            splits[name], root, steps, min_valid, label_column, feature_set
        )
        built[name].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        skip_path = out_dir / f"{name}_skipped.csv"
        if len(skipped[name]):
            skipped[name].to_csv(skip_path, index=False, encoding="utf-8-sig")
        elif skip_path.exists():
            skip_path.unlink()
        print(
            f"{name:5s}: {len(built[name])} rows · skipped {len(skipped[name])} · columns {built[name].shape[1]}"
        )

    summary = make_summary(built, skipped, labels, steps, dim)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print()
    print(
        f"Frame Feature: {dim}-D ({feature_set}) · Step {steps} · clip 1개 = {steps * dim} Feature"
    )
    print(f"Saved: {out_dir}")
    print(f"Saved: {summary_path}")
    print("1조 Day06 Sequence Dataset: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
