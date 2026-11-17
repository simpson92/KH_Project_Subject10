from __future__ import annotations

# ==================================================
# day04_team_sequence_qa.py
# 1조 Sequence 100개 → 영상별 QA 요약표 — 4일차 팀 확장 STEP 5
# ==================================================
# 교재 23_summarize_action_sequences.py의 요약(성공률·Confidence·인접 변화량)을
# clip_id별로 계산하고, 팀 확장 문서 STEP 5 목록의 항목을 더합니다. (23번 원본은 그대로)
#
# 영상 한 개 = 한 줄
#   rows · pose_ok_ratio · feature_ok_ratio · mean_keypoint_conf       ← 23번과 같음
#   essential_low_ratio   : 필수 관절 중 하나라도 conf < kp-conf 인 Frame 비율
#   max_consecutive_bad   : (Pose/Feature 실패 또는 필수 관절 저신뢰) Frame이 연달아 나온 최대 길이
#   multi_person_ratio    : 사람이 2명 이상 검출된 Frame 비율 (Tracking 대신)
#   shoulder_collapse_frames : 어깨 폭이 그 영상 중앙값 × collapse-ratio 미만인 Frame 수
#   mean/max_adjacent_change : 23번 변화량(34개 정규화 좌표, L2)의 평균과 최댓값
#   manifest_status · keyframe_review_result : manifest 판정 · STEP 3 검수표에서 사람이 적은 결과
#   suggested_status · suggest_reasons       : 자동 제안 (PASS / REVIEW) + 이유
#
# 자동 제안은 REVIEW "후보"까지만 합니다.
#   팀 확장 문서: "모든 팀에 적용할 수 있는 단일 Confidence·성공률 합격 기준은 없다."
#   RESHOOT / EXCLUDE 결정은 사람이 영상을 보고 합니다 (5일차 QA).
#
# 필수 관절 (1조, 9개)
#   nose · 양 shoulder · 양 elbow · 양 wrist · 양 hip
#   = 17번 규칙(코·손목·어깨) + 각도(팔꿈치) + 정규화 기준(어깨·골반)
#   무릎·발목은 3일차 측정에서 U파울·푸싱 구분에 쓸모없어 제외 (feature_design.md)
#
# 실행
#   python scripts/day04_team_sequence_qa.py
#   python scripts/day04_team_sequence_qa.py --max-bad-run 3   # 기준 바꿔 보기
#
# 출력
#   reports/team_projects/team01/tables/sequence_qa.csv  (실명 없음, Git 제외)
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import COCO_KEYPOINT_NAMES

TEAM_DIR = ROOT / "reports" / "team_projects" / "team01"
SEQUENCE_DIR = TEAM_DIR / "sequences"
MANIFEST_CSV = ROOT / "configs" / "team01_clip_manifest.csv"
KEYFRAME_REVIEW_CSV = TEAM_DIR / "tables" / "keyframe_review.csv"
OUTPUT_CSV = TEAM_DIR / "tables" / "sequence_qa.csv"

ESSENTIAL_JOINTS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
COORD_COLUMNS = [f"{n}_{axis}" for n in COCO_KEYPOINT_NAMES for axis in ("x", "y")]
HUMAN_FLAGGED = {"수정", "FAIL", "REVIEW", "RESHOOT", "EXCLUDE"}

OUTPUT_COLUMNS = [
    "clip_id",
    "subject_id",
    "action_id",
    "action_label",
    "take_id",
    "rows",
    "duration_sec",
    "pose_ok_ratio",
    "feature_ok_ratio",
    "mean_keypoint_conf",
    "essential_low_ratio",
    "worst_essential_joint",
    "max_consecutive_bad",
    "multi_person_ratio",
    "shoulder_width_median",
    "shoulder_collapse_frames",
    "mean_adjacent_change",
    "max_adjacent_change",
    "manifest_status",
    "keyframe_review_result",
    "suggested_status",
    "suggest_reasons",
]


# ==================================================
# [01] 실행 옵션 (기준값은 바꿔 볼 수 있게)
# ==================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", default=str(SEQUENCE_DIR))
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--keyframe-review", default=str(KEYFRAME_REVIEW_CSV))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--kp-conf", type=float, default=0.50)
    parser.add_argument(
        "--low-ratio",
        type=float,
        default=0.10,
        help="필수 관절 저신뢰 Frame 비율이 이보다 크면 REVIEW",
    )
    parser.add_argument(
        "--max-bad-run",
        type=int,
        default=5,
        help="나쁜 Frame이 이 길이 이상 연속이면 REVIEW (D5 후보값 5)",
    )
    parser.add_argument(
        "--multi-ratio",
        type=float,
        default=0.0,
        help="여러 사람 Frame 비율이 이보다 크면 REVIEW",
    )
    parser.add_argument(
        "--collapse-ratio",
        type=float,
        default=0.5,
        help="어깨 폭 < 중앙값 × 이 값이면 붕괴 Frame",
    )
    return parser.parse_args(argv)


# ==================================================
# [02] 계산 함수 (한 Sequence = DataFrame)
# ==================================================
def adjacent_changes(df: pd.DataFrame) -> np.ndarray:
    """교재 23번 mean_adjacent_feature_change와 같은 계산.

    이웃한 두 Row가 모두 feature_ok=1일 때 34개 정규화 좌표 차이의 L2 길이.
    """
    if len(df) < 2:
        return np.array([])
    numeric = df[COORD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ok = df["feature_ok"] == 1
    pair_ok = ok & ok.shift(1, fill_value=False)
    diffs = numeric.diff().loc[pair_ok].dropna()
    if diffs.empty:
        return np.array([])
    return np.linalg.norm(diffs.to_numpy(dtype=float), axis=1)


def essential_low_mask(df: pd.DataFrame, kp_conf: float) -> tuple[pd.Series, str]:
    """필수 관절 중 하나라도 conf < kp_conf 인 Frame (Pose 없는 Frame은 False).

    Confidence가 빈칸이면 낮다고 단정하지 않습니다.
    가장 자주 낮았던 관절 이름도 돌려줍니다.
    """
    conf = df[[f"conf_{j}" for j in ESSENTIAL_JOINTS]].apply(
        pd.to_numeric, errors="coerce"
    )
    low = conf < kp_conf  # NaN < x 는 False
    per_joint = low.sum()
    worst = ""
    if per_joint.max() > 0:
        worst = per_joint.idxmax().removeprefix("conf_")
    return low.any(axis=1), worst


def longest_run(mask: pd.Series) -> int:
    best = current = 0
    for flag in mask.to_numpy(dtype=bool):
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def summarize_clip(df: pd.DataFrame, args) -> dict:
    rows = len(df)
    pose_ok = df["pose_ok"] == 1
    feature_ok = df["feature_ok"] == 1
    low_mask, worst = essential_low_mask(df, args.kp_conf)
    bad = (~pose_ok) | (~feature_ok) | low_mask

    width = pd.to_numeric(df["shoulder_width"], errors="coerce")
    median_width = float(width[feature_ok].median()) if feature_ok.any() else np.nan
    collapse = (
        int((width < median_width * args.collapse_ratio).sum())
        if np.isfinite(median_width)
        else 0
    )
    changes = adjacent_changes(df)
    people = pd.to_numeric(df["people"], errors="coerce").fillna(0)
    conf = pd.to_numeric(df["mean_keypoint_conf"], errors="coerce")

    return {
        "clip_id": df["clip_id"].iloc[0],
        "subject_id": df["subject_id"].iloc[0],
        "action_id": df["action_id"].iloc[0],
        "action_label": df["action_label"].iloc[0],
        "take_id": df["take_id"].iloc[0],
        "rows": rows,
        "duration_sec": round(float(df["time_sec"].max()), 3),
        "pose_ok_ratio": round(float(pose_ok.mean()), 4),
        "feature_ok_ratio": round(float(feature_ok.mean()), 4),
        "mean_keypoint_conf": round(float(conf[pose_ok].mean()), 4)
        if pose_ok.any()
        else 0.0,
        "essential_low_ratio": round(float(low_mask[pose_ok].mean()), 4)
        if pose_ok.any()
        else 1.0,
        "worst_essential_joint": worst,
        "max_consecutive_bad": longest_run(bad),
        "multi_person_ratio": round(float((people > 1).mean()), 4),
        "shoulder_width_median": round(median_width, 2)
        if np.isfinite(median_width)
        else "",
        "shoulder_collapse_frames": collapse,
        "mean_adjacent_change": round(float(changes.mean()), 6)
        if changes.size
        else 0.0,
        "max_adjacent_change": round(float(changes.max()), 6) if changes.size else 0.0,
    }


# ==================================================
# [03] 자동 제안 (PASS / REVIEW 후보)
# ==================================================
def suggest(row: dict, args) -> tuple[str, list[str]]:
    reasons = []
    if row["pose_ok_ratio"] < 1.0:
        reasons.append(f"Pose 실패 Frame 있음({row['pose_ok_ratio']:.2f})")
    if row["feature_ok_ratio"] < 1.0:
        reasons.append(f"Feature 실패 Frame 있음({row['feature_ok_ratio']:.2f})")
    if row["essential_low_ratio"] > args.low_ratio:
        reasons.append(
            f"필수 관절 저신뢰 {row['essential_low_ratio']:.0%}"
            f"(주로 {row['worst_essential_joint']})"
        )
    if row["max_consecutive_bad"] >= args.max_bad_run:
        reasons.append(f"나쁜 Frame {row['max_consecutive_bad']}개 연속")
    if row["multi_person_ratio"] > args.multi_ratio:
        reasons.append(f"여러 사람 Frame {row['multi_person_ratio']:.0%}")
    if row["shoulder_collapse_frames"] > 0:
        reasons.append(f"어깨 폭 붕괴 {row['shoulder_collapse_frames']} Frame")
    if row["manifest_status"] and row["manifest_status"] != "PASS":
        reasons.append(f"manifest {row['manifest_status']}")
    if row["keyframe_review_result"] in HUMAN_FLAGGED:
        reasons.append(f"대표 Frame 검수 {row['keyframe_review_result']}")
    return ("REVIEW" if reasons else "PASS"), reasons


# ==================================================
# [04] 실행
# ==================================================
def _lookup(path: Path, column: str) -> dict[str, str]:
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if column not in table.columns:
        return {}
    return dict(zip(table["clip_id"], table[column].str.strip()))


def main(argv=None) -> int:
    args = parse_args(argv)
    sequence_dir = Path(args.sequence_dir)
    files = sorted(sequence_dir.glob("*_sequence.csv"))
    if not files:
        raise SystemExit(
            f"Sequence CSV가 없습니다: {sequence_dir}\n"
            "먼저 python scripts/day04_build_team_sequences.py"
        )

    manifest_status = _lookup(Path(args.manifest), "review_status")
    keyframe_result = _lookup(Path(args.keyframe_review), "review_result")

    rows = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[SKIP] 빈 CSV: {path.name}")
            continue
        row = summarize_clip(df, args)
        row["manifest_status"] = manifest_status.get(row["clip_id"], "")
        row["keyframe_review_result"] = keyframe_result.get(row["clip_id"], "")
        status, reasons = suggest(row, args)
        row["suggested_status"] = status
        row["suggest_reasons"] = " | ".join(reasons)
        rows.append(row)

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---------- 화면 요약 ----------
    missing = sorted(set(manifest_status) - set(result["clip_id"]))
    print(f"Sequence {len(result)}개 · manifest {len(manifest_status)}개", end="")
    print(f" · Sequence 없는 clip {len(missing)}개" if missing else "")
    print(
        f"기준: kp-conf {args.kp_conf} · 저신뢰 > {args.low_ratio:.0%} · "
        f"연속 ≥ {args.max_bad_run} · 여러 사람 > {args.multi_ratio:.0%} · "
        f"어깨 폭 < 중앙값×{args.collapse_ratio}"
    )
    print()
    print("자동 제안:", result["suggested_status"].value_counts().to_dict())
    print()
    print("동작별 평균")
    by_action = result.groupby(["action_id", "action_label"]).agg(
        clips=("clip_id", "count"),
        frames=("rows", "sum"),
        conf=("mean_keypoint_conf", "mean"),
        low=("essential_low_ratio", "mean"),
        change=("mean_adjacent_change", "mean"),
        change_max=("max_adjacent_change", "max"),
    )
    print(by_action.round(4).to_string())

    review = result[result["suggested_status"] == "REVIEW"]
    if not review.empty:
        print()
        print(
            "REVIEW 후보 (사람이 영상·모아보기로 확인 → RESHOOT / EXCLUDE / PASS 결정)"
        )
        print(review[["clip_id", "suggest_reasons"]].to_string(index=False))
    if missing:
        print()
        shown = ", ".join(missing[:10])
        more = f" 외 {len(missing) - 10}개" if len(missing) > 10 else ""
        print(f"Sequence가 없는 clip: {shown}{more}")
    print()
    print(f"Saved: {output_path}")
    print("Team01 sequence QA: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())