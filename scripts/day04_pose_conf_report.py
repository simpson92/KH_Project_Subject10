from __future__ import annotations

# ==================================================
# day04_pose_conf_report.py
# 1조 본 촬영 → YOLO Pose 관절별 confidence 표 (STEP 4 Sequence CSV에서 바로 계산)
# ==================================================
# 질문
#   본 촬영 영상에서 YOLO가 17개 관절을 얼마나 확신하고 찾았나?
#   어느 관절 · 어느 동작 · 어느 clip이 약한가? (스모크 pose_quality.csv와 비교)
#
# 입력
#   STEP 4 Sequence CSV (<clip_id>_sequence.csv) 의 conf_<관절> 열
#   - pose_ok = 1 인 Frame만 셈 (사람을 못 찾은 Frame은 따로 pose_ok 비율로 남김)
#   - YOLO를 다시 돌리지 않음 → STEP 4 다음에 몇 초 안에 끝남
#
# 출력
#   <output-dir>/pose_conf_joints.csv  : 관절 17개 × (Frame 수 · 평균 · 최솟값 · 하위 5% · conf < 기준 비율)
#   <output-dir>/pose_conf_clips.csv   : clip마다 평균 · 팔 평균 · 손목/발목 최솟값 · 가장 약한 관절 · 판정
#   <output-dir>/pose_conf_actions.csv : 사람 × 동작마다 평균 · 손목 최솟값
#   --summary 경로                     : 위 표를 보고서에 붙일 Markdown으로 (Git에 올리는 파일)
#
# 판정 (clip)
#   CHECK : pose_ok 비율 < --min-pose-ok, 또는 손목·발목 4개 중 conf < --kp-conf 인 Frame 비율 > --max-low-ratio
#   PASS  : 그 밖
#
# 실행 (STEP 2~5 다음)
#   uv run python scripts/day04_pose_conf_report.py
#   uv run python scripts/day04_pose_conf_report.py --clip-prefix T01_S05   # 한 사람만
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import COCO_KEYPOINT_NAMES

MANIFEST_CSV = ROOT / "configs" / "team01_main_manifest.csv"
MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"
SEQUENCE_DIR = MAIN_DIR / "sequences"
OUTPUT_DIR = MAIN_DIR / "tables"
SUMMARY_MD = MAIN_DIR / "pose_conf_summary.md"

# 관절 묶음 (COCO 17 번호 순서)
JOINT_GROUPS = {
    "얼굴": COCO_KEYPOINT_NAMES[0:5],
    "팔": COCO_KEYPOINT_NAMES[5:11],
    "골반": COCO_KEYPOINT_NAMES[11:13],
    "다리": COCO_KEYPOINT_NAMES[13:17],
}
ARM_JOINTS = COCO_KEYPOINT_NAMES[5:11]
END_JOINTS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]  # 03 문서 Q-02·Q-04


def group_of(joint: str) -> str:
    for group, joints in JOINT_GROUPS.items():
        if joint in joints:
            return group
    return ""


# ==================================================
# [01] 실행 옵션
# ==================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--sequence-dir", default=str(SEQUENCE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--summary", default=str(SUMMARY_MD), help="Markdown 요약 경로 ('' = 만들지 않음)")
    parser.add_argument("--clip-prefix", default="", help="clip_id 앞부분으로 거르기 (예: T01_S05)")
    parser.add_argument("--kp-conf", type=float, default=0.50, help="관절을 믿는 기준 (STEP 4와 같게)")
    parser.add_argument("--min-pose-ok", type=float, default=0.90)
    parser.add_argument("--max-low-ratio", type=float, default=0.10)
    return parser.parse_args(argv)


# ==================================================
# [02] Sequence CSV 읽기
# ==================================================
def load_sequences(manifest: pd.DataFrame, sequence_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """manifest 순서대로 Sequence CSV를 이어 붙임. 없는 clip은 missing으로 돌려줌."""
    frames, missing = [], []
    for clip in manifest.itertuples(index=False):
        path = sequence_dir / f"{clip.clip_id}_sequence.csv"
        if not path.exists():
            missing.append(clip.clip_id)
            continue
        table = pd.read_csv(path, encoding="utf-8-sig")
        table["clip_id"] = clip.clip_id
        table["subject_id"] = clip.subject_id
        table["action_id"] = clip.action_id
        frames.append(table)
    if not frames:
        return pd.DataFrame(), missing
    return pd.concat(frames, ignore_index=True), missing


def conf_columns() -> list[str]:
    return [f"conf_{name}" for name in COCO_KEYPOINT_NAMES]


# ==================================================
# [03] 표 3개
# ==================================================
def joint_table(frames: pd.DataFrame, kp_conf: float) -> pd.DataFrame:
    """관절마다 평균 · 최솟값 · 하위 5% · 기준 미만 비율 (pose_ok Frame만)."""
    ok = frames[frames["pose_ok"] == 1]
    rows = []
    for name in COCO_KEYPOINT_NAMES:
        values = pd.to_numeric(ok[f"conf_{name}"], errors="coerce").dropna()
        rows.append(
            {
                "joint": name,
                "group": group_of(name),
                "frames": len(values),
                "mean": round(float(values.mean()), 4) if len(values) else np.nan,
                "min": round(float(values.min()), 4) if len(values) else np.nan,
                "p05": round(float(values.quantile(0.05)), 4) if len(values) else np.nan,
                "low_ratio": round(float((values < kp_conf).mean()), 4) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def clip_table(frames: pd.DataFrame, args) -> pd.DataFrame:
    """clip마다 요약 + PASS/CHECK."""
    rows = []
    for clip_id, group in frames.groupby("clip_id", sort=False):
        ok = group[group["pose_ok"] == 1]
        conf = ok[conf_columns()].apply(pd.to_numeric, errors="coerce")
        means = conf.mean()
        ends = conf[[f"conf_{n}" for n in END_JOINTS]]
        low_end = float((ends < args.kp_conf).any(axis=1).mean()) if len(ends) else 1.0
        pose_ok_ratio = len(ok) / len(group) if len(group) else 0.0
        weakest = means.idxmin().removeprefix("conf_") if means.notna().any() else ""
        reasons = []
        if pose_ok_ratio < args.min_pose_ok:
            reasons.append(f"pose_ok {pose_ok_ratio:.0%}")
        if low_end > args.max_low_ratio:
            reasons.append(f"손목·발목 conf<{args.kp_conf} {low_end:.0%}")
        rows.append(
            {
                "clip_id": clip_id,
                "subject_id": group["subject_id"].iloc[0],
                "action_id": group["action_id"].iloc[0],
                "frames": len(group),
                "pose_ok_ratio": round(pose_ok_ratio, 4),
                "mean_conf": round(float(conf.stack().mean()), 4) if len(ok) else np.nan,
                "arm_mean": round(float(conf[[f"conf_{n}" for n in ARM_JOINTS]].stack().mean()), 4)
                if len(ok)
                else np.nan,
                "wrist_min": round(float(conf[["conf_left_wrist", "conf_right_wrist"]].min().min()), 4)
                if len(ok)
                else np.nan,
                "ankle_min": round(float(conf[["conf_left_ankle", "conf_right_ankle"]].min().min()), 4)
                if len(ok)
                else np.nan,
                "end_low_ratio": round(low_end, 4),
                "weakest_joint": weakest,
                "status": "CHECK" if reasons else "PASS",
                "reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def action_table(clips: pd.DataFrame) -> pd.DataFrame:
    """사람 × 동작마다 clip 평균의 평균, 손목 최솟값의 최솟값."""
    return (
        clips.groupby(["subject_id", "action_id"], sort=True)
        .agg(
            clips=("clip_id", "count"),
            mean_conf=("mean_conf", "mean"),
            arm_mean=("arm_mean", "mean"),
            wrist_min=("wrist_min", "min"),
            check=("status", lambda s: int((s == "CHECK").sum())),
        )
        .round(4)
        .reset_index()
    )


# ==================================================
# [04] Markdown 요약 (보고서에 그대로 붙임)
# ==================================================
def _md(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in table.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if np.isnan(value) else f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary_markdown(joints, clips, actions, args, missing) -> str:
    total = int(joints["frames"].max()) if len(joints) else 0
    checks = clips[clips["status"] == "CHECK"]
    weakest = joints.sort_values("mean").head(3)["joint"].tolist()
    subject = (
        clips.groupby("subject_id")
        .agg(clips=("clip_id", "count"), mean_conf=("mean_conf", "mean"), arm_mean=("arm_mean", "mean"),
             wrist_min=("wrist_min", "min"), ankle_min=("ankle_min", "min"),
             check=("status", lambda s: int((s == "CHECK").sum())))
        .round(4)
        .reset_index()
    )
    parts = [
        "# YOLO Pose 관절 confidence 요약 (자동 생성)",
        "",
        "> `scripts/day04_pose_conf_report.py`가 만든 파일. 손으로 고치지 말고 다시 실행.",
        "",
        f"- 대상: clip {len(clips)}개 · pose_ok Frame {total:,}개 · 기준 conf ≥ {args.kp_conf}",
        f"- 전체 평균 conf **{clips['mean_conf'].mean():.3f}** · 팔(5~10번) 평균 **{clips['arm_mean'].mean():.3f}**",
        f"- 가장 약한 관절: {', '.join(weakest)}",
        f"- 판정: PASS {int((clips['status'] == 'PASS').sum())} · CHECK {len(checks)}",
    ]
    if missing:
        parts.append(f"- Sequence CSV 없음 {len(missing)}개: {', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}")
    parts += ["", "## 1. 관절별", "", _md(joints), "", "## 2. 사람별", "", _md(subject), ""]
    parts += ["## 3. 사람 × 동작", "", _md(actions), ""]
    parts += ["## 4. CHECK clip", ""]
    parts.append(_md(checks[["clip_id", "pose_ok_ratio", "wrist_min", "ankle_min", "reason"]]) if len(checks) else "없음")
    parts.append("")
    return "\n".join(parts)


# ==================================================
# [05] 실행
# ==================================================
def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    if args.clip_prefix:
        manifest = manifest[manifest["clip_id"].str.startswith(args.clip_prefix)]
    frames, missing = load_sequences(manifest, Path(args.sequence_dir))
    if frames.empty:
        print("Sequence CSV가 하나도 없습니다. STEP 4(day04_build_team_sequences.py)를 먼저 실행하세요.")
        return 1

    joints = joint_table(frames, args.kp_conf)
    clips = clip_table(frames, args)
    actions = action_table(clips)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joints.to_csv(out / "pose_conf_joints.csv", index=False, encoding="utf-8-sig")
    clips.to_csv(out / "pose_conf_clips.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(out / "pose_conf_actions.csv", index=False, encoding="utf-8-sig")

    print(joints.to_string(index=False))
    print(f"\nclip {len(clips)}개 · PASS {int((clips['status'] == 'PASS').sum())} · CHECK {int((clips['status'] == 'CHECK').sum())}")
    if missing:
        print(f"Sequence CSV 없음 {len(missing)}개")
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary_markdown(joints, clips, actions, args, missing), encoding="utf-8")
        print(f"Saved: {path}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
