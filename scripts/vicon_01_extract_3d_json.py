"""VICON GLB → 3D JSON 일괄 추출 + 자동 검증 (V-01~V-03).

docs/referee/01_관절_매핑_명세.md
- D4 = ③ Python으로 GLB 직접 읽기 (6.3)
- 출력 형식: 7.2 3D JSON
- 검증: 8장 V-01~V-03 (V-03은 몸 기준 좌우 검사)

입력
- data/take_log.csv              (scripts/vicon_00_make_take_log.py 결과, use = Y만 처리)
- data/raw/vicon/**/*.glb        (Unreal Export GLB, 원본 FBX와 같은 이름)
출력
- data/vicon_3d/<clip_id>_3d.json
- reports/vicon/extract_3d_report.csv

실행:
    uv run python scripts/vicon_01_extract_3d_json.py
    uv run python scripts/vicon_01_extract_3d_json.py --clip P001_A002_T03 --overwrite

--clip로 일부만 실행하면 리포트에도 그 Take만 기록됨 (전체 리포트는 옵션 없이 다시 실행)
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vicon_utils import (
    GlbMotion,
    angle_diff_deg,
    facing_deg,
    load_glb_motion,
    toe_deg,
)

DEFAULT_TAKE_LOG = ROOT / "data" / "take_log.csv"
DEFAULT_GLB_DIR = ROOT / "data" / "raw" / "vicon"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "vicon_3d"
DEFAULT_REPORT = ROOT / "reports" / "vicon" / "extract_3d_report.csv"

EXPECTED_BONES = 73
EXPECTED_FPS = 60
# 01 문서 5.2 매핑표에서 쓰는 Bone
REQUIRED_BONES = [
    "Head", "HeadEnd",
    "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
    "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot",
    "LeftHandIndex1", "LeftHandPinky1", "RightHandIndex1", "RightHandPinky1",
]  # fmt: skip
# 방향·발높이 계산에 추가로 쓰는 Bone
HELPER_BONES = [
    "Hips",
    "LeftToeBase",
    "RightToeBase",
    "LeftToeBaseEnd",
    "RightToeBaseEnd",
]

HEIGHT_RANGE_M = (1.2, 2.2)  # 발바닥 ~ HeadEnd 높이. 단위 m 확인용
FACING_TOE_MAX_DEG = 45.0  # V-03: 골반 기준 앞 방향과 발끝 방향 차이 허용치
TPOSE_SPAN_RATIO = 0.7  # 첫 프레임 양손 수평 거리 / 키 > 0.7 이면 T-pose로 봄

REPORT_COLUMNS = [
    "clip_id", "source_glb", "status", "frames", "duration_sec", "fps",
    "height_m", "facing_deg", "toe_deg", "start_tpose", "hips_move_m",
    "V01", "V02", "V03", "message",
]  # fmt: skip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VICON GLB → 3D JSON")
    parser.add_argument("--take-log", type=Path, default=DEFAULT_TAKE_LOG)
    parser.add_argument(
        "--glb-dir",
        type=Path,
        default=DEFAULT_GLB_DIR,
        help="GLB 폴더 (하위 폴더 포함)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--clip", action="append", help="이 clip_id만 처리 (여러 번 사용 가능)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="이미 있는 JSON도 다시 저장"
    )
    return parser.parse_args()


def read_take_log(path: Path, only: list[str] | None) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["use"] == "Y" and r["clip_id"]]
    if only:
        rows = [r for r in rows if r["clip_id"] in set(only)]
    return rows


def index_glbs(folder: Path) -> dict[str, Path]:
    glbs: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".glb":
            if path.stem in glbs:
                raise RuntimeError(f"같은 이름의 GLB가 2개: {glbs[path.stem]} / {path}")
            glbs[path.stem] = path
    return glbs


def check_motion(motion: GlbMotion) -> dict:
    """V-01~V-03 검사 + 요약값."""
    result: dict = {"messages": []}
    names = motion.bone_names

    missing = [b for b in REQUIRED_BONES + HELPER_BONES if b not in names]
    v01 = len(names) == EXPECTED_BONES and not missing
    if len(names) != EXPECTED_BONES:
        result["messages"].append(f"Bone {len(names)}개 (기대 {EXPECTED_BONES})")
    if missing:
        result["messages"].append(f"없는 Bone: {', '.join(missing)}")
    result["V01"] = v01
    if missing:  # 아래 계산 불가
        result.update(V02=False, V03=False)
        return result

    dt = np.diff(motion.times)
    fps = 1.0 / float(np.median(dt))
    uniform = bool(np.allclose(dt, dt[0], atol=1e-4))
    foot_y = np.minimum(
        motion.joint("LeftToeBaseEnd")[:, 1], motion.joint("RightToeBaseEnd")[:, 1]
    )
    height = float(np.median(motion.joint("HeadEnd")[:, 1] - foot_y))
    y_up = bool(np.mean(motion.joint("Head")[:, 1] > motion.joint("Hips")[:, 1]) > 0.95)
    v02 = (
        abs(fps - EXPECTED_FPS) < 0.5
        and uniform
        and y_up
        and HEIGHT_RANGE_M[0] <= height <= HEIGHT_RANGE_M[1]
    )
    if not v02:
        result["messages"].append(
            f"V-02: fps {fps:.2f}, 간격 일정 {uniform}, Y-up {y_up}, 높이 {height:.2f} m"
        )
    result.update(V02=v02, fps=round(fps, 3), height_m=round(height, 3))

    face, toe = facing_deg(motion), toe_deg(motion)
    v03 = angle_diff_deg(face, toe) <= FACING_TOE_MAX_DEG
    if not v03:
        result["messages"].append(
            f"V-03: 골반 앞 {face:.0f}° vs 발끝 {toe:.0f}° → 좌우 뒤바뀜 의심"
        )
    result.update(V03=v03, facing_deg=round(face, 1), toe_deg=round(toe, 1))

    span = float(
        np.linalg.norm(
            (motion.joint("LeftHand")[0] - motion.joint("RightHand")[0])[[0, 2]]
        )
    )
    hips = motion.joint("Hips")[:, [0, 2]]
    result["start_tpose"] = span / height > TPOSE_SPAN_RATIO
    result["hips_move_m"] = round(
        float(np.max(np.linalg.norm(hips - hips[0], axis=1))), 2
    )
    return result


def build_json(row: dict, glb: Path, motion: GlbMotion, check: dict) -> dict:
    frames = []
    for k, t in enumerate(motion.times):
        joints = [
            {
                "name": n,
                "x": round(float(p[0]), 6),
                "y": round(float(p[1]), 6),
                "z": round(float(p[2]), 6),
            }
            for n, p in zip(motion.bone_names, motion.positions[k])
        ]
        frames.append(
            {
                "frame": k,
                "time_sec": round(float(t - motion.times[0]), 6),
                "joints": joints,
            }
        )
    return {
        "clip_id": row["clip_id"],
        "source_file": row["source_file"],
        "source_glb": glb.name,
        "animation": motion.animation_name,
        "person_id": row["person_id"],
        "action_id": row["action_id"],
        "take": int(row["take"]),
        "variant": row["variant"],
        "main_label": row["main_label"],
        "fps": EXPECTED_FPS,
        "duration_sec": round(float(motion.times[-1] - motion.times[0]), 6),
        "coordinate_system": "Y-up",
        "unit": "m",
        "bone_count": len(motion.bone_names),
        "frame_count": motion.frame_count,
        "facing_deg": check["facing_deg"],
        "start_tpose": bool(check["start_tpose"]),
        "frames": frames,
    }


def main() -> int:
    args = parse_args()
    if not args.take_log.is_file():
        print(
            f"[오류] take_log가 없음: {args.take_log} → 먼저 vicon_00_make_take_log.py 실행"
        )
        return 1
    if not args.glb_dir.is_dir():
        print(f"[오류] GLB 폴더가 없음: {args.glb_dir} (--glb-dir로 지정)")
        return 1

    rows = read_take_log(args.take_log, args.clip)
    if not rows:
        print("[오류] 처리할 Take가 없음 (use = Y, --clip 확인)")
        return 1
    glbs = index_glbs(args.glb_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for row in rows:
        stem = Path(row["source_file"]).stem
        line = {c: "" for c in REPORT_COLUMNS}
        line["clip_id"] = row["clip_id"]
        glb = glbs.get(stem)
        if glb is None:
            line.update(status="MISSING", message=f"GLB 없음: {stem}.glb")
            report.append(line)
            print(f"{row['clip_id']:<15} MISSING  {stem}.glb")
            continue

        line["source_glb"] = glb.name
        try:
            motion = load_glb_motion(glb)
            check = check_motion(motion)
        except (
            ValueError,
            KeyError,
            IndexError,
            OSError,
            struct.error,
        ) as exc:  # 파일 손상 등
            line.update(status="FAIL", message=f"읽기 오류: {exc}")
            report.append(line)
            print(f"{row['clip_id']:<15} FAIL     {exc}")
            continue

        ok = check["V01"] and check["V02"] and check["V03"]
        out = args.output_dir / f"{row['clip_id']}_3d.json"
        if ok and (args.overwrite or not out.exists()):
            data = build_json(row, glb, motion, check)
            out.write_text(
                json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            saved = "저장"
        elif ok:
            saved = "있음(건너뜀)"
        else:
            saved = "저장 안 함"

        line.update(
            status="PASS" if ok else "FAIL",
            frames=motion.frame_count,
            duration_sec=round(float(motion.times[-1] - motion.times[0]), 3),
            fps=check.get("fps", ""),
            height_m=check.get("height_m", ""),
            facing_deg=check.get("facing_deg", ""),
            toe_deg=check.get("toe_deg", ""),
            start_tpose=check.get("start_tpose", ""),
            hips_move_m=check.get("hips_move_m", ""),
            V01=check["V01"],
            V02=check["V02"],
            V03=check["V03"],
            message="; ".join(check["messages"]),
        )
        report.append(line)
        print(
            f"{row['clip_id']:<15} {line['status']:<8} {motion.frame_count:>4}f "
            f"향한각 {line['facing_deg']:>7}° T-pose {line['start_tpose']!s:<5} {saved}"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(report)

    counts = {
        s: sum(r["status"] == s for r in report) for s in ("PASS", "FAIL", "MISSING")
    }
    print(
        f"\n[결과] PASS {counts['PASS']} / FAIL {counts['FAIL']} / MISSING {counts['MISSING']} (총 {len(report)})"
    )
    print(f"[리포트] {args.report}")
    print(f"[JSON]   {args.output_dir}")
    return 0 if counts["PASS"] == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())