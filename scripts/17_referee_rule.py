from __future__ import annotations

# ==================================================
# 17_referee_rule.py — 1조 규칙 기반 판정 (U파울)
# ==================================================
# 교재 17_arms_up_rule.py와 같은 구조로,
# Arms Up 대신 1조 신호 "U파울"을 규칙으로 판정합니다.
#
# 1조 결정 (2026-11-16, reports/day03/challenge/feature_design.md 9장)
#   조건 1: 양 손목이 모두 코보다 위
#           (wrist_y - nose_y < 0, 정규화 좌표)
#   조건 2: 양 손목 사이 거리 < T  (기본 T = 1.0 어깨 폭)
#   두 조건 모두 만족 → U파울 True
#
# 실행 방법 두 가지
#   ① 사진 1장 (교재 17번과 같음)
#      python scripts/17_referee_rule.py --source 사진.jpg
#   ② 18번 CSV로 Frame마다 판정
#      python scripts/17_referee_rule.py --frames-csv 영상_frame_features.csv

# ==================================================
# ① 필요한 라이브러리와 공통 함수 가져오기
# ==================================================
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_utils import (  # noqa: E402
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    normalize_by_shoulder_width,
)


# ==================================================
# ② 터미널 실행 옵션
# ==================================================
# --source       : 판정할 사진 (방법 ①)
# --frames-csv   : 18번이 만든 Frame Feature CSV (방법 ②)
# --kp-conf      : 필요한 관절의 최소 Confidence (기본 0.50, 방법 ①만)
# --max-wrist-dist : 조건 2의 T, 어깨 폭 단위 (기본 1.0)
def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source")
    group.add_argument("--frames-csv")
    parser.add_argument("--kp-conf", type=float, default=0.50)
    parser.add_argument("--max-wrist-dist", type=float, default=1.0)
    return parser.parse_args()


# ==================================================
# ③ 규칙 본체 — 정규화 좌표 몇 개만 받아서 판정
# ==================================================
# y는 화면 아래로 갈수록 커집니다.
# wrist_y - nose_y 가 음수이면 손목이 코보다 위입니다.
#
# 사진 모드와 CSV 모드가 같은 함수를 쓰므로
# 두 방법의 판정 기준이 항상 같습니다.
def judge_ufoul(
    nose_y: float,
    left_wrist: tuple[float, float],
    right_wrist: tuple[float, float],
    max_wrist_dist: float,
) -> dict:
    left_minus_nose = left_wrist[1] - nose_y
    right_minus_nose = right_wrist[1] - nose_y
    wrist_distance = math.hypot(
        left_wrist[0] - right_wrist[0],
        left_wrist[1] - right_wrist[1],
    )

    cond1 = left_minus_nose < 0 and right_minus_nose < 0
    cond2 = wrist_distance < max_wrist_dist

    return {
        "left_wrist_minus_nose": left_minus_nose,
        "right_wrist_minus_nose": right_minus_nose,
        "wrist_distance": wrist_distance,
        "cond1_wrists_above_nose": cond1,
        "cond2_wrists_close": cond2,
        "ufoul": cond1 and cond2,
    }


# ==================================================
# ④ 방법 ① — 사진 1장 판정 (교재 17번 흐름)
# ==================================================
def run_image(args) -> None:
    from ultralytics import YOLO

    from src.common import load_config, resolve_device

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    result = model.predict(
        source=str(source),
        imgsz=config["image_size"],
        conf=config["person_confidence"],
        device=device,
        verbose=False,
    )[0]

    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        raise RuntimeError("Pose를 찾지 못했습니다.")

    # 교재 17번과 같이 첫 번째 사람 사용
    keypoints = result.keypoints.xy[0].cpu().numpy()
    if result.keypoints.conf is not None:
        confidence = result.keypoints.conf[0].cpu().numpy()
    else:
        confidence = np.ones(17, dtype=np.float32)

    # 필요한 관절: 코 · 양 손목 (판정) + 양 어깨 (정규화의 자)
    required_ids = [NOSE, LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER]
    min_index = min(required_ids, key=lambda i: float(confidence[i]))
    min_conf = float(confidence[min_index])

    if min_conf < args.kp_conf:
        print(
            f"SKIP: 필요한 관절의 최소 Confidence가 "
            f"{min_conf:.3f}입니다. (관절 번호 {min_index})"
        )
        return

    normalized, _, _ = normalize_by_shoulder_width(keypoints)

    r = judge_ufoul(
        nose_y=float(normalized[NOSE, 1]),
        left_wrist=(float(normalized[LEFT_WRIST, 0]), float(normalized[LEFT_WRIST, 1])),
        right_wrist=(float(normalized[RIGHT_WRIST, 0]), float(normalized[RIGHT_WRIST, 1])),
        max_wrist_dist=args.max_wrist_dist,
    )

    print(f"Source: {source.name}")
    print(f"Min required confidence: {min_conf:.3f}")
    print(f"Left wrist - nose : {r['left_wrist_minus_nose']:+.3f}")
    print(f"Right wrist - nose: {r['right_wrist_minus_nose']:+.3f}")
    print(f"Wrist distance    : {r['wrist_distance']:.3f}  (T = {args.max_wrist_dist})")
    print(f"조건 1 (양 손목이 코보다 위): {r['cond1_wrists_above_nose']}")
    print(f"조건 2 (양 손목이 가까움)   : {r['cond2_wrists_close']}")
    print(f"U파울: {r['ufoul']}")


# ==================================================
# ⑤ 방법 ② — 18번 CSV로 Frame마다 판정
# ==================================================
# 18번 CSV에는 관절별 Confidence가 없으므로
# valid_pose = 1 인 Frame만 판정합니다.
def run_frames_csv(args) -> None:
    import pandas as pd

    path = Path(args.frames_csv)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    valid = df[df["valid_pose"] == 1]

    rows = []
    for _, f in valid.iterrows():
        r = judge_ufoul(
            nose_y=float(f["nose_y"]),
            left_wrist=(float(f["left_wrist_x"]), float(f["left_wrist_y"])),
            right_wrist=(float(f["right_wrist_x"]), float(f["right_wrist_y"])),
            max_wrist_dist=args.max_wrist_dist,
        )
        rows.append(
            {
                "frame_id": int(f["frame_id"]),
                "time_sec": float(f["time_sec"]),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
            }
        )

    output = ROOT / "reports" / "day03" / "tables" / f"{path.stem}_referee_rule.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    n = len(rows)
    n_cond1 = sum(r["cond1_wrists_above_nose"] for r in rows)
    n_true = sum(r["ufoul"] for r in rows)
    true_times = [r["time_sec"] for r in rows if r["ufoul"]]

    print(f"Source: {path.name}")
    print(f"Valid frames: {n} / {len(df)}")
    print(f"조건 1 만족   : {n_cond1} ({n_cond1 / n:.0%})" if n else "조건 1 만족   : 0")
    print(f"U파울 True    : {n_true} ({n_true / n:.0%})" if n else "U파울 True    : 0")
    if true_times:
        print(f"처음 ~ 마지막 True: {min(true_times):.2f}s ~ {max(true_times):.2f}s")
    print(f"Saved: {output}")


def main() -> None:
    args = parse_args()
    if args.source:
        run_image(args)
    else:
        run_frames_csv(args)


if __name__ == "__main__":
    main()