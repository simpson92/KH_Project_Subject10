from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import COCO_KEYPOINT_NAMES, COCO_SKELETON, confidence_state
from src.referee_utils import KEYPOINT_DIR, RAW_DIR, SMOKE_REPORT_DIR


# ③ Confidence 상태별 색 (OpenCV는 BGR 순서)
STATE_COLORS = {
    "VALID": (80, 200, 80),     # 초록
    "WARNING": (0, 200, 255),   # 노랑
    "LOW": (60, 60, 230),       # 빨강
}

# 숫자로 Confidence를 같이 보여줄 팔 관절 (어깨·팔꿈치·손목)
ARM_IDS = [5, 6, 7, 8, 9, 10]
SHORT = {5: "L_sh", 6: "R_sh", 7: "L_el", 8: "R_el", 9: "L_wr", 10: "R_wr"}

OUTPUT_DIR = SMOKE_REPORT_DIR / "videos"
PREDICTION_CSV = SMOKE_REPORT_DIR / "loso_predictions.csv"


def parse_args():
    parser = argparse.ArgumentParser()

    # 무엇을 그릴지 선택 (셋 중 하나)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stem", nargs="+", help="예: S01_ufoul_T01 S04_ufoul_T01")
    group.add_argument("--only-wrong", action="store_true", help="3번에서 틀린 영상만")
    group.add_argument("--all", action="store_true", help="전체 영상")

    # 출력 크기 비율 (폰 영상 1080x1920 → 0.5면 540x960)
    parser.add_argument("--scale", type=float, default=0.5)

    # 왼쪽 위 정보창
    #   mini: 영상 이름 + 평균 Confidence 두 줄 (기본)
    #   full: 팔 관절 6개 값까지
    #   none: 정보창 없음 (관절 옆 숫자와 아래 결과만)
    parser.add_argument("--panel", choices=["mini", "full", "none"], default="mini")

    return parser.parse_args()


def find_video(stem: str) -> Path:
    for path in RAW_DIR.rglob(f"{stem}.*"):
        return path
    raise FileNotFoundError(f"원본 영상이 없습니다: {stem}")


def put_text(image, text, org, scale, color, thickness=1, outline=True):
    """글자 쓰기. outline=True면 검은 테두리를 둘러 밝은 배경에서도 보이게"""
    if outline:
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def dark_box(image, x1, y1, x2, y2, alpha=0.35):
    """반투명 검은 상자 (정보 패널 배경)"""
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def render(stem: str, scale: float, prediction: dict | None, panel: str) -> Path:
    # ④ 원본 영상과 2번에서 만든 관절 CSV 열기
    video_path = find_video(stem)
    keypoints = pd.read_csv(KEYPOINT_DIR / f"{stem}.csv", encoding="utf-8-sig")

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"영상을 읽지 못했습니다: {video_path}")

    height, width = frame.shape[:2]
    out_w, out_h = int(width * scale), int(height * scale)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{stem}_pose.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )

    # 글자·선 크기를 화면 크기에 맞춤
    unit = out_w / 540
    line_w = max(2, int(3 * unit))
    radius = max(3, int(6 * unit))

    frame_index = 0

    # ⑤ Frame마다 관절과 Skeleton 그리기
    while success:
        image = cv2.resize(frame, (out_w, out_h))

        if frame_index < len(keypoints):
            row = keypoints.iloc[frame_index]

            xs = np.array([row[f"{n}_x"] for n in COCO_KEYPOINT_NAMES], dtype=float) * scale
            ys = np.array([row[f"{n}_y"] for n in COCO_KEYPOINT_NAMES], dtype=float) * scale
            cs = np.array([row[f"{n}_c"] for n in COCO_KEYPOINT_NAMES], dtype=float)
            states = [confidence_state(c) for c in cs]

            if row["people"] > 0:
                # Skeleton 선: 두 관절 중 낮은 쪽 상태의 색
                for a, b in COCO_SKELETON:
                    if np.isnan(xs[a]) or np.isnan(xs[b]):
                        continue
                    state = states[a] if cs[a] < cs[b] else states[b]
                    cv2.line(
                        image,
                        (int(xs[a]), int(ys[a])),
                        (int(xs[b]), int(ys[b])),
                        STATE_COLORS[state],
                        line_w,
                        cv2.LINE_AA,
                    )

                # 관절 점
                for i in range(17):
                    if np.isnan(xs[i]):
                        continue
                    cv2.circle(image, (int(xs[i]), int(ys[i])), radius, STATE_COLORS[states[i]], -1, cv2.LINE_AA)
                    cv2.circle(image, (int(xs[i]), int(ys[i])), radius, (0, 0, 0), 1, cv2.LINE_AA)

                # 팔 관절 옆에 Confidence 숫자
                for i in ARM_IDS:
                    if np.isnan(xs[i]):
                        continue
                    put_text(
                        image,
                        f"{cs[i]:.2f}",
                        (int(xs[i]) + radius + 2, int(ys[i]) - radius),
                        0.45 * unit,
                        STATE_COLORS[states[i]],
                        1,
                    )

            # ⑥ 왼쪽 위 정보창 (작은 글씨 + 옅은 배경)
            lines = []
            if panel != "none":
                info = f"{stem}  {row['time_sec']:.1f}s"
                if row["people"] > 0:
                    info += f"  arm {cs[ARM_IDS].mean():.2f}"
                lines.append((info, (255, 255, 255)))
            if panel == "full" and row["people"] > 0:
                for i in ARM_IDS:
                    lines.append((f"{SHORT[i]} {cs[i]:.2f} {states[i]}", STATE_COLORS[states[i]]))

            if lines:
                step = int(17 * unit)
                text_w = max(
                    cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.42 * unit, 1)[0][0]
                    for t, _ in lines
                )
                dark_box(image, 0, 0, text_w + int(14 * unit), int(8 * unit) + step * len(lines))
                y = int(2 * unit)
                for text, color in lines:
                    y += step
                    put_text(image, text, (int(7 * unit), y), 0.42 * unit, color, 1, outline=False)

            # 3번 분류 결과가 있으면 아래쪽에 표시
            if prediction is not None:
                ok = prediction["true"] == prediction["pred"]
                color = (80, 200, 80) if ok else (60, 60, 230)
                text = f"TRUE {prediction['true']}  PRED {prediction['pred']}  {'OK' if ok else 'WRONG'}"
                text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * unit, 1)[0][0]
                dark_box(image, 0, out_h - int(26 * unit), text_w + int(14 * unit), out_h)
                put_text(image, text, (int(7 * unit), out_h - int(9 * unit)), 0.5 * unit, color, 1, outline=False)

        writer.write(image)
        frame_index += 1
        success, frame = capture.read()

    capture.release()
    writer.release()
    return output_path


def main() -> None:
    args = parse_args()

    # ⑦ 3번 결과(정답·예측)가 있으면 읽기
    predictions = {}
    if PREDICTION_CSV.exists():
        df = pd.read_csv(PREDICTION_CSV, encoding="utf-8-sig")
        predictions = {
            row["stem"]: {"true": row["true"], "pred": row["pred"]}
            for _, row in df.iterrows()
        }

    # ⑧ 그릴 영상 목록 정하기
    if args.stem:
        stems = args.stem
    elif args.only_wrong:
        stems = [s for s, p in predictions.items() if p["true"] != p["pred"]]
        if not stems:
            print("틀린 영상이 없습니다.")
            return
    else:
        stems = sorted(p.stem for p in KEYPOINT_DIR.glob("*.csv"))

    for index, stem in enumerate(stems, start=1):
        output = render(stem, args.scale, predictions.get(stem), args.panel)
        print(f"[{index}/{len(stems)}] {output}")

    print()
    print(f"색: 초록 VALID(>=0.50) / 노랑 WARNING(0.30~0.50) / 빨강 LOW(<0.30)")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()