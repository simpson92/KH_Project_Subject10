from __future__ import annotations

# ==================================================
# day05_team_label_sheet.py
# 구간 라벨 확인용 모아보기 — 4 Frame마다 1장, 테두리 색 = 라벨 (각자 PC에서만, Git 제외)
# ==================================================
# 회색 N00 · 노랑 X · 보라 C01 · 초록 신호. 칸 위 숫자 = Frame 번호
# 경계가 틀리면 labels.csv의 start_frame / end_frame을 고치고 labeler를 S## 로 바꿈
#
# 실행
#   uv run python scripts/day05_team_label_sheet.py --clip-id T01_S04_A03_T01 T01_S04_A15_T01
#   uv run python scripts/day05_team_label_sheet.py --status REVIEW        # REVIEW clip만
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN_DIR = ROOT / "reports" / "team_projects" / "team01" / "main"
VIDEO_DIR = ROOT / "data" / "team_projects" / "team01" / "main" / "rgb" / "videos"
COLORS = {"N00": (160, 160, 160), "X": (0, 200, 255), "C01": (255, 0, 255)}   # BGR
SIGNAL_COLOR = (0, 220, 0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(MAIN_DIR / "labels" / "team01_labels.csv"))
    parser.add_argument("--video-dir", default=str(VIDEO_DIR))
    parser.add_argument("--output-dir", default=str(MAIN_DIR / "label_sheets"))
    parser.add_argument("--clip-id", nargs="+", default=None)
    parser.add_argument("--status", default="", help="이 qa_status인 clip만 (예: REVIEW)")
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--cols", type=int, default=14)
    return parser.parse_args(argv)


def frame_labels(rows: pd.DataFrame) -> dict[int, str]:
    out = {}
    for _, r in rows.iterrows():
        for i in range(int(r["start_frame"]), int(r["end_frame"]) + 1):
            out[i] = r["label"]
    return out


def make_sheet(video: Path, labels: dict[int, str], step: int, cols: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    tiles, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            label = labels.get(i, "?")
            color = COLORS.get(label, SIGNAL_COLOR)
            tile = cv2.resize(frame, (90, 160))
            cv2.rectangle(tile, (0, 0), (89, 159), color, 4)
            cv2.putText(tile, str(i), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(tile, label, (4, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            tiles.append(tile)
        i += 1
    cap.release()
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])


def main(argv=None) -> int:
    args = parse_args(argv)
    labels = pd.read_csv(args.labels, keep_default_na=False)
    clips = args.clip_id or sorted(labels["clip_id"].unique())
    if args.status:
        status = labels.groupby("clip_id")["qa_status"].first()
        clips = [c for c in clips if status.get(c) == args.status.upper()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for clip in clips:
        video = Path(args.video_dir) / f"{clip}.mp4"
        if not video.exists():
            print(f"[SKIP] 영상 없음: {video}")
            continue
        sheet = make_sheet(video, frame_labels(labels[labels["clip_id"] == clip]), args.step, args.cols)
        if sheet is not None:
            cv2.imwrite(str(out / f"{clip}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 75])
            saved += 1
    print(f"Saved {saved}장: {out}  (얼굴이 보이므로 Git · 외부 공유 금지)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
