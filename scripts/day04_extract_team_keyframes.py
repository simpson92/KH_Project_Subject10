from __future__ import annotations

# ==================================================
# day04_extract_team_keyframes.py
# 1조 영상 대표 Frame — 4일차 팀 확장 STEP 3
# ==================================================
# 교재 21_extract_clip_keyframes.py는 {action_label}_start.jpg 처럼 저장합니다.
# 같은 동작이 여러 개면 서로 덮어쓰므로, 팀 확장 문서 STEP 3대로
# JPG 이름에 clip_id를 넣습니다. (21번 원본은 그대로 둡니다.)
#
# 만드는 것
#   ① clip별 대표 JPG 3장 : {clip_id}_start / _middle / _end.jpg
#        start = 0, middle = total // 2, end = total - 1   (21번과 같은 계산)
#   ② 모아보기(contact sheet) : 사람 × 동작마다 1장
#        행 = Take, 열 = 영상의 10·30·50·70·90% 위치
#        (교재 §12 현업 개선 "일정 간격 5~10장 · 한 장에 모아 보기")
#        손 수신호처럼 짧은 동작은 Middle 한 장에 안 잡힐 수 있어서 함께 봅니다.
#   ③ 검수표 CSV : clip_id · Label · manifest 상태 + 사람이 채울 빈칸
#        (교재 §13 검수표: 동작 일치 / 문제점 / 사용 구간 / 검수 결과)
#        이미 있으면 덮어쓰지 않습니다 (채운 내용 보호). 다시 만들려면 --overwrite-review
#
# 실행
#   python scripts/day04_extract_team_keyframes.py --limit 2   # 앞의 2개만 시험
#   python scripts/day04_extract_team_keyframes.py
#
# 주의
#   - JPG에는 조원 얼굴이 나옵니다 → Git 제외(*.jpg) · 팀 밖 공유 금지
#   - 정답 Label이 실제 동작과 맞는지는 코드가 판단하지 않습니다. 사람이 보고 검수표에 적습니다.
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV = ROOT / "configs" / "team01_clip_manifest.csv"
VIDEO_DIR = ROOT / "data" / "team_projects" / "team01" / "rgb" / "videos"
TEAM_REPORT_DIR = ROOT / "reports" / "team_projects" / "team01"
KEYFRAME_DIR = TEAM_REPORT_DIR / "keyframes"
REVIEW_CSV = TEAM_REPORT_DIR / "tables" / "keyframe_review.csv"

SHEET_POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
THUMB_WIDTH = 180
LABEL_WIDTH = 150
HEADER_HEIGHT = 40
JPEG_QUALITY = 90

REVIEW_COLUMNS = [
    "clip_id",
    "subject_id",
    "action_id",
    "action_label",
    "take_id",
    "manifest_status",
    "manifest_reason",
    "start_frame",
    "middle_frame",
    "end_frame",
    "contact_sheet",
    # ↓ 사람이 대표 Frame을 보고 채우는 칸
    "label_match",  # O / X
    "issue",  # 예: 다른 사람 등장, 손이 화면 밖, 동작 시작 전 대기 길음
    "use_range_sec",  # 예: 0.8-3.9
    "review_result",  # PASS / 수정 / FAIL
]


# ==================================================
# [01] 실행 옵션
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--video-dir", default=str(VIDEO_DIR))
    parser.add_argument("--output-dir", default=str(KEYFRAME_DIR))
    parser.add_argument("--review-csv", default=str(REVIEW_CSV))
    parser.add_argument(
        "--limit", type=int, default=0, help="0이면 전체, 양수이면 앞에서 N개만"
    )
    parser.add_argument("--thumb-width", type=int, default=THUMB_WIDTH)
    parser.add_argument(
        "--overwrite-review",
        action="store_true",
        help="이미 채운 검수표가 있어도 새로 만듦",
    )
    return parser.parse_args()


# ==================================================
# [02] 뽑을 Frame 번호 계산
# ==================================================
def keyframe_indices(total: int) -> dict[str, int]:
    """교재 21번과 같은 Start · Middle · End."""
    return {"start": 0, "middle": total // 2, "end": total - 1}


def sheet_indices(total: int, positions=SHEET_POSITIONS) -> list[int]:
    """영상 길이의 10·30·50·70·90% 위치."""
    return [round(p * (total - 1)) for p in positions]


# ==================================================
# [03] 필요한 Frame만 순서대로 읽기
# ==================================================
def read_frames(path: Path, wanted: set[int]) -> tuple[dict[int, np.ndarray], int]:
    """처음부터 순서대로 읽으며 wanted 번호의 Frame만 꺼냅니다.

    MP4에서 Frame 번호로 바로 이동(set)하면 위치가 어긋날 수 있어서
    STEP 2에서 확인한 "끝까지 순서대로 읽기"를 그대로 씁니다.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    frames: dict[int, np.ndarray] = {}
    index = 0
    last = max(wanted) if wanted else -1
    while index <= last and capture.grab():
        if index in wanted:
            success, frame = capture.retrieve()
            if success:
                frames[index] = frame
        index += 1
    capture.release()
    return frames, index


# ==================================================
# [04] 모아보기 한 장 만들기
# ==================================================
def thumbnail(frame: np.ndarray, width: int) -> np.ndarray:
    height = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def put_text(image, text, org, scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(
        image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2
    )
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def build_sheet(rows: list[dict], width: int, title: str) -> np.ndarray:
    """rows: [{"clip_id", "take_id", "status", "thumbs": [이미지 5장]}, ...]

    OpenCV putText는 한글을 못 그리므로 그림 안 글자는 영문·번호만 씁니다.
    """
    sample = next(t for row in rows for t in row["thumbs"] if t is not None)
    thumb_h = sample.shape[0]
    cols = len(SHEET_POSITIONS)
    sheet_w = LABEL_WIDTH + cols * width
    sheet_h = HEADER_HEIGHT + len(rows) * thumb_h
    sheet = np.full((sheet_h, sheet_w, 3), 40, np.uint8)

    put_text(sheet, title, (8, 26), 0.6)
    for c, p in enumerate(SHEET_POSITIONS):
        put_text(sheet, f"{int(p * 100)}%", (LABEL_WIDTH + c * width + 6, 26), 0.55)

    for r, row in enumerate(rows):
        y = HEADER_HEIGHT + r * thumb_h
        put_text(sheet, row["take_id"], (8, y + 28), 0.7)
        put_text(sheet, row["clip_id"], (8, y + 52), 0.38)
        if row["status"] and row["status"] != "PASS":
            put_text(sheet, row["status"], (8, y + 76), 0.55, (60, 60, 255), 2)
        for c, thumb in enumerate(row["thumbs"]):
            if thumb is None:
                continue
            x = LABEL_WIDTH + c * width
            h = min(thumb_h, thumb.shape[0])
            sheet[y : y + h, x : x + width] = thumb[:h, :width]
        cv2.line(sheet, (0, y), (sheet_w, y), (90, 90, 90), 1)
    return sheet


# ==================================================
# [05] 실행
# ==================================================
def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    review_path = Path(args.review_csv)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not video_dir.exists():
        raise FileNotFoundError(
            f"{video_dir}\n먼저 python scripts/day04_make_clip_manifest.py --apply --link"
        )

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if args.limit > 0:
        manifest = manifest.head(args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    review_rows = []
    sheet_rows: dict[tuple[str, str], list[dict]] = {}
    failed = []
    total_clips = len(manifest)

    for count, row in enumerate(manifest.to_dict("records"), start=1):
        clip_id = row["clip_id"]
        video_path = video_dir / row["video_file"]
        capture = cv2.VideoCapture(str(video_path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        capture.release()
        if total < 1:
            failed.append(clip_id)
            print(f"[FAIL] {clip_id} 영상을 열 수 없거나 Frame 수가 0")
            continue

        key = keyframe_indices(total)
        sheet_idx = sheet_indices(total)
        frames, read_count = read_frames(video_path, set(key.values()) | set(sheet_idx))

        missing = [name for name, i in key.items() if i not in frames]
        for name, i in key.items():
            if i in frames:
                cv2.imwrite(
                    str(output_dir / f"{clip_id}_{name}.jpg"),
                    frames[i],
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
        if missing:
            failed.append(clip_id)
            print(f"[FAIL] {clip_id} 읽지 못한 Frame: {missing} (읽은 수 {read_count})")

        group = (row["subject_id"], row["action_id"])
        sheet_name = f"contact_{group[0]}_{group[1]}.jpg"
        sheet_rows.setdefault(group, []).append(
            {
                "clip_id": clip_id,
                "take_id": row["take_id"],
                "status": row.get("review_status", ""),
                "thumbs": [
                    thumbnail(frames[i], args.thumb_width) if i in frames else None
                    for i in sheet_idx
                ],
            }
        )
        review_rows.append(
            {
                "clip_id": clip_id,
                "subject_id": row["subject_id"],
                "action_id": row["action_id"],
                "action_label": row.get("action_label", ""),
                "take_id": row["take_id"],
                "manifest_status": row.get("review_status", ""),
                "manifest_reason": row.get("review_reason", ""),
                "start_frame": key["start"],
                "middle_frame": key["middle"],
                "end_frame": key["end"],
                "contact_sheet": sheet_name,
                "label_match": "",
                "issue": "",
                "use_range_sec": "",
                "review_result": "",
            }
        )
        if count % 20 == 0 or count == total_clips:
            print(f"  extracted: {count}/{total_clips}")

    # ---------- 모아보기 저장 ----------
    for (subject, action), rows in sorted(sheet_rows.items()):
        if not any(t is not None for r in rows for t in r["thumbs"]):
            continue
        rows = sorted(rows, key=lambda r: r["take_id"])
        sheet = build_sheet(rows, args.thumb_width, f"{subject} {action}")
        cv2.imwrite(
            str(sheet_dir / f"contact_{subject}_{action}.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )

    # ---------- 검수표 저장 (채운 내용 보호) ----------
    review = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    if review_path.exists() and not args.overwrite_review:
        review_note = (
            "이미 있음 → 덮어쓰지 않음 (--overwrite-review로 새로 만들 수 있음)"
        )
    else:
        review.to_csv(review_path, index=False, encoding="utf-8-sig")
        review_note = "새로 만듦"

    jpg_count = len(list(output_dir.glob("*.jpg")))
    sheet_count = len(list(sheet_dir.glob("contact_*.jpg")))
    print()
    print(f"대표 JPG  : {jpg_count}장 → {output_dir}")
    print(f"모아보기  : {sheet_count}장 → {sheet_dir}")
    print(f"검수표    : {review_path} ({review_note})")
    flagged = review[review["manifest_status"] != "PASS"]
    if not flagged.empty:
        print()
        print("manifest에서 PASS가 아닌 clip (모아보기에 빨간 글씨로 표시)")
        print(
            flagged[["clip_id", "manifest_status", "manifest_reason"]].to_string(
                index=False
            )
        )
    print()
    print("Team01 keyframes:", "FAIL " + ", ".join(failed) if failed else "READY")
    print("※ JPG에는 조원 얼굴이 있습니다. Git·팀 밖 공유 금지.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())