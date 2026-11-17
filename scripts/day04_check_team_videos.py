from __future__ import annotations

# ==================================================
# day04_check_team_videos.py
# 1조 영상 전체 검사 — 4일차 팀 확장 STEP 2
# ==================================================
# 교재 19_check_day04_data.py는 "정해진 6개 영상의 존재 + 첫 Frame"만 봅니다.
# 팀 확장 문서 STEP 2는 이 검사를 manifest의 모든 MP4로 넓히고
# "모든 MP4의 정상 재생 구간"까지 확인하라고 합니다. 이 스크립트가 그 일을 합니다.
# (19번 원본은 그대로 둡니다.)
#
# 검사 항목 (STEP 2 표)
#   1. 파일   : 영상 존재, clip_id · video_file 중복
#   2. 재생   : OpenCV로 열림 + 끝까지 읽은 실제 Frame 수 vs 헤더 Frame 수
#   3. 촬영   : FPS · Frame 수 · 재생시간 · 해상도 · 세로/가로
#   4. 라벨   : clip_id(T01_S##_A##_T##) ↔ manifest의 subject/action/take 칸,
#               action_id가 team01_action_classes.csv에 있는지
#   5. 수량   : 사람 × 동작별 clip 수
#
# 판정
#   FAIL : 파일 없음 · 열리지 않음 · Frame을 하나도 못 읽음 · FPS 없음 · clip_id 형식 오류
#   WARN : 헤더와 실제 Frame 수가 다름 · manifest의 fps/frame/해상도 값과 다름 ·
#          clip_id와 manifest 칸 불일치 · 정의표에 없는 action_id · 중복
#   OK   : 위 문제가 없음
#   ※ 세로 영상(스모크 1080×1920)은 기록만 하고 문제로 보지 않습니다.
#   ※ 정답 Label이 실제 동작과 맞는지는 숫자로 알 수 없습니다 → STEP 3 대표 Frame에서 사람이 확인.
#
# 실행
#   python scripts/day04_check_team_videos.py
#   python scripts/day04_check_team_videos.py --limit 3      # 앞의 3개만 빠르게
#
# 출력
#   reports/team_projects/team01/tables/video_inventory.csv  (실명 없음, Git 제외)
import argparse
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV = ROOT / "configs" / "team01_clip_manifest.csv"
ACTION_CLASSES_CSV = ROOT / "configs" / "team01_action_classes.csv"
VIDEO_DIR = ROOT / "data" / "team_projects" / "team01" / "rgb" / "videos"
OUTPUT_CSV = (
    ROOT / "reports" / "team_projects" / "team01" / "tables" / "video_inventory.csv"
)

CLIP_ID_PATTERN = re.compile(
    r"^(?P<team>T\d{2})_(?P<subject>S\d{2})_(?P<action>A\d{2})_(?P<take>T\d{2})$"
)

FPS_TOLERANCE = 0.05  # manifest fps와 비교할 때 허용 오차
DURATION_DIGITS = 2

OUTPUT_COLUMNS = [
    "clip_id",
    "subject_id",
    "action_id",
    "take_id",
    "video_file",
    "exists",
    "open_ok",
    "fps",
    "header_frames",
    "decoded_frames",
    "duration_sec",
    "width",
    "height",
    "orientation",
    "status",
    "issues",
]


# ==================================================
# [01] 실행 옵션
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--action-classes", default=str(ACTION_CLASSES_CSV))
    parser.add_argument("--video-dir", default=str(VIDEO_DIR))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument(
        "--limit", type=int, default=0, help="0이면 전체, 양수이면 앞에서 N개만"
    )
    return parser.parse_args()


# ==================================================
# [02] 영상 한 개 열어서 끝까지 읽기
# ==================================================
def probe_video(path: Path) -> dict:
    """헤더 정보 + 실제로 끝까지 읽은 Frame 수를 돌려줍니다."""
    info = {
        "exists": path.exists(),
        "open_ok": False,
        "fps": 0.0,
        "header_frames": 0,
        "decoded_frames": 0,
        "width": 0,
        "height": 0,
    }
    if not info["exists"]:
        return info

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return info

    info["open_ok"] = True
    info["fps"] = float(capture.get(cv2.CAP_PROP_FPS))
    info["header_frames"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    info["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 첫 Frame만이 아니라 끝까지 디코딩해서 "정상 재생 구간"을 확인합니다.
    decoded = 0
    while capture.grab():
        decoded += 1
    capture.release()
    info["decoded_frames"] = decoded
    return info


def orientation_of(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return ""
    if height > width:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


# ==================================================
# [03] 한 행 판정 (FAIL / WARN / OK)
# ==================================================
def judge(row: dict, info: dict, action_ids: set[str]) -> tuple[str, list[str]]:
    fails: list[str] = []
    warns: list[str] = []

    # 3-1. clip_id 형식과 manifest 칸 일치
    match = CLIP_ID_PATTERN.match(str(row.get("clip_id", "")))
    if match is None:
        fails.append("clip_id 형식 오류")
    else:
        for key, column in (
            ("subject", "subject_id"),
            ("action", "action_id"),
            ("take", "take_id"),
        ):
            if match.group(key) != str(row.get(column, "")):
                warns.append(f"{column} 불일치({match.group(key)}≠{row.get(column)})")
        if f"{row.get('clip_id')}.mp4" != str(row.get("video_file", "")):
            warns.append("video_file 이름이 clip_id와 다름")

    if str(row.get("action_id", "")) not in action_ids:
        warns.append("정의표에 없는 action_id")

    # 3-2. 파일·재생
    if not info["exists"]:
        fails.append("파일 없음")
    elif not info["open_ok"]:
        fails.append("열리지 않음")
    else:
        if info["fps"] <= 0:
            fails.append("FPS 없음")
        if info["decoded_frames"] == 0:
            fails.append("Frame을 읽지 못함")
        elif info["decoded_frames"] != info["header_frames"]:
            warns.append(
                f"Frame 수 차이(헤더 {info['header_frames']}·실제 {info['decoded_frames']})"
            )

        # 3-3. manifest에 적힌 값과 비교 (값이 있을 때만)
        expected_fps = pd.to_numeric(row.get("fps"), errors="coerce")
        if pd.notna(expected_fps) and abs(info["fps"] - expected_fps) > FPS_TOLERANCE:
            warns.append(f"fps가 manifest와 다름({expected_fps}→{info['fps']:.2f})")
        for column, key in (
            ("frame_count", "header_frames"),  # manifest는 헤더 값을 기록함
            ("width", "width"),
            ("height", "height"),
        ):
            expected = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(expected) and int(expected) != int(info[key]):
                warns.append(f"{column}이 manifest와 다름({int(expected)}→{info[key]})")

    if fails:
        return "FAIL", fails + warns
    if warns:
        return "WARN", warns
    return "OK", []


# ==================================================
# [04] 중복 검사 (clip_id · video_file)
# ==================================================
def duplicated_values(manifest: pd.DataFrame, column: str) -> set[str]:
    values = manifest[column].astype(str)
    return set(values[values.duplicated(keep=False)])


# ==================================================
# [05] 실행
# ==================================================
def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    video_dir = Path(args.video_dir)
    output_path = Path(args.output)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not video_dir.exists():
        raise FileNotFoundError(
            f"{video_dir}\n먼저 python scripts/day04_make_clip_manifest.py --apply --link"
        )

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if args.limit > 0:
        manifest = manifest.head(args.limit)
    action_ids = set(
        pd.read_csv(args.action_classes, dtype=str, keep_default_na=False)["action_id"]
    )

    dup_clip = duplicated_values(manifest, "clip_id")
    dup_file = duplicated_values(manifest, "video_file")

    rows = []
    total = len(manifest)
    for index, row in enumerate(manifest.to_dict("records"), start=1):
        info = probe_video(video_dir / row["video_file"])
        status, issues = judge(row, info, action_ids)
        if row["clip_id"] in dup_clip:
            issues.append("clip_id 중복")
        if row["video_file"] in dup_file:
            issues.append("video_file 중복")
        if status == "OK" and issues:
            status = "WARN"

        duration = info["decoded_frames"] / info["fps"] if info["fps"] > 0 else 0.0
        rows.append(
            {
                "clip_id": row["clip_id"],
                "subject_id": row["subject_id"],
                "action_id": row["action_id"],
                "take_id": row["take_id"],
                "video_file": row["video_file"],
                "exists": int(info["exists"]),
                "open_ok": int(info["open_ok"]),
                "fps": round(info["fps"], 3),
                "header_frames": info["header_frames"],
                "decoded_frames": info["decoded_frames"],
                "duration_sec": round(duration, DURATION_DIGITS),
                "width": info["width"],
                "height": info["height"],
                "orientation": orientation_of(info["width"], info["height"]),
                "status": status,
                "issues": " | ".join(issues),
            }
        )
        if index % 20 == 0 or index == total:
            print(f"  checked: {index}/{total}")

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---------- 화면 요약 ----------
    print()
    print("판정:", result["status"].value_counts().to_dict())
    print()
    print("사람 × 동작별 clip 수")
    print(pd.crosstab(result["subject_id"], result["action_id"]).to_string())
    print()
    readable = result[result["open_ok"] == 1]
    if not readable.empty:
        print("촬영 정보")
        print(f"  해상도   : {readable.groupby(['width', 'height']).size().to_dict()}")
        print(f"  방향     : {readable['orientation'].value_counts().to_dict()}")
        print(f"  FPS      : {readable['fps'].value_counts().to_dict()}")
        print(
            "  Frame 수 : "
            f"min {readable['decoded_frames'].min()} · "
            f"median {int(readable['decoded_frames'].median())} · "
            f"max {readable['decoded_frames'].max()} "
            f"(합계 {readable['decoded_frames'].sum()})"
        )
        print(
            "  재생시간 : "
            f"{readable['duration_sec'].min():.2f} ~ {readable['duration_sec'].max():.2f}초"
        )

    problems = result[result["status"] != "OK"]
    if not problems.empty:
        print()
        print("확인이 필요한 clip")
        print(problems[["clip_id", "status", "issues"]].to_string(index=False))

    print()
    print(f"Saved: {output_path}")
    fail_count = int((result["status"] == "FAIL").sum())
    print("Team01 video check:", "FAIL" if fail_count else "READY")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())