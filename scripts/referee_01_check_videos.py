from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2


# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.referee_utils import (
    MANIFEST_CSV,
    RAW_DIR,
    list_videos,
    parse_name,
)


def main() -> None:

    # ③ 영상 목록 가져오기
    videos = list_videos()

    if not videos:
        raise RuntimeError(f"영상이 없습니다: {RAW_DIR}")

    rows = []

    # ④ 영상을 하나씩 열어 기본 정보 확인
    for path in videos:
        info = parse_name(path.stem)

        row = {
            "video": str(path.relative_to(ROOT)),
            "stem": path.stem,
            "subject": info["subject"] if info else "",
            "action": info["action"] if info else "",
            "take": info["take"] if info else "",
            "fps": 0.0,
            "frame_count": 0,
            "duration_sec": 0.0,
            "width": 0,
            "height": 0,
            "orientation": "",
            "status": "OK",
        }

        # 파일명이 S01_ufoul_T01 형식이 아니면 표시만 하고 계속
        if info is None:
            row["status"] = "NAME_ERROR"
            rows.append(row)
            continue

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            row["status"] = "UNREADABLE"
            rows.append(row)
            continue

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # ⑤ 첫 Frame을 실제로 읽어 해상도 확인
        #    OpenCV는 폰 영상의 회전 정보를 자동 적용하므로
        #    실제 Frame 크기로 세로/가로를 판단
        success, frame = capture.read()
        capture.release()

        if not success:
            row["status"] = "UNREADABLE"
            rows.append(row)
            continue

        height, width = frame.shape[:2]

        row.update(
            {
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "duration_sec": round(frame_count / fps, 2) if fps > 0 else 0.0,
                "width": width,
                "height": height,
                "orientation": "portrait" if height > width else "landscape",
            }
        )

        rows.append(row)

    # ⑥ 영상 목록 CSV 저장
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ⑦ 요약 출력
    ok_rows = [row for row in rows if row["status"] == "OK"]
    bad_rows = [row for row in rows if row["status"] != "OK"]

    print(f"전체 영상: {len(rows)}개 / 정상: {len(ok_rows)}개 / 문제: {len(bad_rows)}개")
    print()

    # 동작 × 사람별 take 수
    table = defaultdict(Counter)
    for row in ok_rows:
        table[row["action"]][row["subject"]] += 1

    subjects = sorted({row["subject"] for row in ok_rows})

    print("동작별 take 수")
    print("action".ljust(10) + "".join(s.rjust(5) for s in subjects))
    for action in sorted(table):
        print(
            action.ljust(10)
            + "".join(str(table[action][s]).rjust(5) for s in subjects)
        )
    print()

    # FPS·해상도·방향 종류 (섞여 있으면 여기서 보임)
    print("FPS 종류:", dict(Counter(row["fps"] for row in ok_rows)))
    print(
        "해상도 종류:",
        dict(Counter(f"{row['width']}x{row['height']}" for row in ok_rows)),
    )
    print("방향:", dict(Counter(row["orientation"] for row in ok_rows)))

    if ok_rows:
        durations = [row["duration_sec"] for row in ok_rows]
        print(f"길이: 최소 {min(durations):.2f}초 / 최대 {max(durations):.2f}초")

    if bad_rows:
        print()
        print("확인이 필요한 영상")
        for row in bad_rows:
            print(f"  [{row['status']}] {row['video']}")

    print()
    print(f"Manifest saved: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()