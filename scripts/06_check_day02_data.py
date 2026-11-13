from __future__ import annotations

# ① 필요한 라이브러리 불러오기
from pathlib import Path
import cv2


# ② 프로젝트 경로 및 2일차 이미지 폴더 지정
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "day02" / "images"


# ③ 2일차 실습에 필요한 이미지 6장 지정
REQUIRED_IMAGES = [
    "front.jpg",
    "arms_up.jpg",
    "side.jpg",
    "partial_body.jpg",
    "occlusion.jpg",
    "distance_far.jpg",
]


def main() -> None:

    # ④ 이미지 검사 결과를 저장할 목록 준비
    missing = []      # 파일이 없는 이미지
    unreadable = []   # 파일은 있지만 읽을 수 없는 이미지

    print(f"Image directory: {IMAGE_DIR}")
    print()

    # ⑤ 이미지 6장을 순서대로 검사
    for filename in REQUIRED_IMAGES:
        image_path = IMAGE_DIR / filename

        # 이미지 파일이 존재하는지 확인
        if not image_path.exists():
            missing.append(filename)
            print(f"[MISSING] {filename}")
            continue

        # OpenCV로 이미지 읽기
        image = cv2.imread(str(image_path))

        # 파일은 있지만 이미지를 읽을 수 없는 경우
        if image is None:
            unreadable.append(filename)
            print(f"[UNREADABLE] {filename}")
            continue

        # 정상 이미지의 너비와 높이 확인
        height, width = image.shape[:2]

        print(
            f"[OK] {filename:18s} "
            f"{width} x {height}"
        )

    # ⑥ 이미지 검사 결과 확인
    print()

    if missing or unreadable:

        # 존재하지 않는 이미지 목록 출력
        if missing:
            print("Missing:", ", ".join(missing))

        # 읽을 수 없는 이미지 목록 출력
        if unreadable:
            print("Unreadable:", ", ".join(unreadable))

        # 하나라도 문제가 있으면 오류 발생
        raise RuntimeError(
            "2일차 입력 이미지를 다시 확인하세요."
        )

    # ⑦ 모든 이미지가 정상이면 준비 완료 출력
    print("Day02 image data: READY")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()