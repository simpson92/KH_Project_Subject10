from __future__ import annotations

# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
# pathlib : 파일과 폴더 경로 처리
# cv2     : 이미지 읽기
# numpy   : 이미지 픽셀 비교

from pathlib import Path

import cv2
import numpy as np


# ==================================================
# [02] 프로젝트 및 이미지 폴더 경로 설정
# ==================================================
# 현재 스크립트의 상위 폴더를 기준으로
# 프로젝트 ROOT와 3일차 이미지 폴더를 찾습니다.

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "day03" / "images"


# ==================================================
# [03] 필수 이미지 8개 파일명 등록
# ==================================================
# 아래 이미지가 모두 있어야 실습을 진행할 수 있습니다.

REQUIRED_IMAGES = [
    "arms_up_person_a.jpg",
    "arms_up_person_b.jpg",
    "arms_up_near.jpg",
    "arms_up_far.jpg",
    "left_arm_up.jpg",
    "right_arm_up.jpg",
    "sit_pose.jpg",
    "bend_pose.jpg",
]


# ==================================================
# [04] 이미지 파일 존재 여부 확인
# ==================================================
# filename을 전달받아 전체 경로를 만듭니다.
# 파일이 없다면 MISSING 메시지를 반환합니다.

def read_image(filename: str):

    path = IMAGE_DIR / filename

    if not path.exists():
        return None, f"[MISSING] {filename}"

    # ==============================================
    # [05] OpenCV로 이미지 읽기
    # ==============================================
    # 파일이 있어도 손상되었다면
    # OpenCV가 이미지를 읽지 못할 수 있습니다.

    image = cv2.imread(str(path))

    if image is None:
        return None, f"[UNREADABLE] {filename}"

    # ==============================================
    # [06] 이미지 크기 확인 및 결과 반환
    # ==============================================
    # shape[:2]는 높이와 너비를 반환합니다.
    # 정상 이미지와 OK 메시지를 함께 반환합니다.

    height, width = image.shape[:2]

    return image, (
        f"[OK] {filename:22s} "
        f"{width} x {height}"
    )


def main() -> None:

    # ==============================================
    # [07] 검사 결과를 저장할 변수 준비
    # ==============================================
    # loaded : 정상적으로 읽은 이미지 저장
    # errors : 문제가 있는 이미지 파일명 저장

    loaded = {}
    errors = []

    print(f"Image directory: {IMAGE_DIR}")
    print()

    # ==============================================
    # [08] 필수 이미지 8개 반복 검사
    # ==============================================
    # read_image()를 호출합니다.
    # 오류 이미지는 errors에 저장하고,
    # 정상 이미지는 loaded에 저장합니다.

    for filename in REQUIRED_IMAGES:

        image, message = read_image(filename)

        print(message)

        if image is None:
            errors.append(filename)
        else:
            loaded[filename] = image

    # ==============================================
    # [09] 이미지 오류가 있으면 실행 중단
    # ==============================================
    # errors 리스트에 파일명이 하나라도 있다면
    # 이후 실습을 진행할 수 없으므로 중단합니다.

    if errors:
        raise RuntimeError(
            "3일차 입력 이미지의 파일명과 경로를 다시 확인하세요."
        )

    # ==============================================
    # [10] 비교할 이미지 3개 선택
    # ==============================================
    # Near와 Far는 화면 속 사람 크기를 비교합니다.
    # Person B와 Near는 동일 원본 관계를 확인합니다.

    near = loaded["arms_up_near.jpg"]

    far = loaded["arms_up_far.jpg"]

    person_b = loaded["arms_up_person_b.jpg"]

    print()
    print("Comparison checks")

    # ==============================================
    # [11] Near와 Far 해상도 비교
    # ==============================================
    # 두 이미지의 높이와 너비가 같은지 확인합니다.
    # 같으면 True, 다르면 False입니다.

    same_canvas = near.shape[:2] == far.shape[:2]

    print(
        "near / far same canvas :",
        same_canvas,
    )

    # ==============================================
    # [12] Person B와 Near 해상도 비교
    # ==============================================
    # 두 이미지의 높이와 너비를 비교합니다.

    same_person_b_size = (
        near.shape[:2]
        == person_b.shape[:2]
    )

    print(
        "person_b / near same size:",
        same_person_b_size,
    )

    # ==============================================
    # [13] Person B와 Near 픽셀 비교
    # ==============================================
    # 이미지 배열 크기가 같을 때만
    # np.array_equal()로 모든 픽셀을 비교합니다.
    #
    # 모든 픽셀이 같으면 True,
    # 하나라도 다르면 False입니다.

    same_pixels = False

    if near.shape == person_b.shape:

        same_pixels = np.array_equal(
            near,
            person_b,
        )

    print(
        "person_b / near identical :",
        same_pixels,
    )

    # ==============================================
    # [14] Near와 Far 해상도 최종 검사
    # ==============================================
    # 두 이미지의 전체 화면 크기가 다르면
    # 공정한 비교를 위해 실행을 중단합니다.

    if not same_canvas:
        raise RuntimeError(
            "arms_up_near.jpg와 arms_up_far.jpg의 "
            "최종 이미지 해상도를 같게 맞추세요."
        )

    # ==============================================
    # [15] Person B와 Near 해상도 경고
    # ==============================================
    # 두 이미지 크기가 다르면 WARNING을 출력합니다.
    # 경고이므로 프로그램은 중단하지 않습니다.

    if not same_person_b_size:
        print(
            "[WARNING] person_b와 near의 "
            "이미지 크기가 다릅니다."
        )

    # ==============================================
    # [16] Person B와 Near 픽셀 확인 안내
    # ==============================================
    # 모든 픽셀이 같지 않으면 INFO를 출력합니다.
    # 원본 관계를 확인하도록 안내하지만
    # 실행을 중단하지는 않습니다.

    if not same_pixels:
        print(
            "[INFO] person_b와 near가 픽셀 단위로 "
            "완전히 같지는 않습니다. 같은 원본을 "
            "복사한 파일인지 확인하세요."
        )

    # ==============================================
    # [17] 모든 검사 완료
    # ==============================================
    # 필수 이미지가 정상적으로 읽혔고
    # Near와 Far의 해상도 조건도 통과했습니다.
    #
    # 이제 3일차 관절각도와 정규화 실습을
    # 진행할 준비가 완료되었습니다.

    print()
    print("Day03 image data: READY")


# ==================================================
# 프로그램 실행 시작점
# ==================================================
# 이 파일을 직접 실행하면 main()을 호출합니다.

if __name__ == "__main__":
    main()