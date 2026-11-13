from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
from pathlib import Path
import cv2


# ② 터미널에서 이미지 경로 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="읽을 이미지 파일 경로",
    )

    return parser.parse_args()


# ③ 이미지 파일 경로 확인
def main() -> None:
    args = parse_args()

    image_path = Path(args.source)

    # 입력한 경로에 파일이 없으면 오류 발생
    if not image_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {image_path}"
        )

    # ④ OpenCV로 이미지 읽기
    image = cv2.imread(str(image_path))

    # 이미지 읽기에 실패하면 오류 발생
    if image is None:
        raise RuntimeError(
            f"OpenCV가 이미지를 읽지 못했습니다: {image_path}"
        )

    # ⑤ 이미지의 크기와 정보 확인
    height, width = image.shape[:2]

    print(f"Image path : {image_path}")
    print(f"Width      : {width}")
    print(f"Height     : {height}")
    print(f"Channels   : {image.shape[2]}")
    print(f"Shape      : {image.shape}")
    print(f"DataType   : {image.dtype}")


# ⑥ 프로그램 실행 시작
if __name__ == "__main__":
    main()