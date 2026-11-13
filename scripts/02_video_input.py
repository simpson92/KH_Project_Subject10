from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
from pathlib import Path
import cv2


# ② 터미널에서 MP4 영상 경로 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="읽을 MP4 파일 경로",
    )

    return parser.parse_args()


def main() -> None:

    # ③ 영상 파일 경로 확인
    args = parse_args()

    video_path = Path(args.source)

    # 입력한 경로에 파일이 없으면 오류 발생
    if not video_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {video_path}"
        )

    # ④ OpenCV로 MP4 영상 열기
    capture = cv2.VideoCapture(str(video_path))

    # 영상을 열지 못하면 오류 발생
    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path}"
        )

    # ⑤ 영상의 기본 정보 가져오기
    fps = capture.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # ⑥ 영상을 Frame 단위로 끝까지 읽기
    frame_number = 0

    while True:
        success, _frame = capture.read()

        # 더 이상 읽을 Frame이 없으면 반복 종료
        if not success:
            break

        # 정상적으로 읽은 Frame 수 증가
        frame_number += 1

    # 영상 사용이 끝났으므로 자원 해제
    capture.release()

    # ⑦ 영상 정보와 재생시간 출력
    print(f"Video       : {video_path}")
    print(f"FPS         : {fps:.2f}")
    print(f"Frames      : {total_frames}")
    print(f"Read Frames : {frame_number}")
    print(f"Resolution  : {width} x {height}")

    # FPS가 정상적으로 확인되면 재생시간 계산
    if fps > 0:
        duration = total_frames / fps
        print(f"Duration    : {duration:.2f} sec")
    else:
        print("Duration    : FPS 정보를 읽지 못해 계산하지 않음")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()