from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
from pathlib import Path
import cv2


# ② 프로젝트 경로 및 이미지 저장 위치 설정
ROOT = Path(__file__).resolve().parents[1]

SAVE_DIR = ROOT / "data" / "day01" / "ip_webcam"


# ③ 터미널에서 IP Webcam 주소와 실행 옵션 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    # 스마트폰 카메라의 영상 스트림 주소
    parser.add_argument(
        "--url",
        required=True,
        help="IP Webcam video URL",
    )

    # 첫 Frame 한 장만 저장하는 연결 테스트 옵션
    parser.add_argument(
        "--one-frame",
        action="store_true",
        help="첫 Frame 한 장만 저장하고 종료",
    )

    return parser.parse_args()


# ④ 촬영한 Frame을 JPG 이미지로 저장하는 공통 함수
def save_image(
    frame,
    filename: str,
) -> Path:

    # 저장 폴더가 없으면 생성
    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 저장할 이미지 경로 만들기
    output_path = SAVE_DIR / filename

    # Frame을 JPG 파일로 저장
    saved = cv2.imwrite(
        str(output_path),
        frame,
    )

    # 이미지 저장에 실패하면 오류 발생
    if not saved:
        raise RuntimeError(
            f"이미지를 저장하지 못했습니다: {output_path}"
        )

    print(f"Saved: {output_path}")

    return output_path


def main() -> None:

    # ⑤ IP Webcam 연결 및 Frame 수신
    args = parse_args()

    # 스마트폰 IP Webcam 영상 연결
    capture = cv2.VideoCapture(args.url)

    if not capture.isOpened():
        raise RuntimeError(
            f"IP Webcam에 연결할 수 없습니다: {args.url}"
        )

    # 영상 버퍼 크기 설정 시도
    capture.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    # 첫 번째 Frame 읽기
    success, frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError(
            "IP Webcam의 첫 Frame을 읽지 못했습니다."
        )

    # 수신한 영상의 해상도 확인
    height, width = frame.shape[:2]

    print(
        f"Connected: {width} x {height}"
    )

    # ⑥ 첫 Frame 한 장만 저장하는 연결 테스트
    if args.one_frame:
        save_image(
            frame,
            "connection_test.jpg",
        )

        capture.release()
        return

    # ⑦ 촬영할 자세와 키보드 조작 방법 안내
    print("1 : 정면 자세 저장")
    print("2 : 양팔 들기 저장")
    print("3 : 먼 거리 자세 저장")
    print("Q : 종료")

    # ⑧ 실시간 카메라 영상 화면에 표시
    while True:

        # 원본 Frame을 복사하여 화면 표시용으로 사용
        display_frame = frame.copy()

        # 화면 상단에 키보드 사용방법 표시
        cv2.putText(
            display_frame,
            "1: Front  2: Arms Up  3: Far  Q: Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # 실시간 영상 화면 출력
        cv2.imshow(
            "Subject10 IP Webcam",
            display_frame,
        )

        # 키보드 입력 확인
        key = cv2.waitKey(1) & 0xFF

        # ⑨ 키보드 입력에 따라 이미지 저장 또는 종료
        if key == ord("q"):
            break

        # 1번: 정면 자세 저장
        if key == ord("1"):
            save_image(
                frame,
                "front.jpg",
            )

        # 2번: 양팔 들기 자세 저장
        if key == ord("2"):
            save_image(
                frame,
                "arms_up.jpg",
            )

        # 3번: 먼 거리 자세 저장
        if key == ord("3"):
            save_image(
                frame,
                "far.jpg",
            )

        # 다음 Frame 읽기
        success, frame = capture.read()

        if not success:
            print(
                "다음 Frame을 읽지 못했습니다."
            )
            break

    # ⑩ 카메라 연결 해제 및 화면 종료
    capture.release()
    cv2.destroyAllWindows()


# 프로그램 실행 시작
if __name__ == "__main__":
    main()