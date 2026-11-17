from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# csv  : 동영상 검사 결과를 CSV 파일로 저장합니다.
# Path : 동영상 파일과 결과 파일의 경로를 관리합니다.
# cv2  : OpenCV로 MP4를 열고 영상 정보를 확인합니다.

import csv
from pathlib import Path

import cv2


# ==================================================
# ② 프로젝트 경로와 검사할 동영상 목록 설정
# ==================================================
# ROOT:
# 현재 Python 파일의 상위 프로젝트 폴더입니다.
#
# VIDEO_DIR:
# 4일차에 사용할 MP4 파일이 저장된 폴더입니다.
#
# OUTPUT_PATH:
# 검사 결과를 저장할 CSV 파일의 경로입니다.
#
# REQUIRED_VIDEOS:
# 반드시 준비되어 있어야 하는 6개 동영상입니다.

ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = (
    ROOT
    / "data"
    / "day04"
    / "videos"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day04"
    / "tables"
    / "video_metadata.csv"
)

REQUIRED_VIDEOS = [
    "bend_return.mp4",
    "leg_raise_lower.mp4",
    "walk_turn_walk.mp4",
    "drink_return.mp4",
    "sit_stand.mp4",
    "wave.mp4",
]


# ==================================================
# ③ 동영상을 열고 정상적으로 열렸는지 확인
# ==================================================
# read_video_info()는 동영상 하나를 검사하는 함수입니다.
#
# OpenCV VideoCapture로 MP4 파일을 엽니다.
#
# 파일을 열지 못하면 open_ok=0으로 기록합니다.
# FPS, Frame 수, 해상도 등은 기본값 0으로 반환합니다.
#
# 정상적으로 열리면 다음 단계에서
# 실제 동영상 정보를 읽습니다.

def read_video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        return {
            "open_ok": 0,
            "read_ok": 0,
            "fps": 0.0,
            "frames": 0,
            "duration_sec": 0.0,
            "width": 0,
            "height": 0,
        }

    # ==============================================
    # ④ 동영상의 FPS, Frame 수, 해상도 가져오기
    # ==============================================
    # FPS:
    # 1초에 포함된 Frame 수입니다.
    #
    # FRAME_COUNT:
    # 영상의 전체 Frame 수입니다.
    #
    # FRAME_WIDTH:
    # 영상의 가로 크기(Pixel)입니다.
    #
    # FRAME_HEIGHT:
    # 영상의 세로 크기(Pixel)입니다.
    #
    # 예:
    # FPS=25, Frames=100, Width=640, Height=480

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # ==============================================
    # ⑤ 첫 Frame 읽기 및 재생시간 계산
    # ==============================================
    # 파일이 열리더라도 실제 Frame을 읽지 못하는
    # 경우가 있을 수 있습니다.
    #
    # 첫 번째 Frame을 읽어 정상 여부를 확인합니다.
    #
    # 정상적으로 읽었다면 read_ok=1입니다.
    # 읽지 못했다면 read_ok=0입니다.
    #
    # 검사가 끝나면 VideoCapture를 해제합니다.
    #
    # 재생시간은 다음 식으로 계산합니다.
    #
    # duration_sec = 전체 Frame 수 / FPS
    #
    # FPS가 0보다 크지 않으면 재생시간은
    # 0으로 설정하여 0으로 나누는 오류를 방지합니다.
    #
    # 검사 결과를 딕셔너리로 반환합니다.

    success, first_frame = (
        capture.read()
    )

    read_ok = int(
        success
        and first_frame is not None
    )

    capture.release()

    duration = (
        frames / fps
        if fps > 0
        else 0.0
    )

    return {
        "open_ok": 1,
        "read_ok": read_ok,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_sec": round(
            duration,
            3,
        ),
        "width": width,
        "height": height,
    }


def main() -> None:

    # ==============================================
    # ⑥ 검사 결과와 오류 목록 준비
    # ==============================================
    # rows:
    # 정상적으로 검사된 동영상 정보를 저장합니다.
    #
    # errors:
    # 누락되었거나 검사에 실패한 파일명을 저장합니다.
    #
    # 검사할 동영상 폴더의 위치를 출력합니다.

    rows = []
    errors = []

    print(
        f"Video directory: "
        f"{VIDEO_DIR}"
    )

    print()

    # ==============================================
    # ⑦ 6개 필수 MP4 파일의 존재 여부 확인
    # ==============================================
    # REQUIRED_VIDEOS의 파일을 하나씩 확인합니다.
    #
    # 파일이 없으면 [MISSING]을 출력합니다.
    # errors에 파일명을 추가합니다.
    #
    # continue를 사용하여 현재 파일 검사를
    # 중단하고 다음 동영상으로 이동합니다.
    #
    # 파일이 존재하면 실제 영상 검사를 진행합니다.

    for filename in REQUIRED_VIDEOS:
        path = VIDEO_DIR / filename

        if not path.exists():
            print(
                f"[MISSING] {filename}"
            )

            errors.append(filename)
            continue

        # ==========================================
        # ⑧ 동영상 열기와 첫 Frame 읽기 결과 확인
        # ==========================================
        # read_video_info() 함수를 호출합니다.
        #
        # open_ok=0:
        # 동영상 파일을 열지 못한 상태입니다.
        # [OPEN FAIL]을 출력합니다.
        #
        # read_ok=0:
        # 동영상은 열렸지만 첫 Frame을
        # 정상적으로 읽지 못한 상태입니다.
        # [READ FAIL]을 출력합니다.
        #
        # 실패한 파일은 errors에 추가합니다.
        # 정상인 경우 다음 검사로 이동합니다.

        info = read_video_info(path)

        if not info["open_ok"]:
            print(
                f"[OPEN FAIL] "
                f"{filename}"
            )

            errors.append(filename)
            continue

        if not info["read_ok"]:
            print(
                f"[READ FAIL] "
                f"{filename}"
            )

            errors.append(filename)
            continue

        # ==========================================
        # ⑨ 영상 Metadata가 정상인지 검사
        # ==========================================
        # Metadata는 동영상의 기본 정보입니다.
        #
        # 다음 네 가지 값이 모두 0보다 커야 합니다.
        #
        # FPS
        # Frame 수
        # Width
        # Height
        #
        # 하나라도 0 이하이면 정상적인 영상 정보로
        # 판단하지 않고 [METADATA FAIL]을 출력합니다.

        metadata_ok = (
            info["fps"] > 0
            and info["frames"] > 0
            and info["width"] > 0
            and info["height"] > 0
        )

        if not metadata_ok:
            print(
                f"[METADATA FAIL] "
                f"{filename}"
            )

            errors.append(filename)
            continue

        # ==========================================
        # ⑩ 정상 동영상 정보 출력 및 결과 저장
        # ==========================================
        # 모든 검사를 통과하면 [OK]를 출력합니다.
        #
        # 화면에 다음 정보를 표시합니다.
        #
        # 파일명
        # 해상도
        # FPS
        # 전체 Frame 수
        # 재생시간
        #
        # 정상 동영상 정보만 rows에 추가합니다.
        #
        # **info는 read_video_info()에서 반환한
        # 딕셔너리의 모든 항목을 가져옵니다.

        print(
            f"[OK] {filename:24s} "
            f"{info['width']}x"
            f"{info['height']} "
            f"fps={info['fps']:.3f} "
            f"frames={info['frames']} "
            f"duration="
            f"{info['duration_sec']:.3f}s"
        )

        rows.append(
            {
                "filename": filename,
                **info,
            }
        )

    # ==============================================
    # ⑪ 전체 검사 결과에 오류가 있는지 확인
    # ==============================================
    # 6개 동영상의 검사가 모두 끝난 후 실행합니다.
    #
    # errors에 파일명이 하나라도 있다면
    # 어떤 파일에 문제가 있는지 오류를 발생시킵니다.
    #
    # 오류가 있는 경우 프로그램이 중단되므로
    # 새로운 CSV 파일을 저장하지 않습니다.
    #
    # 오류가 없다면 CSV 저장 단계로 이동합니다.

    if errors:
        raise RuntimeError(
            "4일차 입력 동영상을 "
            "다시 확인하세요: "
            + ", ".join(errors)
        )

    # ==============================================
    # ⑫ 검사 결과를 CSV 파일로 저장
    # ==============================================
    # CSV 파일을 저장할 폴더가 없으면 생성합니다.
    #
    # csv.DictWriter를 이용하여
    # 정상 동영상 6개의 정보를 표로 저장합니다.
    #
    # 첫 번째 행에는 컬럼명을 기록합니다.
    # 그 아래에는 각 동영상의 정보를 저장합니다.
    #
    # utf-8-sig:
    # Excel에서도 한글을 읽기 쉽게 저장합니다.
    #
    # 저장 위치:
    # reports/day04/tables/video_metadata.csv

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "open_ok",
                "read_ok",
                "fps",
                "frames",
                "duration_sec",
                "width",
                "height",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # ==============================================
    # ⑬ 최종 저장 위치와 검사 완료 메시지 출력
    # ==============================================
    # 6개 동영상이 모두 검사를 통과하고
    # CSV 파일까지 저장되었다면
    # 최종 결과를 화면에 출력합니다.
    #
    # Saved:
    # CSV 파일이 저장된 위치입니다.
    #
    # Day04 video data: READY
    # 4일차 입력 동영상의 기본 검사가
    # 모두 완료되었다는 의미입니다.

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "Day04 video data: READY"
    )


# ==================================================
# 프로그램 실행 시작
# ==================================================
# Python 파일을 직접 실행했을 때
# main() 함수를 호출합니다.

if __name__ == "__main__":
    main()