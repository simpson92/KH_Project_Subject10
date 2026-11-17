from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse : 터미널 실행 옵션을 입력받습니다.
# csv      : Action 설정 CSV 파일을 읽습니다.
# Path     : 입력 영상과 출력 이미지 경로를 관리합니다.
# cv2      : 동영상을 읽고 특정 Frame을 JPG로 저장합니다.

import argparse
import csv
from pathlib import Path

import cv2


# ==================================================
# ② 프로젝트 경로와 기본 입출력 폴더 설정
# ==================================================
# ROOT:
# 현재 Python 파일의 상위 프로젝트 폴더입니다.
#
# DEFAULT_ACTION_FILE:
# Action ID, Label, MP4 파일명이 기록된 설정 파일입니다.
#
# DEFAULT_VIDEO_DIR:
# 6개의 Action MP4가 저장된 폴더입니다.
#
# DEFAULT_OUTPUT_DIR:
# 추출한 대표 Frame JPG를 저장할 폴더입니다.
#
# 이후 터미널 옵션을 이용해 이 경로들을
# 다른 경로로 변경할 수도 있습니다.

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACTION_FILE = (
    ROOT
    / "configs"
    / "action_classes.csv"
)

DEFAULT_VIDEO_DIR = (
    ROOT
    / "data"
    / "day04"
    / "videos"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "reports"
    / "day04"
    / "keyframes"
)


# ==================================================
# ③ 터미널 실행 옵션 설정
# ==================================================
# --action-file:
# Action 설정 CSV 파일의 위치입니다.
#
# --video-dir:
# MP4 동영상이 저장된 폴더입니다.
#
# --output-dir:
# 추출한 JPG를 저장할 폴더입니다.
#
# 옵션을 입력하지 않으면 위에서 정의한
# 기본 경로를 사용합니다.
#
# parse_args()는 최종 실행 옵션을 반환합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action-file",
        default=str(
            DEFAULT_ACTION_FILE
        ),
    )

    parser.add_argument(
        "--video-dir",
        default=str(
            DEFAULT_VIDEO_DIR
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    return parser.parse_args()


# ==================================================
# ④ Action 설정 CSV 파일 읽기
# ==================================================
# read_actions()는 Action 설정 파일을 읽는 함수입니다.
#
# action_classes.csv에서 다음 정보를 읽습니다.
#
# action_id
# action_label
# type
# video_file
# description
#
# csv.DictReader:
# CSV의 각 행을 딕셔너리 형태로 읽습니다.
#
# list()를 사용하여 모든 Action을 목록으로 만듭니다.
#
# 예:
# action["action_label"] → "wave"
# action["video_file"]   → "wave.mp4"
#
# 이 목록을 이용하여 어떤 MP4를 처리할지 결정합니다.

def read_actions(
    action_file: Path,
) -> list[dict]:
    with action_file.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return list(
            csv.DictReader(file)
        )


# ==================================================
# ⑤ 특정 Frame을 읽는 함수와 동영상 열기
# ==================================================
# read_frame_at()은 MP4에서 지정된 번호의
# Frame 하나를 가져오는 함수입니다.
#
# video_path:
# 분석할 MP4 파일 경로입니다.
#
# frame_index:
# 가져올 Frame의 번호입니다.
#
# Frame 번호는 0부터 시작합니다.
#
# 예:
# frame_index=0 → 첫 번째 Frame
#
# OpenCV VideoCapture로 영상을 엽니다.
# 열리지 않으면 RuntimeError를 발생시킵니다.

def read_frame_at(
    video_path: Path,
    frame_index: int,
):
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: "
            f"{video_path}"
        )

    # ==============================================
    # ⑥ 지정된 Frame 위치로 이동하여 읽기
    # ==============================================
    # CAP_PROP_POS_FRAMES:
    # OpenCV에서 읽을 Frame 위치를 지정합니다.
    #
    # capture.set()으로 원하는 Frame에 이동합니다.
    # capture.read()로 해당 Frame을 읽습니다.
    #
    # 정상적으로 읽었다면:
    # Frame 이미지 배열을 반환합니다.
    #
    # 읽기에 실패했다면:
    # 다음 단계에서 다른 방법으로 다시 시도합니다.
    #
    # capture.release():
    # 사용한 동영상 객체를 해제합니다.

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, frame = (
        capture.read()
    )

    capture.release()

    if (
        success
        and frame is not None
    ):
        return frame

    # ==============================================
    # ⑦ Frame 직접 이동 실패 시 순차 읽기로 재시도
    # ==============================================
    # 일부 동영상 Codec에서는 특정 Frame으로
    # 직접 이동하여 읽는 기능이 불안정할 수 있습니다.
    #
    # 이 경우 MP4 파일을 처음부터 다시 엽니다.
    #
    # 첫 번째 Frame부터 하나씩 순서대로 읽습니다.
    #
    # 원하는 frame_index에 도달할 때까지 반복합니다.
    #
    # 예:
    # frame_index=42
    # → Frame 0부터 Frame 42까지 순서대로 읽기
    #
    # 도중에 읽기 실패:
    # RuntimeError를 발생시킵니다.
    #
    # 성공하면 마지막에 읽은 Frame을 반환합니다.

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 다시 열 수 없습니다: "
            f"{video_path}"
        )

    frame = None

    for _ in range(
        frame_index + 1
    ):
        success, frame = (
            capture.read()
        )

        if not success:
            capture.release()

            raise RuntimeError(
                "Frame을 읽지 못했습니다: "
                f"{video_path.name} "
                f"frame={frame_index}"
            )

    capture.release()

    return frame


def main() -> None:

    # ==============================================
    # ⑧ 실행 옵션 읽기 및 입출력 경로 확인
    # ==============================================
    # 터미널 실행 옵션을 읽습니다.
    #
    # Action 설정 CSV 경로
    # 동영상 입력 폴더 경로
    # JPG 출력 폴더 경로
    #
    # 세 경로를 Path 객체로 변환합니다.
    #
    # Action 설정 파일이 없다면 오류를 발생시킵니다.
    # 동영상 폴더가 없다면 오류를 발생시킵니다.
    #
    # 출력 폴더는 없을 경우 자동으로 생성합니다.
    #
    # exist_ok=True:
    # 이미 폴더가 있더라도 오류를 발생시키지 않습니다.

    args = parse_args()

    action_file = Path(
        args.action_file
    )

    video_dir = Path(
        args.video_dir
    )

    output_dir = Path(
        args.output_dir
    )

    if not action_file.exists():
        raise FileNotFoundError(
            action_file
        )

    if not video_dir.exists():
        raise FileNotFoundError(
            video_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================
    # ⑨ Action 목록 읽기 및 데이터 존재 확인
    # ==============================================
    # read_actions() 함수를 호출합니다.
    #
    # action_classes.csv에 등록된 Action 정보를
    # 목록 형태로 가져옵니다.
    #
    # Action 정보가 하나도 없다면
    # RuntimeError를 발생시킵니다.
    #
    # 정상적으로 읽었다면
    # 각 Action의 MP4를 순서대로 처리합니다.

    actions = read_actions(
        action_file
    )

    if not actions:
        raise RuntimeError(
            "Action 정의가 없습니다."
        )

    # ==============================================
    # ⑩ 각 Action MP4 열기 및 전체 Frame 수 확인
    # ==============================================
    # CSV에 등록된 Action을 하나씩 가져옵니다.
    #
    # video_file 정보를 이용해 MP4 경로를 만듭니다.
    #
    # MP4 파일이 존재하는지 확인합니다.
    # 없으면 FileNotFoundError를 발생시킵니다.
    #
    # VideoCapture로 동영상을 엽니다.
    # 열리지 않으면 RuntimeError를 발생시킵니다.
    #
    # CAP_PROP_FRAME_COUNT:
    # 동영상의 전체 Frame 수를 가져옵니다.
    #
    # Frame 수가 1보다 작으면 정상적으로
    # 처리할 수 없으므로 오류를 발생시킵니다.

    for action in actions:
        video_path = (
            video_dir
            / action["video_file"]
        )

        if not video_path.exists():
            raise FileNotFoundError(
                video_path
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"영상을 열 수 없습니다: "
                f"{video_path}"
            )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        capture.release()

        if total_frames < 1:
            raise RuntimeError(
                "Frame 수를 확인할 수 "
                f"없습니다: {video_path}"
            )

        # ==========================================
        # ⑪ 시작·중간·끝 Frame 번호 계산
        # ==========================================
        # 각 영상에서 추출할 위치는 세 곳입니다.
        #
        # start:
        # 첫 번째 Frame이므로 0입니다.
        #
        # middle:
        # 전체 Frame 수를 2로 나눈 몫입니다.
        #
        # end:
        # Frame 번호가 0부터 시작하므로
        # 전체 Frame 수에서 1을 뺍니다.
        #
        # 예:
        # 전체 Frame 수 = 85
        #
        # Start  = 0
        # Middle = 85 // 2 = 42
        # End    = 85 - 1 = 84
        #
        # Middle은 시간상 가운데 위치일 뿐,
        # 동작이 가장 크게 발생한 순간은 아닙니다.

        positions = {
            "start": 0,
            "middle": (
                total_frames // 2
            ),
            "end": (
                total_frames - 1
            ),
        }

        # ==========================================
        # ⑫ 대표 Frame 추출 및 JPG 이미지 저장
        # ==========================================
        # positions에 저장된 세 위치를 반복합니다.
        #
        # start
        # middle
        # end
        #
        # Action Label과 위치 이름을 조합하여
        # JPG 파일명을 생성합니다.
        #
        # 예:
        # bend_return_start.jpg
        # bend_return_middle.jpg
        # bend_return_end.jpg
        #
        # read_frame_at()으로 해당 Frame을 읽습니다.
        #
        # cv2.imwrite()를 이용해 JPG로 저장합니다.
        #
        # 저장에 실패하면 RuntimeError를 발생시킵니다.

        for name, frame_index in (
            positions.items()
        ):
            output_path = (
                output_dir
                / (
                    f"{action['action_label']}"
                    f"_{name}.jpg"
                )
            )

            frame = read_frame_at(
                video_path,
                frame_index,
            )

            saved = cv2.imwrite(
                str(output_path),
                frame,
            )

            if not saved:
                raise RuntimeError(
                    "Frame을 저장하지 "
                    "못했습니다: "
                    f"{output_path}"
                )

        # ==========================================
        # ⑬ 현재 Action 처리 완료 결과 출력
        # ==========================================
        # 하나의 Action에서 세 개의 JPG가
        # 정상적으로 저장되면 [OK]를 출력합니다.
        #
        # Action Label과 전체 Frame 수도 표시합니다.
        #
        # 다음 Action이 있다면
        # ⑩~⑬ 과정을 반복합니다.
        #
        # 6개의 Action을 처리하면
        # 정상적으로 총 18개 JPG가 생성됩니다.

        print(
            f"[OK] "
            f"{action['action_label']} "
            f"frames={total_frames}"
        )

    # ==============================================
    # ⑭ 최종 JPG 저장 경로 출력
    # ==============================================
    # 모든 Action의 대표 Frame 추출이 끝나면
    # 결과 이미지가 저장된 폴더를 출력합니다.
    #
    # 기본 저장 위치:
    # reports/day04/keyframes/
    #
    # 이 폴더에서 각 Action의
    # Start / Middle / End 이미지를 비교합니다.

    print()
    print(
        f"Saved: {output_dir}"
    )


# ==================================================
# 프로그램 실행 시작
# ==================================================
# Python 파일을 직접 실행했을 때
# main() 함수를 호출합니다.

if __name__ == "__main__":
    main()