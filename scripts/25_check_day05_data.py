
from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# csv  : 검사한 18개 영상의 정보를 CSV 파일로 저장합니다.
# re   : 영상 파일명이 정해진 규칙과 일치하는지 검사합니다.
# Path : 영상 폴더와 파일 경로를 다룹니다.
# cv2  : OpenCV로 MP4를 열고 Frame 및 영상 정보를 확인합니다.

import csv
import re
from pathlib import Path

import cv2


# ==================================================
# ② 프로젝트 ROOT와 입출력 경로 설정
# ==================================================
# ROOT는 현재 Python 파일이 들어 있는 scripts 폴더의
# 상위 폴더, 즉 프로젝트 폴더를 가리킵니다.
#
# VIDEO_ROOT:
# S01~S03의 원본 MP4를 보관하는 폴더입니다.
#
# OUTPUT_PATH:
# 검사 결과를 기록할 video_inventory.csv입니다.

ROOT = Path(__file__).resolve().parents[1]

VIDEO_ROOT = ROOT / "data" / "day05" / "videos"

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day05"
    / "tables"
    / "video_inventory.csv"
)


# ==================================================
# ③ 필요한 Subject·Action·Take 기준 설정
# ==================================================
# Subject:
# 서로 다른 촬영 대상 3명(S01~S03)입니다.
#
# Action:
# 각 사람이 수행할 동작 6개(A001~A006)입니다.
#
# Take:
# 각 동작의 첫 번째 촬영본(T01)입니다.
#
# 따라서 필수 영상 개수는 다음과 같습니다.
#
# 3명 × 6동작 × 1회 = 18개

SUBJECTS = ["S01", "S02", "S03"]

ACTIONS = [
    "A001",
    "A002",
    "A003",
    "A004",
    "A005",
    "A006",
]

TAKE = "T01"


# ==================================================
# ④ MP4 파일명 규칙을 정규표현식으로 정의
# ==================================================
# 요구되는 파일명 형식:
# Subject_Action_Take.mp4
#
# 예:
# S02_A004_T01.mp4
#
# 정규표현식 설명:
#
# S\d{2} : S 다음에 숫자 두 자리
# A\d{3} : A 다음에 숫자 세 자리
# T\d{2} : T 다음에 숫자 두 자리
# \.mp4  : 확장자 .mp4
#
# ^와 $는 파일명 전체가 이 규칙에
# 일치해야 한다는 의미입니다.
#
# IGNORECASE는 정규표현식에서 대소문자를
# 무시하지만, 예상 경로와 실제 경로는
# 이후 별도로 비교합니다.

FILENAME_PATTERN = re.compile(
    r"^(S\d{2})_(A\d{3})_(T\d{2})\.mp4$",
    re.IGNORECASE,
)


# ==================================================
# ⑤ 반드시 존재해야 하는 18개 MP4 경로 생성
# ==================================================
# expected_paths()는 정상적으로 준비되어야 하는
# 18개 MP4의 전체 경로를 만드는 함수입니다.
#
# Subject 3개와 Action 6개를 이중 반복합니다.
#
# 예:
# data/day05/videos/S01/S01_A001_T01.mp4
#
# 각 조합을 set에 추가합니다.
#
# 최종적으로 예상 파일 경로 18개를 반환합니다.
#
# 이후 실제 MP4 목록과 비교하여
# 누락되거나 추가된 파일을 찾습니다.

def expected_paths() -> set[Path]:
    paths = set()

    for subject in SUBJECTS:
        for action in ACTIONS:
            filename = (
                f"{subject}_"
                f"{action}_"
                f"{TAKE}.mp4"
            )

            paths.add(
                VIDEO_ROOT
                / subject
                / filename
            )

    return paths


# ==================================================
# ⑥ MP4 파일을 열고 열기 성공 여부 확인
# ==================================================
# read_video_info()는 동영상 하나를 검사하는 함수입니다.
#
# OpenCV VideoCapture로 MP4 파일을 엽니다.
#
# 열 수 없다면:
# open_ok = 0
# read_ok = 0
#
# FPS, Frame 수, 해상도, 재생시간도 0으로
# 설정하여 반환합니다.
#
# 정상적으로 열린다면 다음 단계에서
# 영상의 실제 정보를 읽습니다.

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
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }

    # ==============================================
    # ⑦ FPS·전체 Frame 수·해상도 읽기
    # ==============================================
    # FPS:
    # 1초에 포함되는 Frame 수입니다.
    #
    # frames:
    # 동영상의 전체 Frame 수입니다.
    #
    # width:
    # 영상의 가로 Pixel 수입니다.
    #
    # height:
    # 영상의 세로 Pixel 수입니다.
    #
    # 이 값들은 VideoCapture가 제공하는
    # 영상의 기본 Metadata입니다.
    #
    # 예:
    # FPS = 25
    # Frames = 100
    # Width = 640
    # Height = 480

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
    # ⑧ 첫 Frame 읽기·재생시간 계산 후 정보 반환
    # ==============================================
    # 파일이 정상적으로 열리더라도
    # 실제 Frame을 읽지 못할 수 있습니다.
    #
    # 그래서 첫 번째 Frame을 읽어 봅니다.
    #
    # 읽기에 성공했다면:
    # read_ok = 1
    #
    # 실패했다면:
    # read_ok = 0
    #
    # 검사가 끝나면 VideoCapture를 해제합니다.
    #
    # 재생시간 계산식:
    # 전체 Frame 수 / FPS
    #
    # FPS가 0이면 0으로 나누는 오류를
    # 방지하기 위해 0.0을 사용합니다.
    #
    # 주의:
    # 첫 Frame만 검사하므로 영상의 모든 Frame이
    # 정상이라는 사실까지 보장하지는 않습니다.

    success, first_frame = (
        capture.read()
    )

    capture.release()

    read_ok = int(
        success
        and first_frame is not None
    )

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
        "width": width,
        "height": height,
        "duration_sec": round(
            duration,
            3,
        ),
    }


def main() -> None:

    # ==============================================
    # ⑨ 예상 MP4와 실제 MP4 파일 목록 수집
    # ==============================================
    # expected_paths()를 실행하여
    # 반드시 존재해야 하는 18개 경로를 가져옵니다.
    #
    # rglob("*.mp4"):
    # VIDEO_ROOT의 모든 하위 폴더에서
    # 확장자가 .mp4인 파일을 찾습니다.
    #
    # expected:
    # 정상적으로 있어야 하는 파일 목록입니다.
    #
    # actual:
    # 실제 폴더에서 찾은 MP4 목록입니다.
    #
    # 두 집합의 차집합을 이용해
    # 누락 파일과 예상하지 않은 파일을 찾습니다.
    #
    # expected - actual:
    # 있어야 하지만 없는 파일
    #
    # actual - expected:
    # 예상 목록에는 없는 추가 파일

    expected = expected_paths()

    actual = set(
        VIDEO_ROOT.rglob("*.mp4")
    )

    missing = sorted(
        expected - actual
    )

    unexpected = sorted(
        actual - expected
    )

    # ==============================================
    # ⑩ 누락·예상 외 MP4를 출력하고 오류 처리
    # ==============================================
    # missing에 파일이 있다면
    # [MISSING]을 출력합니다.
    #
    # unexpected에 파일이 있다면
    # [UNEXPECTED]를 출력합니다.
    #
    # 문제가 발견되면 잘못된 파일 경로를
    # 터미널에 표시합니다.
    #
    # 누락 또는 추가 파일이 하나라도 존재하면
    # RuntimeError를 발생시키고 중단합니다.
    #
    # 이 단계에서 폴더 위치나 파일명 오류를
    # 먼저 찾아낼 수 있습니다.

    if missing:
        print("[MISSING]")

        for path in missing:
            print(
                path.relative_to(ROOT)
            )

    if unexpected:
        print("[UNEXPECTED]")

        for path in unexpected:
            print(
                path.relative_to(ROOT)
            )

    if missing or unexpected:
        raise RuntimeError(
            "5일차 MP4 파일명과 "
            "폴더 구조를 다시 확인하세요."
        )

    # ==============================================
    # ⑪ 영상별 검사 결과와 문제 목록 준비
    # ==============================================
    # rows:
    # 각 MP4에서 읽은 영상 정보를 저장합니다.
    #
    # problems:
    # 파일명, 폴더명, 영상 Metadata 오류를
    # 기록하는 목록입니다.
    #
    # 실제 발견한 MP4를 이름순으로 정렬한 후
    # 하나씩 검사합니다.

    rows = []
    problems = []

    for path in sorted(actual):

        # ==========================================
        # ⑫ 파일명 규칙 확인 및 Subject·Action·Take 추출
        # ==========================================
        # FILENAME_PATTERN을 이용하여
        # 파일명이 규칙에 맞는지 검사합니다.
        #
        # 예:
        # S03_A005_T01.mp4
        #
        # group(1) → S03
        # group(2) → A005
        # group(3) → T01
        #
        # upper():
        # 추출한 문자열을 대문자로 변환합니다.
        #
        # 파일명 규칙에 맞지 않으면
        # problems에 오류를 기록합니다.
        #
        # continue를 이용하여
        # 다음 MP4 검사로 넘어갑니다.

        match = FILENAME_PATTERN.match(
            path.name
        )

        if match is None:
            problems.append(
                f"파일명 규칙 오류: "
                f"{path.name}"
            )
            continue

        subject = (
            match.group(1).upper()
        )

        action_id = (
            match.group(2).upper()
        )

        take = (
            match.group(3).upper()
        )

        # ==========================================
        # ⑬ 파일명의 Subject와 실제 폴더명 비교
        # ==========================================
        # 예:
        # S02_A004_T01.mp4
        #
        # 이 파일은 S02 폴더 안에 있어야 합니다.
        #
        # path.parent.name:
        # 현재 MP4가 들어 있는 폴더의 이름입니다.
        #
        # 파일명 Subject와 폴더명이 다르면
        # problems에 오류를 추가합니다.

        if path.parent.name != subject:
            problems.append(
                "폴더 Subject와 "
                "파일명 Subject가 다름: "
                f"{path}"
            )

        # ==========================================
        # ⑭ OpenCV 영상 검사 결과와 Metadata 검증
        # ==========================================
        # 앞에서 만든 read_video_info()를 호출합니다.
        #
        # 검사 조건:
        #
        # open_ok = 1
        # read_ok = 1
        # FPS > 0
        # Frame 수 > 0
        # Width > 0
        # Height > 0
        #
        # 모든 조건을 만족해야
        # metadata_ok = True입니다.
        #
        # 하나라도 실패하면
        # problems에 오류를 기록합니다.

        info = read_video_info(path)

        metadata_ok = (
            info["open_ok"] == 1
            and info["read_ok"] == 1
            and info["fps"] > 0
            and info["frames"] > 0
            and info["width"] > 0
            and info["height"] > 0
        )

        if not metadata_ok:
            problems.append(
                "영상 읽기 또는 "
                "Metadata 오류: "
                f"{path.name}"
            )

        # ==========================================
        # ⑮ 검사 정보 출력 및 rows에 추가
        # ==========================================
        # 현재 MP4의 다음 정보를 출력합니다.
        #
        # 파일명
        # 해상도
        # FPS
        # 전체 Frame 수
        #
        # 또한 Subject, Action, Take, 영상 경로,
        # Metadata를 하나의 딕셔너리로 만듭니다.
        #
        # **info:
        # info 딕셔너리 안의 모든 항목을
        # 현재 딕셔너리에 추가하는 문법입니다.
        #
        # 주의:
        # 기존 코드에서는 Metadata 오류가 있어도
        # 이 부분에서 [OK]가 출력될 수 있습니다.
        #
        # 최종 검사 성공 여부는 ⑯의 오류 검사와
        # 마지막 READY 메시지로 확인합니다.

        print(
            f"[OK] {path.name:20s} "
            f"{info['width']}x"
            f"{info['height']} "
            f"fps={info['fps']:.3f} "
            f"frames={info['frames']}"
        )

        rows.append(
            {
                "subject": subject,
                "action_id": action_id,
                "take": take,
                "video_path": str(
                    path.relative_to(ROOT)
                ),
                **info,
            }
        )

    # ==============================================
    # ⑯ 전체 검사 오류와 18개 영상 수량 최종 확인
    # ==============================================
    # 모든 영상 검사가 끝나면
    # problems 목록을 확인합니다.
    #
    # 오류가 하나라도 있다면
    # [ERROR] 메시지와 함께 문제를 출력합니다.
    #
    # RuntimeError를 발생시켜 프로그램을 중단합니다.
    #
    # 오류가 없다면 rows 개수를 확인합니다.
    #
    # 이번 기본 데이터는 반드시 18개여야 합니다.
    #
    # 18개가 아니라면 오류를 발생시킵니다.
    #
    # 모든 조건을 통과해야 CSV를 저장합니다.

    if problems:
        for problem in problems:
            print(
                "[ERROR]",
                problem,
            )

        raise RuntimeError(
            "5일차 입력 영상을 "
            "다시 확인하세요."
        )

    if len(rows) != 18:
        raise RuntimeError(
            "5일차 기본 데이터는 "
            f"18개여야 합니다. "
            f"현재 {len(rows)}개입니다."
        )

    # ==============================================
    # ⑰ 18개 영상의 목록표를 CSV로 저장
    # ==============================================
    # 저장 폴더가 없다면 새로 생성합니다.
    #
    # csv.DictWriter를 이용하여
    # 영상 정보를 CSV 파일로 저장합니다.
    #
    # 첫 번째 행에는 컬럼명을 작성합니다.
    # 이후 18개 영상 정보를 한 행씩 저장합니다.
    #
    # utf-8-sig:
    # Excel에서도 한글을 읽기 쉽게 저장합니다.
    #
    # 최종 저장 위치:
    # reports/day05/tables/video_inventory.csv

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
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ==============================================
    # ⑱ 전체 Clip 수·저장 위치·완료 메시지 출력
    # ==============================================
    # 모든 검사를 통과하고 CSV 파일까지 저장하면
    # 최종 결과를 터미널에 출력합니다.
    #
    # Total clips:
    # 검사에 통과한 MP4 개수입니다.
    #
    # Saved:
    # CSV 파일 저장 경로입니다.
    #
    # Day05 video data: READY:
    # 5일차 입력 영상의 기본 검사가
    # 정상적으로 완료되었다는 의미입니다.
    #
    # 이후 26번 Keypoint 추출 실습으로 진행합니다.

    print()
    print(
        f"Total clips: {len(rows)}"
    )
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print(
        "Day05 video data: READY"
    )


# ==================================================
# ⑲ Python 파일 직접 실행 시 main() 호출
# ==================================================
# Python 파일을 직접 실행하면 main()을 호출합니다.
# 앞의 함수들은 먼저 정의되고,
# 실제 프로그램의 검사 작업은 main()에서 시작합니다.

if __name__ == "__main__":
    main()