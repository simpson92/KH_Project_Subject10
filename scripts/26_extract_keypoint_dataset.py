
from __future__ import annotations

# ====================================================
# 1번. 필요한 라이브러리 가져오기
# ====================================================
# argparse: 터미널의 실행 옵션을 읽습니다.
# csv: Action 설정과 결과 CSV를 읽고 저장합니다.
# re: 영상 파일명 규칙을 검사합니다.
# sys, Path: 프로젝트 모듈과 파일 경로를 관리합니다.
# cv2: MP4를 Frame별로 읽습니다.
# NumPy: Confidence 평균과 대표 인물 선택에 사용합니다.
# YOLO: 각 Frame의 사람 Pose를 추론합니다.

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ====================================================
# 2번. 프로젝트 ROOT 설정 및 공통 모듈 가져오기
# ====================================================
# scripts 폴더의 상위 위치를 프로젝트 ROOT로 사용합니다.
# sys.path에 ROOT를 등록하여 src 모듈을 읽습니다.
# load_config: 프로젝트 설정을 가져오는 함수입니다.
# resolve_device: CPU/GPU 실행 장치를 선택합니다.
# COCO_KEYPOINT_NAMES: Pose의 관절 이름 17개입니다.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
)
from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ====================================================
# 3번. 입력 영상·출력 CSV 경로 및 파일명 규칙 정의
# ====================================================
# 입력: data/day05/videos/S01~S03의 MP4 18개입니다.
# 출력: data/day05/keypoints/ 아래의 Clip별 CSV입니다.
# Action 기준: configs/action_classes.csv입니다.
# 전체 목록: keypoint_dataset_manifest.csv입니다.
# 파일명은 S01_A001_T01.mp4 형식을 사용합니다.
# 정규표현식은 Subject / Action / Take를 구분합니다.

DEFAULT_VIDEO_ROOT = (
    ROOT / "data" / "day05" / "videos"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT / "data" / "day05" / "keypoints"
)

ACTION_FILE = (
    ROOT / "configs" / "action_classes.csv"
)

DEFAULT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "keypoint_dataset_manifest.csv"
)

FILENAME_PATTERN = re.compile(
    r"^(S\d{2})_"
    r"(A\d{3})_"
    r"(T\d{2})\.mp4$",
    re.IGNORECASE,
)


# ====================================================
# 4번. 터미널 실행 옵션 설정
# ====================================================
# --video-root: MP4가 있는 입력 폴더입니다.
# --output-root: Keypoint CSV를 저장할 폴더입니다.
# --manifest-path: 전체 Clip 목록표를 저장할 경로입니다.
# --expected-clips: 예상 MP4 수(기본 18개)입니다.
# 옵션을 입력하지 않으면 위의 기본 경로를 사용합니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-root",
        default=str(
            DEFAULT_VIDEO_ROOT
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--manifest-path",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
    )

    parser.add_argument(
        "--expected-clips",
        type=int,
        default=18,
        help=(
            "기본 본수업은 18개. "
            "Challenge에서는 3을 사용합니다."
        ),
    )

    return parser.parse_args()


# ====================================================
# 5번. Action ID와 실제 동작 이름 연결
# ====================================================
# action_classes.csv를 DictReader로 읽습니다.
# Action ID의 앞뒤 공백 제거 후 대문자로 맞춥니다.
# Action Label도 앞뒤 공백을 제거하여 연결합니다.
# 예: A006 -> wave, A001 -> bend_return입니다.
# A001~A006이 모두 존재하는지 ID 집합을 검사합니다.
# 정상이라면 Action 매핑 딕셔너리를 반환합니다.

def load_action_map() -> dict[str, str]:
    with ACTION_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    action_map = {
        row["action_id"]
        .strip()
        .upper():
        row["action_label"]
        .strip()
        for row in rows
    }

    expected = {
        "A001",
        "A002",
        "A003",
        "A004",
        "A005",
        "A006",
    }

    if set(action_map) != expected:
        raise RuntimeError(
            "action_classes.csv의 "
            "A001~A006 정의를 "
            "확인하세요."
        )

    return action_map


# ====================================================
# 6번. 17개 관절의 총 85개 컬럼 이름 생성
# ====================================================
# COCO 관절마다 다섯 가지 값을 기록합니다.
# 1) x_px: 픽셀 기준 가로 좌표
# 2) y_px: 픽셀 기준 세로 좌표
# 3) x_norm: 이미지 가로 크기로 나눈 좌표
# 4) y_norm: 이미지 세로 크기로 나눈 좌표
# 5) conf: 관절 Confidence
# 17개 관절 × 5개 항목 = 총 85개 컬럼입니다.

def build_keypoint_columns() -> list[str]:
    columns = []

    for name in COCO_KEYPOINT_NAMES:
        columns.extend(
            [
                f"{name}_x_px",
                f"{name}_y_px",
                f"{name}_x_norm",
                f"{name}_y_norm",
                f"{name}_conf",
            ]
        )

    return columns


# ====================================================
# 7번. 영상 파일명에서 Subject·Action·Take 추출
# ====================================================
# 예: S03_A006_T01.mp4 -> S03, A006, T01입니다.
# 파일명이 정규표현식에 맞지 않으면 오류를 냅니다.
# 파일을 보관한 Subject 폴더도 같은지 확인합니다.
# Action ID를 실제 Action Label과 연결합니다.
# 정상이라면 네 값(Subject, Action ID, Label, Take)을 반환합니다.

def parse_clip_name(
    path: Path,
    action_map: dict[str, str],
):
    match = FILENAME_PATTERN.match(
        path.name
    )

    if match is None:
        raise ValueError(
            f"파일명 규칙 오류: "
            f"{path.name}"
        )

    subject = match.group(1).upper()
    action_id = match.group(2).upper()
    take = match.group(3).upper()

    if path.parent.name != subject:
        raise ValueError(
            "Subject 폴더와 파일명이 "
            f"다릅니다: {path}"
        )

    action_label = action_map.get(
        action_id
    )

    if action_label is None:
        raise ValueError(
            f"등록되지 않은 Action: "
            f"{action_id}"
        )

    return (
        subject,
        action_id,
        action_label,
        take,
    )


# ====================================================
# 8번. 검출된 사람의 Pose 데이터 존재 여부 확인
# ====================================================
# YOLO 결과에 BBox와 Keypoint가 있는지 확인합니다.
# 하나라도 없거나 검출 수가 0이면 None을 반환합니다.
# 유효한 사람이 있다면 다음 단계에서 대표자를 선택합니다.

def select_person(result):
    if (
        result.boxes is None
        or result.keypoints is None
        or result.keypoints.xy is None
        or len(result.boxes) == 0
        or len(result.keypoints.xy) == 0
    ):
        return None

    # ====================================================
    # 9번. BBox Confidence가 가장 높은 사람 선택
    # ====================================================
    # BBox Confidence를 NumPy 배열로 가져옵니다.
    # BBox와 Keypoint 양쪽에 존재하는 사람 수만 비교합니다.
    # np.argmax()로 BBox Confidence가 가장 높은 인덱스를 선택합니다.
    # 영상에 여러 사람이 있어도 한 명의 Keypoint만 저장합니다.
    # 주의: Frame마다 대표자가 달라질 수 있으며 사람 추적은 아닙니다.

    box_conf = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    person_count = min(
        len(box_conf),
        len(result.keypoints.xy),
    )

    if person_count == 0:
        return None

    return int(
        np.argmax(
            box_conf[:person_count]
        )
    )


# ====================================================
# 10번. Clip 하나의 기본 정보 및 영상 Metadata 확인
# ====================================================
# process_clip()은 MP4 한 개 전체를 처리합니다.
# 파일명에서 Subject·Action·Take를 가져옵니다.
# OpenCV로 영상을 열고 FPS, Width, Height, Frame 수를 읽습니다.
# 영상이 열리지 않거나 FPS·해상도가 올바르지 않으면 중단합니다.
# metadata_frames는 파일에 기록된 예상 전체 Frame 수입니다.

def process_clip(
    model,
    clip_path: Path,
    action_map: dict[str, str],
    config: dict,
    device,
    output_root: Path,
) -> dict:
    (
        subject,
        action_id,
        action_label,
        take,
    ) = parse_clip_name(
        clip_path,
        action_map,
    )

    capture = cv2.VideoCapture(
        str(clip_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: "
            f"{clip_path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
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
    metadata_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        fps <= 0
        or width <= 0
        or height <= 0
    ):
        capture.release()
        raise RuntimeError(
            "영상 Metadata를 확인할 수 "
            f"없습니다: {clip_path}"
        )

    # ====================================================
    # 11번. Clip별 출력 폴더와 CSV 컬럼 구성
    # ====================================================
    # Subject / Action Label별 하위 폴더를 만듭니다.
    # 예: keypoints/S01/wave/S01_A006_T01.csv
    # 기본정보 10개 + Keypoint 항목 85개 = 총 95개 컬럼입니다.
    # 기본정보에는 Frame 번호·시각·검출 인원·Pose 여부 등이 포함됩니다.

    output_dir = (
        output_root
        / subject
        / action_label
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{clip_path.stem}.csv"
    )

    keypoint_columns = (
        build_keypoint_columns()
    )

    fieldnames = [
        "clip_id",
        "frame_id",
        "time_sec",
        "subject",
        "action_id",
        "action_label",
        "take",
        "person_count",
        "selected_box_conf",
        "pose_detected",
        *keypoint_columns,
    ]

    # ====================================================
    # 12번. Frame 처리 결과와 집계 변수 초기화
    # ====================================================
    # rows: Frame별 정보를 저장할 목록입니다.
    # frame_id: 0부터 시작하는 Frame 번호입니다.
    # detected_frames: Pose 검출에 성공한 Frame 수입니다.
    # single_person_frames: 사람 1명만 검출한 Frame 수입니다.
    # frame_mean_confidences: 검출 Frame별 평균 관절 Confidence입니다.

    rows = []
    frame_id = 0
    detected_frames = 0
    single_person_frames = 0
    frame_mean_confidences = []

    # ====================================================
    # 13번. MP4를 처음부터 끝까지 Frame 단위로 읽기
    # ====================================================
    # capture.read()로 한 Frame씩 읽습니다.
    # 읽기에 실패하면 while 반복을 종료합니다.
    # 각 Frame마다 다음 YOLO Pose 추론 단계로 넘어갑니다.

    while True:
        success, frame = capture.read()

        if not success:
            break

        # ====================================================
        # 14번. 현재 Frame에 YOLO Pose 추론 실행
        # ====================================================
        # Frame 이미지를 YOLO Pose 모델에 전달합니다.
        # config의 image_size와 person_confidence를 사용합니다.
        # CPU 또는 GPU 실행 장치를 지정합니다.
        # 추론 결과의 첫 번째 항목([0])을 사용합니다.

        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config[
                "person_confidence"
            ],
            device=device,
            verbose=False,
        )[0]

        # ====================================================
        # 15번. 검출 인원 수 계산 및 Frame 기본 행 생성
        # ====================================================
        # BBox 수로 현재 Frame에서 검출된 사람 수를 계산합니다.
        # 정확히 한 명이면 single_person_frames를 증가시킵니다.
        # Frame ID, 시간(frame_id/FPS), Subject·Action·Take를 기록합니다.
        # 처음에는 pose_detected=0으로 둡니다.
        # 85개 Keypoint 컬럼도 모두 빈 문자열로 초기화합니다.
        # 따라서 Pose가 없는 Frame도 시간 순서대로 보존됩니다.

        person_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        if person_count == 1:
            single_person_frames += 1

        row = {
            "clip_id": clip_path.stem,
            "frame_id": frame_id,
            "time_sec": round(
                frame_id / fps,
                4,
            ),
            "subject": subject,
            "action_id": action_id,
            "action_label": action_label,
            "take": take,
            "person_count": person_count,
            "selected_box_conf": "",
            "pose_detected": 0,
        }

        for column in keypoint_columns:
            row[column] = ""

        # ====================================================
        # 16번. 대표 인물 선택 및 COCO 17 Keypoint 수 검사
        # ====================================================
        # select_person()으로 대표 인물 인덱스를 찾습니다.
        # 인덱스가 있으면 해당 사람의 Keypoint xy를 NumPy로 변환합니다.
        # 관절 수가 COCO 17개와 다르면 오류를 발생시킵니다.
        # 사람이 없으면 관절 정보가 빈 행으로 남습니다.

        selected_index = select_person(
            result
        )

        if selected_index is not None:
            xy = (
                result.keypoints.xy[
                    selected_index
                ]
                .detach()
                .cpu()
                .numpy()
            )

            if len(xy) != len(
                COCO_KEYPOINT_NAMES
            ):
                capture.release()
                raise RuntimeError(
                    "COCO 17 Keypoint 수가 "
                    f"아닙니다: "
                    f"{clip_path.name}"
                )

            # ====================================================
            # 17번. 관절·BBox Confidence 가져오기 및 검출 통계 기록
            # ====================================================
            # Keypoint Confidence가 있으면 선택 인물의 값을 읽습니다.
            # 없으면 기존 코드 기준으로 17개를 모두 1.0으로 처리합니다.
            # 이는 실제 측정 Confidence가 아닌 대체값입니다.
            # 선택된 사람의 BBox Confidence를 기록합니다.
            # pose_detected=1 및 detected_frames를 증가시킵니다.
            # 현재 Frame의 관절 Confidence 평균을 모아 둡니다.

            if (
                result.keypoints.conf
                is not None
            ):
                kp_conf = (
                    result.keypoints.conf[
                        selected_index
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                kp_conf = np.ones(
                    len(
                        COCO_KEYPOINT_NAMES
                    ),
                    dtype=np.float32,
                )

            box_conf = float(
                result.boxes.conf[
                    selected_index
                ]
                .detach()
                .cpu()
                .item()
            )

            row[
                "selected_box_conf"
            ] = round(
                box_conf,
                4,
            )
            row["pose_detected"] = 1

            detected_frames += 1

            frame_mean_confidences.append(
                float(
                    np.mean(kp_conf)
                )
            )

            # ====================================================
            # 18번. Pixel·Image Normalized·Confidence 값 저장
            # ====================================================
            # 17개 관절을 순서대로 반복합니다.
            # x_px와 y_px는 원본 이미지의 픽셀 좌표입니다.
            # x_norm = x_px / 영상 Width 입니다.
            # y_norm = y_px / 영상 Height 입니다.
            # _conf는 모델이 제공한 관절 신뢰도입니다.
            # 관절마다 5개 수치를 행에 기록합니다.
            # 주의: 이미지 기준 정규화는 3일차의 몸 중심 정규화와 다릅니다.

            for index, name in enumerate(
                COCO_KEYPOINT_NAMES
            ):
                x_px, y_px = xy[index]

                x_norm = (
                    float(x_px) / width
                )
                y_norm = (
                    float(y_px) / height
                )
                confidence = float(
                    kp_conf[index]
                )

                row[
                    f"{name}_x_px"
                ] = round(
                    float(x_px),
                    3,
                )
                row[
                    f"{name}_y_px"
                ] = round(
                    float(y_px),
                    3,
                )
                row[
                    f"{name}_x_norm"
                ] = round(
                    x_norm,
                    6,
                )
                row[
                    f"{name}_y_norm"
                ] = round(
                    y_norm,
                    6,
                )
                row[
                    f"{name}_conf"
                ] = round(
                    confidence,
                    4,
                )

        # ====================================================
        # 19번. Frame별 결과 보존 및 60 Frame마다 진행상황 출력
        # ====================================================
        # 현재 Frame의 모든 정보를 rows에 추가합니다.
        # Pose를 못 찾은 Frame도 빈 관절값을 가진 채 저장됩니다.
        # frame_id를 1 증가시킵니다.
        # 60 Frame을 처리할 때마다 진행상황을 출력합니다.

        rows.append(row)
        frame_id += 1

        if frame_id % 60 == 0:
            print(
                f"  frame {frame_id}"
            )

    # ====================================================
    # 20번. 영상 해제 및 빈 영상 검사
    # ====================================================
    # 영상 끝까지 반복한 후 VideoCapture를 해제합니다.
    # 읽은 Frame이 0개이면 RuntimeError를 발생시킵니다.
    # Frame이 하나 이상이면 CSV 저장을 진행합니다.

    capture.release()

    if frame_id == 0:
        raise RuntimeError(
            "읽은 Frame이 없습니다: "
            f"{clip_path}"
        )

    # ====================================================
    # 21번. Clip 한 개의 Keypoint CSV 저장
    # ====================================================
    # utf-8-sig 인코딩으로 Clip별 CSV를 저장합니다.
    # 첫 줄은 95개 컬럼명, 이후에는 Frame당 한 행입니다.
    # 영상 하나당 Keypoint CSV 파일 하나를 생성합니다.

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    # ====================================================
    # 22번. Pose 성공률·1인 검출률·평균 관절 신뢰도 계산
    # ====================================================
    # pose_ratio = Pose 검출 Frame 수 / 전체 읽은 Frame 수
    # single_person_ratio = 정확히 한 명만 검출한 Frame 수 / 전체 Frame 수
    # mean_kp_conf = 검출 Frame의 평균 관절 Confidence들의 평균
    # 검출 Frame이 없다면 평균 관절 신뢰도는 0.0입니다.
    # 이 값들은 다음 QA에서 Clip의 품질을 판단하는 참고값입니다.

    pose_ratio = (
        detected_frames / frame_id
    )
    single_person_ratio = (
        single_person_frames / frame_id
    )

    mean_kp_conf = (
        float(
            np.mean(
                frame_mean_confidences
            )
        )
        if frame_mean_confidences
        else 0.0
    )

    # ====================================================
    # 23번. Clip별 Manifest 항목 반환
    # ====================================================
    # Clip ID, 영상 경로, Keypoint CSV 경로를 기록합니다.
    # Subject·Action·Take, 해상도, FPS도 포함합니다.
    # Metadata Frame 수와 실제 읽은 Frame 수도 구분합니다.
    # Pose 검출 수·비율, 1인 검출률, 신뢰도 평균을 함께 반환합니다.
    # 이 딕셔너리가 나중에 전체 Manifest의 한 행이 됩니다.

    return {
        "clip_id": clip_path.stem,
        "video_path": str(
            clip_path.relative_to(ROOT)
        ),
        "keypoint_csv": str(
            output_path.relative_to(ROOT)
        ),
        "subject": subject,
        "action_id": action_id,
        "action_label": action_label,
        "take": take,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "metadata_frames": (
            metadata_frames
        ),
        "read_frames": frame_id,
        "pose_detected_frames": (
            detected_frames
        ),
        "pose_ratio": round(
            pose_ratio,
            4,
        ),
        "single_person_ratio": round(
            single_person_ratio,
            4,
        ),
        "mean_keypoint_conf": round(
            mean_kp_conf,
            4,
        ),
    }


# ====================================================
# 24번. main 실행 옵션과 필수 입력 검증
# ====================================================
# 터미널 옵션을 실제 Path로 변환합니다.
# 예상 Clip 수가 1개 이상인지 검사합니다.
# action_classes.csv와 MP4 입력 폴더가 있는지 확인합니다.
# 잘못된 설정이 있으면 다음 단계로 넘어가지 않습니다.

def main() -> None:
    args = parse_args()

    video_root = Path(
        args.video_root
    )
    output_root = Path(
        args.output_root
    )
    manifest_path = Path(
        args.manifest_path
    )

    if args.expected_clips < 1:
        raise ValueError(
            "--expected-clips는 "
            "1 이상이어야 합니다."
        )

    if not ACTION_FILE.exists():
        raise FileNotFoundError(
            ACTION_FILE
        )

    if not video_root.exists():
        raise FileNotFoundError(
            video_root
        )

    # ====================================================
    # 25번. Action 매핑·모델 설정·MP4 목록 준비
    # ====================================================
    # Action ID와 Label 매핑을 읽습니다.
    # 프로젝트 설정을 읽고 CPU/GPU를 정합니다.
    # MP4를 정렬하여 전체 입력 목록을 만듭니다.
    # 입력 개수가 --expected-clips와 다른 경우 오류를 냅니다.
    # 주의: 여기서는 개수만 검사하므로 25번 사전 검사도 중요합니다.

    action_map = load_action_map()
    config = load_config()
    device = resolve_device(
        config["device"]
    )

    clips = sorted(
        video_root.rglob("*.mp4")
    )

    if len(clips) != args.expected_clips:
        raise RuntimeError(
            "입력 MP4 개수가 예상과 다릅니다. "
            f"expected={args.expected_clips}, "
            f"actual={len(clips)}"
        )

    # ====================================================
    # 26번. YOLO Pose 모델 로드 후 모든 Clip 처리
    # ====================================================
    # 출력 폴더를 만들고 YOLO 모델을 한 번 로드합니다.
    # manifest_rows를 초기화합니다.
    # 18개 Clip을 하나씩 선택해 process_clip()을 호출합니다.
    # 각 Clip의 결과를 Manifest 목록에 추가합니다.
    # Pose 검출 비율과 평균 신뢰도를 콘솔에 출력합니다.

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(
        config["pose_model"]
    )

    manifest_rows = []

    print(f"Clips: {len(clips)}")
    print()

    for index, clip in enumerate(
        clips,
        start=1,
    ):
        print(
            f"[{index}/{len(clips)}] "
            f"{clip.name}"
        )

        item = process_clip(
            model=model,
            clip_path=clip,
            action_map=action_map,
            config=config,
            device=device,
            output_root=output_root,
        )

        manifest_rows.append(item)

        print(
            "  pose_ratio="
            f"{item['pose_ratio']} "
            "mean_conf="
            f"{item['mean_keypoint_conf']}"
        )

    # ====================================================
    # 27번. 전체 Manifest CSV 저장 및 완료 메시지 출력
    # ====================================================
    # 전체 Clip의 결과를 keypoint_dataset_manifest.csv로 저장합니다.
    # 한 행이 한 Clip을 대표합니다(기본 본수업은 18행).
    # Keypoint extraction: COMPLETE 메시지와 경로를 출력합니다.
    # 이후 27번 코드에서 자동 QA를 진행할 준비가 됩니다.

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                manifest_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    print()
    print(
        "Keypoint extraction: COMPLETE"
    )
    print(
        f"Manifest: {manifest_path}"
    )


# ====================================================
# 28번. Python 파일 직접 실행 시 main() 시작
# ====================================================
# 이 파일을 터미널에서 직접 실행한 경우 main()을 호출합니다.
# 함수 정의들은 위에서 준비되고 실제 실행은 main()부터 시작합니다.

if __name__ == "__main__":
    main()