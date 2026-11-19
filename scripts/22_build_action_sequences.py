from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse: 실행 옵션, csv: 설정/결과 CSV 처리
# sys/Path: 프로젝트 경로와 파일 위치 관리
# cv2: MP4 읽기, NumPy: 관절 통계와 값 검증
# YOLO: 각 Frame에서 사람과 COCO 관절 추정
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ==================================================
# ② 프로젝트 ROOT 설정 및 공통 함수 연결
# ==================================================
# 이 파일의 상위 프로젝트 폴더를 찾습니다.
# sys.path에 ROOT를 추가하여 src의 공통 함수를 불러옵니다.
# feature_utils는 정규화와 팔꿈치/무릎 각도 계산에 사용합니다.
ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# ③ Action 설정·동영상·결과 폴더 지정
# ==================================================
# Action 목록: configs/action_classes.csv
# 동영상 입력: data/day04/videos/
# Sequence CSV 저장: reports/day04/sequences/
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
    / "sequences"
)


# ==================================================
# ④ 터미널 실행 옵션 정의
# ==================================================
# --source: 특정 MP4만 처리(기본값 None = 전체 Action)
# --kp-conf: 저신뢰도 관절을 세는 기준(기본 0.50)
# --max-frames: 각 영상의 최대 처리 Row 수(0 = 전체)
# --frame-step: Frame 간격(1 = 모두, 2 = 0, 2, 4, ...)
# 설정/영상/출력 경로도 옵션으로 변경할 수 있습니다.
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=None,
        help=(
            "특정 MP4만 처리할 때 파일명 입력. "
            "예: bend_return.mp4"
        ),
    )

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

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "0이면 끝까지 처리, "
            "양수이면 최대 처리 Row 수"
        ),
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=(
            "1이면 모든 Frame, "
            "2이면 2 Frame마다 처리"
        ),
    )

    return parser.parse_args()


# ==================================================
# ⑤ Frame당 38개 Feature의 이름 생성
# ==================================================
# COCO 17개 관절 x/y = 34개 정규화 좌표 이름
# 팔꿈치·무릎 관절각도 4개 이름을 추가합니다.
# 34 + 4 = 최종 38개 Feature 이름을 반환합니다.
def make_feature_names() -> list[str]:
    names = []

    for keypoint_name in (
        COCO_KEYPOINT_NAMES
    ):
        names.append(
            f"{keypoint_name}_x"
        )

        names.append(
            f"{keypoint_name}_y"
        )

    names.extend(
        [
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
        ]
    )

    return names


# ==================================================
# ⑥ Action 설정 CSV 읽기
# ==================================================
# action_classes.csv 각 행을 딕셔너리로 읽습니다.
# action_id / action_label / video_file 등을 사용합니다.
def load_actions(
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
# ⑦ 실패 Frame을 위한 빈 Feature 만들기
# ==================================================
# 38개 Feature 열의 값을 모두 빈 문자열로 만듭니다.
# Pose/Feature 실패 시에도 Frame 행을 남기기 위한 함수입니다.
def make_empty_feature(
    feature_names: list[str],
) -> dict:
    return {
        name: ""
        for name in feature_names
    }


# ==================================================
# ⑧ Action Clip 한 개를 처리하는 함수 시작
# ==================================================
# Action에 연결된 MP4 파일 경로를 구성하고 확인합니다.
# OpenCV로 영상을 엽니다. 열리지 않으면 오류를 발생시킵니다.
# 이 함수는 영상 하나를 Sequence CSV 하나로 변환합니다.
def process_clip(
    model,
    action: dict,
    config: dict,
    device,
    kp_threshold: float,
    max_frames: int,
    frame_step: int,
    video_dir: Path,
    output_dir: Path,
) -> None:
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

    # ==================================================
    # ⑨ FPS·Feature 목록·Frame 카운터 준비
    # ==================================================
    # FPS로 frame_id / fps 시간을 계산합니다.
    # FPS가 0 이하이면 오류를 발생시킵니다.
    # rows에 처리 Frame 결과를 저장합니다.
    # frame_id는 0부터 시작하도록 -1로 초기화합니다.
    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:
        capture.release()

        raise RuntimeError(
            f"FPS를 확인할 수 없습니다: "
            f"{video_path}"
        )

    feature_names = (
        make_feature_names()
    )

    rows = []

    frame_id = -1
    processed_count = 0

    # ==================================================
    # ⑩ 원본 MP4를 Frame 순서대로 읽기
    # ==================================================
    # 한 Frame을 읽고 frame_id를 증가시킵니다.
    # frame-step으로 지정된 Frame만 Pose 분석합니다.
    # 선택되지 않은 Frame은 CSV에 기록하지 않습니다.
    # processed_count는 실제 분석 대상으로 선택된 Frame 수입니다.
    while True:
        success, frame = (
            capture.read()
        )

        if not success:
            break

        frame_id += 1

        if frame_id % frame_step != 0:
            continue

        processed_count += 1

        time_sec = (
            frame_id / fps
        )

        # ==================================================
        # ⑪ 현재 Frame의 기본 Metadata 행 생성
        # ==================================================
        # Action ID/Label, 파일명, 원본 Frame 번호, 시간을 기록합니다.
        # 처음에는 pose_ok=0, feature_ok=0으로 둡니다.
        # 추가 품질/정규화 값은 빈칸으로 초기화합니다.
        base_row = {
            "action_id": (
                action["action_id"]
            ),
            "action_label": (
                action["action_label"]
            ),
            "source": (
                action["video_file"]
            ),
            "frame_id": frame_id,
            "time_sec": round(
                time_sec,
                4,
            ),
            "pose_ok": 0,
            "feature_ok": 0,
            "mean_keypoint_conf": "",
            "low_keypoint_count": "",
            "hip_center_x": "",
            "hip_center_y": "",
            "shoulder_width": "",
        }

        # ==================================================
        # ⑫ YOLO Pose로 현재 Frame 분석
        # ==================================================
        # 사람 Detection과 Pose 추론을 수행합니다.
        # person_confidence는 사람 검출 임계값입니다.
        # 관절 품질 계산의 kp-conf와는 다른 설정입니다.
        result = model.predict(
            source=frame,
            imgsz=config[
                "image_size"
            ],
            conf=config[
                "person_confidence"
            ],
            device=device,
            verbose=False,
        )[0]

        # ==================================================
        # ⑬ Pose를 찾지 못한 Frame도 기록
        # ==================================================
        # 사람 BBox 또는 Keypoint가 없으면 실패 행을 기록합니다.
        # pose_ok=0, feature_ok=0, 38개 Feature는 빈칸입니다.
        # Frame 자체는 삭제하지 않아 시간 위치를 보존합니다.
        if (
            result.boxes is None
            or result.keypoints
            is None
            or result.keypoints.xy
            is None
            or len(result.boxes) == 0
            or len(
                result.keypoints.xy
            ) == 0
        ):
            rows.append(
                {
                    **base_row,
                    **make_empty_feature(
                        feature_names
                    ),
                }
            )
        else:
            # ==================================================
            # ⑭ 여러 사람 중 대표 인물 선택
            # ==================================================
            # BBox Confidence와 사람별 Keypoint 좌표를 읽습니다.
            # 둘의 공통 개수(person_count)를 계산합니다.
            # 검출 인원이 있다면 BBox Confidence 최대 인물을 선택합니다.
            # 인물 추적이 아닌 Frame별 최고 점수 선택 방식입니다.
            boxes_conf = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            xy = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            person_count = min(
                len(boxes_conf),
                len(xy),
            )

            if person_count == 0:
                rows.append(
                    {
                        **base_row,
                        **make_empty_feature(
                            feature_names
                        ),
                    }
                )
            else:
                main_index = int(
                    np.argmax(
                        boxes_conf[
                            :person_count
                        ]
                    )
                )

                keypoints = xy[
                    main_index
                ]

                # ==================================================
                # ⑮ 선택한 사람의 관절 Confidence 분석
                # ==================================================
                # 17개 관절 Confidence의 평균을 계산합니다.
                # kp_threshold 미만 관절의 수를 계산합니다.
                # Confidence가 제공되지 않으면 모두 1.0으로 간주합니다.
                # Pose가 있으면 pose_ok=1로 기록합니다.
                # 저신뢰도 관절이 있어도 이 단계만으로 제외하지 않습니다.
                if (
                    result.keypoints.conf
                    is not None
                ):
                    confidence = (
                        result.keypoints.conf
                        .cpu()
                        .numpy()[
                            main_index
                        ]
                    )
                else:
                    confidence = (
                        np.ones(
                            17,
                            dtype=np.float32,
                        )
                    )

                mean_conf = float(
                    np.mean(
                        confidence
                    )
                )

                low_count = int(
                    np.sum(
                        confidence
                        < kp_threshold
                    )
                )

                detected_row = {
                    **base_row,
                    "pose_ok": 1,
                    "mean_keypoint_conf":
                        round(
                            mean_conf,
                            4,
                        ),
                    "low_keypoint_count":
                        low_count,
                }

                # ==================================================
                # ⑯ 관절 좌표 정규화와 관절각도 계산
                # ==================================================
                # 어깨 너비 기준으로 17개 관절 좌표를 정규화합니다.
                # 중심좌표(center)와 어깨 너비(scale)도 받습니다.
                # 왼쪽/오른쪽 팔꿈치 및 무릎 각도를 계산합니다.
                try:
                    normalized, (
                        center
                    ), scale = (
                        normalize_by_shoulder_width(
                            keypoints
                        )
                    )

                    angles = (
                        selected_joint_angles(
                            keypoints
                        )
                    )

                    # ==================================================
                    # ⑰ 34개 좌표 + 4개 각도 = 38차원 Feature
                    # ==================================================
                    # 정규화 좌표 17개에서 각 x/y를 순서대로 수집합니다.
                    # 이후 팔꿈치 2개와 무릎 2개의 각도를 추가합니다.
                    feature_values = []

                    for point in normalized:
                        feature_values.extend(
                            [
                                float(
                                    point[0]
                                ),
                                float(
                                    point[1]
                                ),
                            ]
                        )

                    feature_values.extend(
                        [
                            angles[
                                "left_elbow_angle"
                            ],
                            angles[
                                "right_elbow_angle"
                            ],
                            angles[
                                "left_knee_angle"
                            ],
                            angles[
                                "right_knee_angle"
                            ],
                        ]
                    )

                    # ==================================================
                    # ⑱ Feature 개수와 NaN·Inf 검사
                    # ==================================================
                    # Feature 수가 정확히 38개인지 확인합니다.
                    # NaN이나 무한대 값이 있으면 ValueError를 발생시킵니다.
                    # 유효하지 않은 Feature를 정상 데이터로 저장하지 않습니다.
                    if len(
                        feature_values
                    ) != 38:
                        raise ValueError(
                            "Frame Feature 수가 "
                            "38개가 아닙니다."
                        )

                    if not np.all(
                        np.isfinite(
                            feature_values
                        )
                    ):
                        raise ValueError(
                            "Frame Feature에 "
                            "NaN 또는 Inf가 있습니다."
                        )

                    # ==================================================
                    # ⑲ 정상 Feature를 CSV 행에 기록
                    # ==================================================
                    # Feature 38개를 이름과 매칭하여 소수점 6자리로 정리합니다.
                    # feature_ok=1과 중심좌표, 어깨 너비를 함께 저장합니다.
                    # 해당 Frame 결과를 rows에 추가합니다.
                    feature_row = {
                        name: round(
                            float(value),
                            6,
                        )
                        for name, value
                        in zip(
                            feature_names,
                            feature_values,
                        )
                    }

                    rows.append(
                        {
                            **detected_row,
                            "feature_ok": 1,
                            "hip_center_x":
                                round(
                                    float(
                                        center[0]
                                    ),
                                    4,
                                ),
                            "hip_center_y":
                                round(
                                    float(
                                        center[1]
                                    ),
                                    4,
                                ),
                            "shoulder_width":
                                round(
                                    float(scale),
                                    4,
                                ),
                            **feature_row,
                        }
                    )

                # ==================================================
                # ⑳ Feature 생성 실패 시 빈값으로 기록
                # ==================================================
                # 정규화/각도/개수/유한값 검증에서 ValueError가 나면
                # pose_ok=1, feature_ok=0 및 빈 38개 Feature를 기록합니다.
                # 실패한 Frame도 rows에 추가하여 위치를 유지합니다.
                except ValueError:
                    rows.append(
                        {
                            **detected_row,
                            **make_empty_feature(
                                feature_names
                            ),
                        }
                    )

        # ==================================================
        # ㉑ 처리 Row 제한·진행 출력·영상 해제
        # ==================================================
        # max_frames가 양수이면 지정한 수에서 반복을 멈춥니다.
        # 60개 Row마다 진행 상황을 출력합니다.
        # 반복 종료 후 VideoCapture를 해제하고 빈 결과를 검사합니다.
        if (
            max_frames > 0
            and processed_count
            >= max_frames
        ):
            break

        if (
            processed_count % 60
            == 0
        ):
            print(
                f"  processed: "
                f"{processed_count}"
            )

    capture.release()

    if not rows:
        raise RuntimeError(
            f"처리된 Frame이 없습니다: "
            f"{video_path}"
        )

    # ==================================================
    # ㉒ Action Sequence CSV 저장 및 성공 건수 요약
    # ==================================================
    # 결과 폴더를 만들고 action_label_sequence.csv로 저장합니다.
    # 기본 Metadata 12개 + Feature 38개 = 총 50개 열입니다.
    # pose_ok와 feature_ok 건수를 출력합니다.
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"{action['action_label']}"
            "_sequence.csv"
        )
    )

    fieldnames = [
        "action_id",
        "action_label",
        "source",
        "frame_id",
        "time_sec",
        "pose_ok",
        "feature_ok",
        "mean_keypoint_conf",
        "low_keypoint_count",
        "hip_center_x",
        "hip_center_y",
        "shoulder_width",
        *feature_names,
    ]

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

    pose_ok_count = sum(
        int(row["pose_ok"])
        for row in rows
    )

    feature_ok_count = sum(
        int(row["feature_ok"])
        for row in rows
    )

    print(
        f"[SAVED] "
        f"{action['action_label']} "
        f"rows={len(rows)} "
        f"pose_ok={pose_ok_count} "
        f"feature_ok="
        f"{feature_ok_count}"
    )

    print(
        f"        {output_path}"
    )


# ==================================================
# ㉓ 실행 옵션 값 검증
# ==================================================
# frame-step은 1 이상, max-frames는 0 이상이어야 합니다.
# kp-conf는 0~1 범위여야 합니다.
# 잘못된 입력은 ValueError로 중단합니다.
def main() -> None:
    args = parse_args()

    if args.frame_step < 1:
        raise ValueError(
            "--frame-step은 "
            "1 이상이어야 합니다."
        )

    if args.max_frames < 0:
        raise ValueError(
            "--max-frames는 "
            "0 이상이어야 합니다."
        )

    if not (
        0.0
        <= args.kp_conf
        <= 1.0
    ):
        raise ValueError(
            "--kp-conf는 "
            "0~1 범위여야 합니다."
        )

    # ==================================================
    # ㉔ Action 목록 읽기 및 처리 대상 선택
    # ==================================================
    # 설정 파일/동영상 폴더 존재 여부를 검사합니다.
    # Action 정의를 읽고 --source가 있으면 해당 MP4만 남깁니다.
    # 아무것도 지정하지 않으면 설정된 모든 Action을 처리합니다.
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

    actions = load_actions(
        action_file
    )

    if not actions:
        raise RuntimeError(
            "Action 정의가 없습니다."
        )

    if args.source is not None:
        actions = [
            action
            for action in actions
            if action["video_file"]
            == args.source
        ]

        if not actions:
            raise RuntimeError(
                "설정 파일에서 "
                f"{args.source}를 "
                "찾지 못했습니다."
            )

    # ==================================================
    # ㉕ YOLO 모델 1회 로드 후 Action별 반복 처리
    # ==================================================
    # 설정과 CPU/GPU 장치를 선택하고 YOLO Pose를 로드합니다.
    # Action마다 process_clip()을 호출하여 Sequence CSV를 만듭니다.
    # 모델은 영상마다 새로 로드하지 않고 재사용합니다.
    config = load_config()

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

    for action in actions:
        print()
        print(
            f"Processing: "
            f"{action['video_file']}"
        )

        process_clip(
            model=model,
            action=action,
            config=config,
            device=device,
            kp_threshold=args.kp_conf,
            max_frames=args.max_frames,
            frame_step=args.frame_step,
            video_dir=video_dir,
            output_dir=output_dir,
        )


# ==================================================
# ㉖ Python 파일을 직접 실행하면 main() 시작
# ==================================================
# 다른 파일에서 import만 할 때는 main()을 자동 실행하지 않습니다.
if __name__ == "__main__":
    main()