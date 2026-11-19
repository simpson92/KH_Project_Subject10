from __future__ import annotations

# ==================================================
# D6 대표 인물 선택 테스트
# ==================================================
# 1조 결정 D6: 여러 사람이 보이면 "가장 큰 BBox"의 사람을 사용
# (docs/referee/01_관절_매핑_명세.md, feature_design.md 9.3 S02_pushing_T06·T07)
#
# 진짜 YOLO 결과 대신, 같은 모양(.xy / .conf / .xyxy + .cpu().numpy())을 가진
# 가짜 결과 객체를 만들어 검사합니다.

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.pose_utils import select_main_person

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# 가짜 YOLO 결과
# --------------------------------------------------
class FakeTensor:
    """torch Tensor처럼 .cpu().numpy()와 인덱싱을 지원하는 NumPy 래퍼."""

    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __len__(self):
        return len(self._array)

    def __getitem__(self, index):
        return FakeTensor(self._array[index])


def make_result(people_xy, boxes_xyxy, box_conf, kp_conf=None):
    if kp_conf is None:
        kp_conf = np.full((len(people_xy), 17), 0.9)
    keypoints = types.SimpleNamespace(
        xy=FakeTensor(people_xy), conf=FakeTensor(kp_conf)
    )
    boxes = types.SimpleNamespace(
        xyxy=FakeTensor(boxes_xyxy), conf=FakeTensor(box_conf)
    )
    return types.SimpleNamespace(keypoints=keypoints, boxes=boxes)


@pytest.fixture
def two_people(standing_pose):
    """0번 = 뒤쪽의 작은 사람(BBox Confidence 높음), 1번 = 앞쪽의 큰 사람(심판).

    수업 09번 방식(conf)과 D6 방식(area)이 서로 다른 사람을 고르는 상황입니다.
    """
    back = standing_pose * 0.4 + np.array([500, 100], dtype=np.float32)
    front = standing_pose.copy()
    return make_result(
        people_xy=np.stack([back, front]),
        boxes_xyxy=[[590, 160, 650, 345], [230, 160, 370, 610]],
        box_conf=[0.95, 0.88],
    )


# --------------------------------------------------
# select_main_person
# --------------------------------------------------
def test_area_picks_largest_bbox(two_people):
    assert select_main_person(two_people, "area") == 1


def test_default_mode_is_area(two_people):
    assert select_main_person(two_people) == 1


def test_conf_picks_highest_box_conf(two_people):
    # D6 근거 1: conf 방식은 뒤쪽 사람을 고를 위험
    assert select_main_person(two_people, "conf") == 0


def test_first_keeps_textbook_behavior(two_people):
    assert select_main_person(two_people, "first") == 0


def test_single_person_is_zero(standing_pose):
    result = make_result(standing_pose[None], [[230, 160, 370, 610]], [0.9])
    for mode in ("area", "conf", "first"):
        assert select_main_person(result, mode) == 0


def test_no_person_is_none():
    empty = types.SimpleNamespace(
        keypoints=types.SimpleNamespace(xy=FakeTensor(np.zeros((0, 17, 2))), conf=None),
        boxes=None,
    )
    assert select_main_person(empty) is None
    assert select_main_person(types.SimpleNamespace(keypoints=None, boxes=None)) is None


def test_missing_boxes_falls_back_to_first(two_people):
    two_people.boxes = None
    assert select_main_person(two_people, "area") == 0


def test_invalid_mode_raises(two_people):
    with pytest.raises(ValueError):
        select_main_person(two_people, "biggest")


# --------------------------------------------------
# 18번 frame_to_row가 대표 인물의 Feature를 쓰는가
# --------------------------------------------------
@pytest.fixture(scope="module")
def script18():
    # 18번은 맨 위에서 cv2 · ultralytics · torch(src.common)를 import합니다.
    # 테스트에서는 frame_to_row만 쓰므로,
    # 설치되지 않은 환경(CI 등)에서는 빈 모듈로 대신합니다.
    # 각자 PC처럼 모두 설치된 환경에서는 진짜 모듈을 그대로 씁니다.
    stubbed = []
    for name in ("cv2", "torch", "ultralytics"):
        if importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            if name == "ultralytics":
                module.YOLO = None
            sys.modules[name] = module
            stubbed.append(name)

    path = ROOT / "scripts" / "18_video_frame_features.py"
    spec = importlib.util.spec_from_file_location("video_frame_features_18", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module

    for name in stubbed:
        sys.modules.pop(name, None)


def test_frame_to_row_uses_main_person(script18, two_people, standing_pose):
    names = script18.make_feature_names()
    row = script18.frame_to_row(two_people, 1, 0.0, 0.5, names, "area")

    assert row["valid_pose"] == 1
    assert row["people"] == 2
    assert row["person_index"] == 1
    # 앞사람(standing_pose)의 어깨 폭 100 px, 골반 중심 (300, 400)
    assert row["shoulder_width"] == pytest.approx(100.0)
    assert (row["hip_center_x"], row["hip_center_y"]) == pytest.approx((300.0, 400.0))


def test_frame_to_row_first_mode_matches_textbook(script18, two_people):
    names = script18.make_feature_names()
    row = script18.frame_to_row(two_people, 1, 0.0, 0.5, names, "first")
    assert row["person_index"] == 0
    assert row["shoulder_width"] == pytest.approx(40.0)  # 뒤쪽 사람 = 0.4배


def test_frame_to_row_keeps_38d_column_order(script18, two_people):
    # 대표 인물 열(people · person_index)은 맨 뒤에 붙어 기존 열 위치가 그대로여야 함
    names = script18.make_feature_names()
    row = script18.frame_to_row(two_people, 1, 0.0, 0.5, names)
    keys = list(row)
    assert keys[:8] == [
        "frame_id",
        "time_sec",
        "valid_pose",
        "mean_keypoint_conf",
        "low_keypoint_count",
        "hip_center_x",
        "hip_center_y",
        "shoulder_width",
    ]
    assert keys[8 : 8 + 38] == names
    assert keys[-2:] == ["people", "person_index"]


def test_frame_to_row_no_person(script18):
    names = script18.make_feature_names()
    empty = types.SimpleNamespace(
        keypoints=types.SimpleNamespace(xy=FakeTensor(np.zeros((0, 17, 2))), conf=None),
        boxes=None,
    )
    row = script18.frame_to_row(empty, 7, 0.2, 0.5, names)
    assert row["valid_pose"] == 0
    assert row["people"] == 0
    assert row["person_index"] == ""
