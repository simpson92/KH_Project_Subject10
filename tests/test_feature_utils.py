from __future__ import annotations

# ==================================================
# src/feature_utils.py 단위 테스트 (3일차 교재 계산 검증)
# ==================================================
# 실행: uv run pytest -v
#
# 교재에서 눈으로 확인한 성질을 자동 검사로 바꿉니다.
#   - 직각 90°, 일직선 180°          (교재 5절)
#   - 골반 중심 (0, 0), 어깨 폭 1.0   (교재 11절)
#   - Near / Far 불변성               (교재 14~15절)
#   - 좌우 반전 = x 부호 + Left/Right 교환 (교재 17~18절)
#   - 34 + 4 = 38-D 순서              (교재 19~20절)
#
# 4일차에 팀 Feature 함수를 추가할 때 이 테스트가 깨지면
# 기존 38-D 계산이 바뀌었다는 뜻입니다.

import math

import numpy as np
import pytest

from src.feature_utils import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_RIGHT_PAIRS,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    body_center,
    calculate_angle,
    flip_keypoints_horizontal,
    normalize_by_shoulder_width,
    selected_joint_angles,
    shoulder_width,
)


# ==================================================
# [1] calculate_angle — 세 점으로 관절각도
# ==================================================
class TestCalculateAngle:
    def test_right_angle_is_90(self):
        # B=(0,0) 기준 BA=(0,-1) 위쪽, BC=(1,0) 오른쪽 → 직각
        assert calculate_angle((0, -1), (0, 0), (1, 0)) == pytest.approx(90.0, abs=1e-4)

    def test_straight_line_is_180(self):
        # 팔을 쭉 편 상태: 세 점이 한 줄
        assert calculate_angle((0, 0), (0, 1), (0, 2)) == pytest.approx(180.0, abs=1e-3)

    def test_folded_is_0(self):
        # 완전히 접힌 상태: A와 C가 B에서 같은 방향
        assert calculate_angle((0, 1), (0, 0), (0, 2)) == pytest.approx(0.0, abs=1e-3)

    def test_45_degree(self):
        assert calculate_angle((1, 0), (0, 0), (1, 1)) == pytest.approx(45.0, abs=1e-4)

    def test_zero_length_vector_returns_nan(self):
        # A와 B가 같은 점이면 방향이 없으므로 각도를 정할 수 없음
        assert math.isnan(calculate_angle((5, 5), (5, 5), (6, 6)))

    def test_order_of_end_points_does_not_matter(self):
        # A와 C를 바꿔도 B 기준 각도는 같음
        a, b, c = (3, 3), (4, 6), (8, 7)
        assert calculate_angle(a, b, c) == pytest.approx(
            calculate_angle(c, b, a), abs=1e-4
        )

    def test_result_range(self):
        # 어떤 입력이든 0~180° 범위 (acos 결과)
        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b, c = rng.uniform(-100, 100, size=(3, 2))
            angle = calculate_angle(a, b, c)
            assert 0.0 <= angle <= 180.0


# ==================================================
# [2] 몸 중심 · 어깨 폭
# ==================================================
def test_body_center_is_hip_midpoint(standing_pose):
    center = body_center(standing_pose)
    np.testing.assert_allclose(center, [300, 400])


def test_shoulder_width(standing_pose):
    assert shoulder_width(standing_pose) == pytest.approx(100.0)


# ==================================================
# [3] normalize_by_shoulder_width — 정규화
# ==================================================
class TestNormalize:
    def test_returns_center_and_scale(self, standing_pose):
        normalized, center, scale = normalize_by_shoulder_width(standing_pose)
        assert normalized.shape == (17, 2)
        np.testing.assert_allclose(center, [300, 400])
        assert scale == pytest.approx(100.0)

    def test_hip_center_moves_to_origin(self, standing_pose):
        normalized, _, _ = normalize_by_shoulder_width(standing_pose)
        hip_mid = (normalized[LEFT_HIP] + normalized[RIGHT_HIP]) / 2
        np.testing.assert_allclose(hip_mid, [0.0, 0.0], atol=1e-6)

    def test_shoulder_width_becomes_one(self, standing_pose):
        normalized, _, _ = normalize_by_shoulder_width(standing_pose)
        width = np.linalg.norm(normalized[LEFT_SHOULDER] - normalized[RIGHT_SHOULDER])
        assert width == pytest.approx(1.0, abs=1e-6)

    def test_y_axis_direction_kept(self, standing_pose):
        # 화면 위쪽 = y 음수 (교재 11절: centered_y < 0 → 골반보다 위)
        normalized, _, _ = normalize_by_shoulder_width(standing_pose)
        assert normalized[LEFT_SHOULDER, 1] < 0  # 어깨는 골반보다 위
        assert normalized[15, 1] > 0  # 발목은 골반보다 아래

    @pytest.mark.parametrize(
        ("scale", "shift"),
        [
            (0.5, (0, 0)),  # Far: 화면 속 사람이 절반 크기
            (2.0, (0, 0)),  # Near: 두 배 크기
            (1.0, (250, -80)),  # 화면 다른 위치에 서 있음
            (
                0.65,
                (120, 200),
            ),  # 교재 Near/Far 어깨 폭 비율(103.7/160.2)과 비슷하게 + 이동
        ],
    )
    def test_near_far_invariance(self, standing_pose, scale, shift):
        # 교재 14~15절 실험을 숫자로 재현:
        # 같은 Pose를 확대·축소·이동해도 정규화 결과는 같아야 함
        moved = standing_pose * scale + np.array(shift, dtype=np.float32)
        base, _, _ = normalize_by_shoulder_width(standing_pose)
        other, _, _ = normalize_by_shoulder_width(moved)
        np.testing.assert_allclose(other, base, atol=1e-5)

    def test_pixel_distance_large_but_normalized_distance_zero(self, standing_pose):
        # 교재 15절 Mean Joint Distance: Pixel은 크고 Normalized는 작아짐
        far = standing_pose * 0.5 + np.array([100, 50], dtype=np.float32)
        pixel_gap = np.linalg.norm(standing_pose - far, axis=1).mean()
        a, _, _ = normalize_by_shoulder_width(standing_pose)
        b, _, _ = normalize_by_shoulder_width(far)
        norm_gap = np.linalg.norm(a - b, axis=1).mean()
        assert pixel_gap > 50
        assert norm_gap < 1e-5

    def test_zero_shoulder_width_raises(self, standing_pose):
        broken = standing_pose.copy()
        broken[RIGHT_SHOULDER] = broken[LEFT_SHOULDER]  # 두 어깨가 겹침
        with pytest.raises(ValueError):
            normalize_by_shoulder_width(broken)

    def test_input_not_modified(self, standing_pose):
        before = standing_pose.copy()
        normalize_by_shoulder_width(standing_pose)
        np.testing.assert_array_equal(standing_pose, before)


# ==================================================
# [4] flip_keypoints_horizontal — 좌우 반전
# ==================================================
class TestFlip:
    def test_flip_twice_is_identity(self, standing_pose):
        normalized, _, _ = normalize_by_shoulder_width(standing_pose)
        twice = flip_keypoints_horizontal(flip_keypoints_horizontal(normalized))
        np.testing.assert_allclose(twice, normalized, atol=1e-6)

    def test_left_right_swapped_and_x_negated(self, standing_pose):
        normalized, _, _ = normalize_by_shoulder_width(standing_pose)
        flipped = flip_keypoints_horizontal(normalized)
        for left_id, right_id in LEFT_RIGHT_PAIRS:
            # 반전 후 Left 자리 = 원래 Right 좌표의 x 부호만 바뀐 값
            np.testing.assert_allclose(
                flipped[left_id],
                [-normalized[right_id, 0], normalized[right_id, 1]],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                flipped[right_id],
                [-normalized[left_id, 0], normalized[left_id, 1]],
                atol=1e-6,
            )

    def test_nose_only_x_negated(self, standing_pose):
        # 0번 코는 짝이 없으므로 x 부호만 바뀜
        normalized, _, _ = normalize_by_shoulder_width(standing_pose)
        flipped = flip_keypoints_horizontal(normalized)
        np.testing.assert_allclose(
            flipped[0], [-normalized[0, 0], normalized[0, 1]], atol=1e-6
        )

    def test_symmetric_pose_unchanged(self, standing_pose):
        # 가짜 Pose는 좌우 대칭 → 반전해도 그대로
        # (교재 18절 "반전 후 차이 감소"의 이상적인 경우)
        normalized, _, _ = normalize_by_shoulder_width(standing_pose)
        np.testing.assert_allclose(
            flip_keypoints_horizontal(normalized), normalized, atol=1e-6
        )

    def test_left_arm_up_becomes_right_arm_up(self, standing_pose):
        # 교재 17~18절: 왼팔 올리기를 반전하면 실제 오른팔 올리기와 같아야 함
        left_up = standing_pose.copy()
        left_up[LEFT_ELBOW] = [360, 140]
        left_up[LEFT_WRIST] = [365, 60]

        right_up = standing_pose.copy()
        right_up[8] = [240, 140]  # right_elbow (x는 골반 중심 300 기준 대칭)
        right_up[RIGHT_WRIST] = [235, 60]

        left_norm, _, _ = normalize_by_shoulder_width(left_up)
        right_norm, _, _ = normalize_by_shoulder_width(right_up)
        flipped = flip_keypoints_horizontal(left_norm)

        before = np.linalg.norm(left_norm - right_norm, axis=1).mean()
        after = np.linalg.norm(flipped - right_norm, axis=1).mean()
        assert before > 0.1
        assert after == pytest.approx(0.0, abs=1e-6)

    def test_input_not_modified(self, standing_pose):
        before = standing_pose.copy()
        flip_keypoints_horizontal(standing_pose)
        np.testing.assert_array_equal(standing_pose, before)


# ==================================================
# [5] selected_joint_angles — 38-D의 각도 4개
# ==================================================
def test_selected_joint_angles_keys_and_order(standing_pose):
    angles = selected_joint_angles(standing_pose)
    # 16·18번이 이 순서로 38-D 뒤 4칸을 채움
    assert list(angles) == [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
    ]


def test_straight_legs_are_180(standing_pose):
    angles = selected_joint_angles(standing_pose)
    assert angles["left_knee_angle"] == pytest.approx(180.0, abs=1e-3)
    assert angles["right_knee_angle"] == pytest.approx(180.0, abs=1e-3)


def test_angles_invariant_to_scale_and_shift(standing_pose):
    # 관절각도는 정규화하지 않아도 크기·위치에 영향을 받지 않음 (교재 13절)
    moved = standing_pose * 0.4 + np.array([77, -33], dtype=np.float32)
    a = selected_joint_angles(standing_pose)
    b = selected_joint_angles(moved)
    for key in a:
        assert b[key] == pytest.approx(a[key], abs=1e-3)


def test_bent_elbow_is_90(standing_pose):
    pose = standing_pose.copy()
    # 왼어깨 (350,220) → 왼팔꿈치 (350,300) 아래로, 손목 (430,300) 옆으로 → 직각
    pose[LEFT_ELBOW] = [350, 300]
    pose[LEFT_WRIST] = [430, 300]
    assert selected_joint_angles(pose)["left_elbow_angle"] == pytest.approx(
        90.0, abs=1e-3
    )
