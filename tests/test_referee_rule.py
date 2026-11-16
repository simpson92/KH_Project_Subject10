from __future__ import annotations

# ==================================================
# scripts/17_referee_rule.py — 1조 U파울 규칙 테스트
# ==================================================
# 규칙 (feature_design.md 9장)
#   조건 1: 양 손목이 모두 코보다 위 (wrist_y - nose_y < 0)
#   조건 2: 양 손목 거리 < T (기본 1.0 어깨 폭)
#
# 파일 이름이 숫자로 시작해서 `import 17_referee_rule`이 불가능하므로
# importlib로 파일 경로에서 직접 불러옵니다.

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def judge_ufoul():
    path = ROOT / "scripts" / "17_referee_rule.py"
    spec = importlib.util.spec_from_file_location("referee_rule_17", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main()은 __name__ 검사 때문에 실행되지 않음
    return module.judge_ufoul


NOSE_Y = -2.5  # 정규화 좌표에서 코는 골반 중심보다 약 2.5 어깨 폭 위


def test_ufoul_true_when_both_conditions(judge_ufoul):
    # 실측 S02 U파울 가운데 Frame과 비슷한 값
    r = judge_ufoul(NOSE_Y, (0.2, -3.7), (-0.2, -3.7), max_wrist_dist=1.0)
    assert r["cond1_wrists_above_nose"]
    assert r["cond2_wrists_close"]
    assert r["ufoul"]


def test_pushing_is_false(judge_ufoul):
    # 푸싱: 손목이 가슴 앞(코보다 아래), 양손 간격 넓음
    r = judge_ufoul(NOSE_Y, (0.7, -1.4), (-0.7, -1.4), max_wrist_dist=1.0)
    assert not r["cond1_wrists_above_nose"]
    assert not r["ufoul"]


def test_one_hand_above_nose_is_false(judge_ufoul):
    # 한 손만 들면 U파울 아님 (양손 신호)
    r = judge_ufoul(NOSE_Y, (0.1, -3.5), (-0.6, -1.0), max_wrist_dist=1.0)
    assert not r["cond1_wrists_above_nose"]
    assert not r["ufoul"]


def test_hands_above_but_apart_is_false(judge_ufoul):
    # 손은 머리 위지만 아직 손목을 잡기 전 (조건 2가 걸러내는 Frame)
    r = judge_ufoul(NOSE_Y, (0.8, -3.5), (-0.8, -3.5), max_wrist_dist=1.0)
    assert r["cond1_wrists_above_nose"]
    assert not r["cond2_wrists_close"]
    assert not r["ufoul"]


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.99, True),
        (1.00, False),  # 규칙은 "< T" → 정확히 T면 False
        (1.01, False),
    ],
)
def test_threshold_boundary(judge_ufoul, distance, expected):
    r = judge_ufoul(
        NOSE_Y, (distance / 2, -3.5), (-distance / 2, -3.5), max_wrist_dist=1.0
    )
    assert r["wrist_distance"] == pytest.approx(distance)
    assert r["ufoul"] is expected


def test_wrist_at_nose_height_is_not_above(judge_ufoul):
    # wrist_y == nose_y 는 "위"가 아님 (< 0 조건)
    r = judge_ufoul(NOSE_Y, (0.1, NOSE_Y), (-0.1, NOSE_Y), max_wrist_dist=1.0)
    assert not r["cond1_wrists_above_nose"]


def test_custom_threshold(judge_ufoul):
    # --max-wrist-dist 옵션으로 T를 바꾸면 결과도 바뀜
    args = (NOSE_Y, (0.6, -3.5), (-0.6, -3.5))
    assert not judge_ufoul(*args, max_wrist_dist=1.0)["ufoul"]
    assert judge_ufoul(*args, max_wrist_dist=1.5)["ufoul"]
