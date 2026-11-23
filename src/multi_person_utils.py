from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)

from src.realtime_utils import (
    PredictionStabilizer,
    TimedFeatureBuffer,
)


@dataclass
class PersonRuntimeState:
    feature_buffer: TimedFeatureBuffer
    stabilizer: PredictionStabilizer
    last_seen_time: float
    raw_label: str = "WARMING_UP"
    threshold_label: str = "WARMING_UP"
    stable_label: str = "WARMING_UP"
    confidence: float = 0.0
    last_prediction_time: float = -1e9


class MultiPersonStateManager:
    def __init__(
        self,
        window_seconds: float,
        stabilization_buffer: int,
        stabilization_min_votes: int,
        lost_timeout_seconds: float,
    ):
        self.window_seconds = float(window_seconds)
        self.stabilization_buffer = int(
            stabilization_buffer
        )
        self.stabilization_min_votes = int(
            stabilization_min_votes
        )
        self.lost_timeout_seconds = float(
            lost_timeout_seconds
        )

        if self.window_seconds <= 0:
            raise ValueError(
                "window_seconds는 0보다 커야 합니다."
            )

        if self.lost_timeout_seconds <= 0:
            raise ValueError(
                "lost_timeout_seconds는 0보다 커야 합니다."
            )

        self.states: dict[int, PersonRuntimeState] = {}

    def get_or_create(
        self,
        track_id: int,
        now: float,
    ) -> PersonRuntimeState:
        track_id = int(track_id)
        now = float(now)

        if track_id not in self.states:
            self.states[track_id] = PersonRuntimeState(
                feature_buffer=TimedFeatureBuffer(
                    self.window_seconds
                ),
                stabilizer=PredictionStabilizer(
                    buffer_size=self.stabilization_buffer,
                    min_votes=self.stabilization_min_votes,
                ),
                last_seen_time=now,
            )

        state = self.states[track_id]
        state.last_seen_time = now

        return state

    def mark_missing(
        self,
        now: float,
        seen_ids: set[int],
    ) -> None:
        now = float(now)

        for track_id, state in self.states.items():
            if track_id not in seen_ids:
                state.feature_buffer.append(
                    now,
                    None,
                )

    def remove_lost(
        self,
        now: float,
    ) -> list[int]:
        now = float(now)
        removed = []

        for track_id, state in list(
            self.states.items()
        ):
            if (
                now - state.last_seen_time
                > self.lost_timeout_seconds
            ):
                removed.append(track_id)
                del self.states[track_id]

        return sorted(removed)

    def clear(self) -> None:
        self.states.clear()

    def active_ids(self) -> list[int]:
        return sorted(self.states.keys())


def feature_from_keypoints(keypoints):
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if keypoints.shape != (17, 2):
        return None

    if not np.all(np.isfinite(keypoints)):
        return None

    try:
        normalized, _, _ = (
            normalize_by_shoulder_width(keypoints)
        )
    except ValueError:
        return None

    angles = selected_joint_angles(keypoints)

    feature = []

    for point in normalized:
        feature.extend(
            [
                float(point[0]),
                float(point[1]),
            ]
        )

    feature.extend(
        [
            float(angles["left_elbow_angle"]),
            float(angles["right_elbow_angle"]),
            float(angles["left_knee_angle"]),
            float(angles["right_knee_angle"]),
        ]
    )

    feature = np.asarray(
        feature,
        dtype=np.float32,
    )

    if len(feature) != 38:
        return None

    if not np.all(np.isfinite(feature)):
        return None

    return feature