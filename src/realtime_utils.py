from __future__ import annotations

from collections import Counter, deque

import numpy as np

from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)
from src.sequence_utils import (
    flatten_sequence,
    resample_sequence,
)


def validate_runtime_values(
    window_seconds: float,
    confidence_threshold: float,
    stabilization_buffer: int,
    stabilization_min_votes: int,
) -> None:
    if float(window_seconds) <= 0:
        raise ValueError(
            "window_seconds는 "
            "0보다 커야 합니다."
        )

    if not (
        0
        <= float(
            confidence_threshold
        )
        <= 1
    ):
        raise ValueError(
            "confidence_threshold는 "
            "0~1 범위여야 합니다."
        )

    if int(
        stabilization_buffer
    ) < 1:
        raise ValueError(
            "stabilization_buffer는 "
            "1 이상이어야 합니다."
        )

    if not (
        1
        <= int(
            stabilization_min_votes
        )
        <= int(
            stabilization_buffer
        )
    ):
        raise ValueError(
            "stabilization_min_votes는 "
            "1 이상이고 "
            "stabilization_buffer보다 "
            "클 수 없습니다."
        )


class TimedFeatureBuffer:
    def __init__(self, window_seconds: float):
        self.window_seconds = float(
            window_seconds
        )

        if self.window_seconds <= 0:
            raise ValueError(
                "window_seconds는 "
                "0보다 커야 합니다."
            )

        self.items = deque()

    def append(self, timestamp: float, feature) -> None:
        timestamp = float(timestamp)
        self.items.append((timestamp, feature))
        self._trim(timestamp)

    def _trim(self, current_time: float) -> None:
        cutoff = float(current_time) - self.window_seconds

        while self.items and self.items[0][0] < cutoff:
            self.items.popleft()

    def clear(self) -> None:
        self.items.clear()

    def duration(self) -> float:
        if len(self.items) < 2:
            return 0.0

        return float(
            self.items[-1][0]
            - self.items[0][0]
        )

    def valid_ratio(self) -> float:
        if not self.items:
            return 0.0

        valid_count = sum(
            feature is not None
            for _, feature in self.items
        )

        return (
            valid_count
            / len(self.items)
        )

    def valid_matrix_and_times(self):
        valid_items = [
            (timestamp, feature)
            for timestamp, feature in self.items
            if feature is not None
        ]

        if not valid_items:
            return (
                np.empty(
                    (0, 38),
                    dtype=np.float32,
                ),
                np.asarray(
                    [],
                    dtype=np.float32,
                ),
            )

        times = np.asarray(
            [
                timestamp
                for timestamp, _ in valid_items
            ],
            dtype=np.float32,
        )

        matrix = np.stack(
            [
                np.asarray(
                    feature,
                    dtype=np.float32,
                )
                for _, feature in valid_items
            ]
        )

        return matrix, times

    def is_ready(
        self,
        min_window_fill_ratio: float,
        min_valid_feature_ratio: float,
        min_valid_features: int = 2,
    ) -> bool:
        required_duration = (
            self.window_seconds
            * float(
                min_window_fill_ratio
            )
        )

        if self.duration() < required_duration:
            return False

        if (
            self.valid_ratio()
            < float(
                min_valid_feature_ratio
            )
        ):
            return False

        matrix, _ = (
            self.valid_matrix_and_times()
        )

        return (
            len(matrix)
            >= int(
                min_valid_features
            )
        )


class PredictionStabilizer:
    def __init__(
        self,
        buffer_size: int = 5,
        min_votes: int = 3,
    ):
        self.buffer_size = int(
            buffer_size
        )
        self.min_votes = int(
            min_votes
        )

        if self.buffer_size < 1:
            raise ValueError(
                "buffer_size는 "
                "1 이상이어야 합니다."
            )

        if not (
            1
            <= self.min_votes
            <= self.buffer_size
        ):
            raise ValueError(
                "min_votes는 1 이상이고 "
                "buffer_size보다 "
                "클 수 없습니다."
            )

        self.buffer = deque(
            maxlen=self.buffer_size
        )

    def clear(self) -> None:
        self.buffer.clear()

    def update(
        self,
        label: str,
    ) -> str:
        self.buffer.append(
            str(label)
        )

        counts = Counter(
            self.buffer
        )

        best_label, votes = (
            counts.most_common(1)[0]
        )

        if votes < self.min_votes:
            return "UNKNOWN"

        return str(
            best_label
        )

    def history(self):
        return list(
            self.buffer
        )


class FPSMeter:
    def __init__(
        self,
        smoothing: float = 0.90,
    ):
        self.smoothing = float(
            smoothing
        )
        self.last_time = None
        self.fps = 0.0

    def update(
        self,
        now: float,
    ) -> float:
        now = float(now)

        if self.last_time is None:
            self.last_time = now
            return 0.0

        delta = (
            now
            - self.last_time
        )

        self.last_time = now

        if delta <= 0:
            return self.fps

        instant_fps = (
            1.0
            / delta
        )

        if self.fps <= 0:
            self.fps = (
                instant_fps
            )
        else:
            self.fps = (
                self.smoothing
                * self.fps
                +
                (
                    1.0
                    - self.smoothing
                )
                * instant_fps
            )

        return float(
            self.fps
        )


def frame_feature_from_pose(
    result,
):
    if (
        result.boxes is None
        or result.keypoints is None
        or result.keypoints.xy is None
        or len(result.boxes) == 0
        or len(result.keypoints.xy) == 0
    ):
        return None

    box_conf = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    person_count = min(
        len(box_conf),
        len(
            result.keypoints.xy
        ),
    )

    if person_count == 0:
        return None

    person_index = int(
        np.argmax(
            box_conf[
                :person_count
            ]
        )
    )

    keypoints = (
        result.keypoints.xy[
            person_index
        ]
        .detach()
        .cpu()
        .numpy()
    )

    if keypoints.shape != (17, 2):
        return None

    try:
        normalized, _, _ = (
            normalize_by_shoulder_width(
                keypoints
            )
        )
    except ValueError:
        return None

    angles = (
        selected_joint_angles(
            keypoints
        )
    )

    feature = []

    for point in normalized:
        feature.extend(
            [
                float(
                    point[0]
                ),
                float(
                    point[1]
                ),
            ]
        )

    feature.extend(
        [
            float(
                angles[
                    "left_elbow_angle"
                ]
            ),
            float(
                angles[
                    "right_elbow_angle"
                ]
            ),
            float(
                angles[
                    "left_knee_angle"
                ]
            ),
            float(
                angles[
                    "right_knee_angle"
                ]
            ),
        ]
    )

    feature = np.asarray(
        feature,
        dtype=np.float32,
    )

    if len(feature) != 38:
        return None

    if not np.all(
        np.isfinite(
            feature
        )
    ):
        return None

    return feature


def predict_from_window(
    model,
    feature_buffer: TimedFeatureBuffer,
    sequence_steps: int,
    confidence_threshold: float,
):
    matrix, times = (
        feature_buffer
        .valid_matrix_and_times()
    )

    if len(matrix) < 2:
        raise ValueError(
            "예측에는 유효한 "
            "Frame Feature가 "
            "2개 이상 필요합니다."
        )

    if len(matrix) != len(times):
        raise ValueError(
            "Feature 수와 Timestamp 수가 "
            "다릅니다."
        )

    if not np.all(
        np.diff(times) > 0
    ):
        raise ValueError(
            "Timestamp는 시간 순서대로 "
            "증가해야 합니다."
        )

    sequence = (
        resample_sequence(
            matrix,
            times,
            target_steps=int(
                sequence_steps
            ),
        )
    )

    flattened = (
        flatten_sequence(
            sequence
        )
    ).reshape(
        1,
        -1,
    )

    expected_features = getattr(
        model,
        "n_features_in_",
        flattened.shape[1],
    )

    if (
        flattened.shape[1]
        != expected_features
    ):
        raise RuntimeError(
            "실시간 Feature 수와 "
            "학습 모델 Feature 수가 "
            "다릅니다: "
            f"{flattened.shape[1]} "
            f"!= {expected_features}"
        )

    probabilities = (
        model.predict_proba(
            flattened
        )[0]
    )

    best_index = int(
        np.argmax(
            probabilities
        )
    )

    raw_label = str(
        model.classes_[
            best_index
        ]
    )

    confidence = float(
        probabilities[
            best_index
        ]
    )

    if (
        confidence
        < float(
            confidence_threshold
        )
    ):
        threshold_label = (
            "UNKNOWN"
        )
    else:
        threshold_label = (
            raw_label
        )

    return (
        threshold_label,
        raw_label,
        confidence,
        probabilities,
    )