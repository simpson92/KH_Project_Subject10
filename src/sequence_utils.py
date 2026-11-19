from __future__ import annotations

import numpy as np
import pandas as pd


from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


FRAME_FEATURE_NAMES = []

for name in COCO_KEYPOINT_NAMES:
    FRAME_FEATURE_NAMES.extend(
        [
            f"{name}_x",
            f"{name}_y",
        ]
    )

FRAME_FEATURE_NAMES.extend(
    [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
    ]
)


def keypoints_from_row(
    row: pd.Series,
):
    keypoints = []

    for name in COCO_KEYPOINT_NAMES:
        x = row.get(
            f"{name}_x_px"
        )
        y = row.get(
            f"{name}_y_px"
        )

        if (
            pd.isna(x)
            or pd.isna(y)
        ):
            return None

        keypoints.append(
            [
                float(x),
                float(y),
            ]
        )

    values = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(values)
    ):
        return None

    return values


def frame_feature_from_row(
    row: pd.Series,
):
    pose_detected = int(
        row.get(
            "pose_detected",
            0,
        )
    )

    if pose_detected != 1:
        return None

    keypoints = keypoints_from_row(
        row
    )

    if keypoints is None:
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

    values = []

    for point in normalized:
        values.extend(
            [
                float(point[0]),
                float(point[1]),
            ]
        )

    values.extend(
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

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) != 38:
        raise RuntimeError(
            "Frame Feature가 "
            "38개가 아닙니다."
        )

    if not np.all(
        np.isfinite(values)
    ):
        return None

    return values


def build_frame_feature_matrix(
    df: pd.DataFrame,
):
    if "frame_id" not in df.columns:
        raise RuntimeError(
            "frame_id 열이 없습니다."
        )

    ordered = (
        df.sort_values(
            "frame_id"
        )
        .reset_index(
            drop=True
        )
    )

    features = []
    frame_ids = []

    for _, row in (
        ordered.iterrows()
    ):
        feature = (
            frame_feature_from_row(
                row
            )
        )

        if feature is None:
            continue

        features.append(feature)
        frame_ids.append(
            int(row["frame_id"])
        )

    if not features:
        return (
            np.empty(
                (
                    0,
                    len(
                        FRAME_FEATURE_NAMES
                    ),
                ),
                dtype=np.float32,
            ),
            np.asarray(
                [],
                dtype=np.int32,
            ),
        )

    return (
        np.stack(features),
        np.asarray(
            frame_ids,
            dtype=np.int32,
        ),
    )


def resample_sequence(
    frame_features,
    frame_ids,
    target_steps: int = 30,
):
    frame_features = np.asarray(
        frame_features,
        dtype=np.float32,
    )

    frame_ids = np.asarray(
        frame_ids,
        dtype=np.float32,
    )

    if frame_features.ndim != 2:
        raise ValueError(
            "frame_features는 "
            "(T, F) 형태여야 합니다."
        )

    if len(frame_features) != len(
        frame_ids
    ):
        raise ValueError(
            "Feature 수와 Frame ID 수가 "
            "다릅니다."
        )

    if len(frame_features) < 2:
        raise ValueError(
            "유효한 Frame Feature가 "
            "2개 이상 필요합니다."
        )

    if target_steps < 2:
        raise ValueError(
            "target_steps는 "
            "2 이상이어야 합니다."
        )

    span = (
        frame_ids[-1]
        - frame_ids[0]
    )

    if span <= 0:
        raise ValueError(
            "Frame 시간 순서를 "
            "확인할 수 없습니다."
        )

    original_x = (
        frame_ids
        - frame_ids[0]
    ) / span

    target_x = np.linspace(
        0.0,
        1.0,
        target_steps,
    )

    output = np.empty(
        (
            target_steps,
            frame_features.shape[1],
        ),
        dtype=np.float32,
    )

    for feature_index in range(
        frame_features.shape[1]
    ):
        output[
            :,
            feature_index,
        ] = np.interp(
            target_x,
            original_x,
            frame_features[
                :,
                feature_index,
            ],
        )

    return output


def flatten_sequence(
    sequence,
):
    return np.asarray(
        sequence,
        dtype=np.float32,
    ).reshape(-1)


def sequence_feature_names(
    target_steps: int,
):
    names = []

    for step in range(
        target_steps
    ):
        for feature_name in (
            FRAME_FEATURE_NAMES
        ):
            names.append(
                f"t{step:02d}_"
                f"{feature_name}"
            )

    return names


def middle_frame_feature(
    frame_features,
):
    frame_features = np.asarray(
        frame_features,
        dtype=np.float32,
    )

    if len(frame_features) == 0:
        raise ValueError(
            "Frame Feature가 없습니다."
        )

    middle_index = (
        len(frame_features)
        // 2
    )

    return frame_features[
        middle_index
    ]