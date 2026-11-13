from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)


# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import COCO_KEYPOINT_NAMES
from src.referee_utils import KEYPOINT_DIR, SMOKE_REPORT_DIR, parse_name


# ③ 관절각도 4개: (끝, 꼭짓점, 끝)
#    팔꿈치 = 어깨-팔꿈치-손목 / 어깨 = 팔꿈치-어깨-골반
ANGLES = [
    (5, 7, 9),    # left elbow
    (6, 8, 10),   # right elbow
    (7, 5, 11),   # left shoulder
    (8, 6, 12),   # right shoulder
]


def parse_args():
    parser = argparse.ArgumentParser()

    # 영상 길이를 같은 Frame 수로 맞춤
    parser.add_argument("--frames", type=int, default=32)

    # 이 값보다 낮은 관절은 비우고 앞뒤 Frame으로 채움
    parser.add_argument("--kp-conf", type=float, default=0.50)

    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def angle_deg(a, b, c):
    """b를 꼭짓점으로 하는 각도(도). 입력은 (T, 2) 배열."""
    v1 = a - b
    v2 = c - b
    cos = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def frame_features(df: pd.DataFrame, kp_threshold: float) -> np.ndarray:
    """keypoint CSV 하나 → (T, 38) Frame Feature"""

    # ④ 좌표와 Confidence를 (T, 17) 배열로
    xs = df[[f"{n}_x" for n in COCO_KEYPOINT_NAMES]].to_numpy(dtype=float)
    ys = df[[f"{n}_y" for n in COCO_KEYPOINT_NAMES]].to_numpy(dtype=float)
    cs = df[[f"{n}_c" for n in COCO_KEYPOINT_NAMES]].to_numpy(dtype=float)

    # Confidence가 낮은 관절은 믿지 않고 비움 (2일차 Threshold 개념)
    low = cs < kp_threshold
    xs[low] = np.nan
    ys[low] = np.nan

    # 비운 값은 앞뒤 Frame 값으로 채움 (시간 방향 보간)
    xs = pd.DataFrame(xs).interpolate(limit_direction="both").to_numpy()
    ys = pd.DataFrame(ys).interpolate(limit_direction="both").to_numpy()

    # ⑤ 위치 정규화: 양쪽 골반의 중심을 원점으로
    hip_x = (xs[:, 11] + xs[:, 12]) / 2
    hip_y = (ys[:, 11] + ys[:, 12]) / 2

    # ⑥ 크기 정규화: 어깨 너비를 1로
    shoulder = np.hypot(xs[:, 5] - xs[:, 6], ys[:, 5] - ys[:, 6])
    scale = np.where(shoulder > 1e-3, shoulder, np.nan)
    scale = np.where(np.isnan(scale), np.nanmedian(scale), scale)

    norm_x = (xs - hip_x[:, None]) / scale[:, None]
    norm_y = (ys - hip_y[:, None]) / scale[:, None]

    # ⑦ 관절각도 4개 (0~1로 나눔)
    points = np.stack([xs, ys], axis=2)  # (T, 17, 2)
    angles = np.stack(
        [angle_deg(points[:, a], points[:, b], points[:, c]) / 180.0 for a, b, c in ANGLES],
        axis=1,
    )

    # 34개 좌표 + 4개 각도 = 38-D
    features = np.concatenate([norm_x, norm_y, angles], axis=1)

    # 끝까지 한 번도 안 잡힌 관절 등 남은 빈칸은 0
    return np.nan_to_num(features, nan=0.0)


def resample(features: np.ndarray, length: int) -> np.ndarray:
    """(T, 38) → (length, 38) 선형 보간으로 길이 맞추기"""
    t_old = np.linspace(0.0, 1.0, len(features))
    t_new = np.linspace(0.0, 1.0, length)
    return np.stack(
        [np.interp(t_new, t_old, features[:, j]) for j in range(features.shape[1])],
        axis=1,
    )


def sequence_vector(seq: np.ndarray) -> np.ndarray:
    """(length, 38) → 1개 벡터: 시간순 값 전체 + 평균 + 표준편차"""
    return np.concatenate([seq.ravel(), seq.mean(axis=0), seq.std(axis=0)])


def main() -> None:
    args = parse_args()

    # ⑧ 2번 스크립트가 만든 keypoint CSV 읽기
    csv_paths = sorted(KEYPOINT_DIR.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            "먼저 python scripts/referee_02_extract_pose.py 를 실행하세요."
        )

    X, y, groups, stems = [], [], [], []

    for path in csv_paths:
        info = parse_name(path.stem)
        if info is None:
            continue

        df = pd.read_csv(path, encoding="utf-8-sig")

        # 사람이 잡힌 Frame이 너무 적으면 제외
        if (df["people"] > 0).sum() < 5:
            print(f"Skip (사람 검출 부족): {path.stem}")
            continue

        seq = resample(frame_features(df, args.kp_conf), args.frames)

        X.append(sequence_vector(seq))
        y.append(info["action"])
        groups.append(info["subject"])
        stems.append(path.stem)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    subjects = sorted(str(subject) for subject in set(groups))
    labels = sorted(str(label) for label in set(y))

    print(f"영상 {len(X)}개 / 사람 {len(subjects)}명 / 동작 {labels}")
    print(f"입력 벡터 길이: {X.shape[1]}  ({args.frames} Frame × 38 + 평균 38 + 표준편차 38)")
    print()

    # ⑨ 사람 한 명씩 빼서 테스트 (Leave-One-Subject-Out)
    #    → 학습에 없던 사람의 동작도 맞히는지 확인
    predictions = np.empty(len(y), dtype=object)

    print("Test 사람별 결과")
    for subject in subjects:
        test = groups == subject
        train = ~test

        model = RandomForestClassifier(
            n_estimators=args.trees,
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(X[train], y[train])
        predictions[test] = model.predict(X[test])

        acc = accuracy_score(y[test], predictions[test])
        print(f"  {subject}: accuracy={acc:.3f}  ({test.sum()}개)")

    # ⑩ 전체 결과
    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(y, predictions, average="macro")
    matrix = confusion_matrix(y, predictions, labels=labels)

    print()
    print(f"전체 Accuracy: {accuracy:.3f}")
    print(f"Macro F1:      {macro_f1:.3f}")
    print()
    print("Confusion Matrix (행=정답, 열=예측)")
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())

    # ⑪ 결과 저장
    SMOKE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(
        {
            "stem": stems,
            "subject": groups,
            "true": y,
            "pred": predictions,
            "correct": y == predictions,
        }
    )
    result.to_csv(SMOKE_REPORT_DIR / "loso_predictions.csv", index=False, encoding="utf-8-sig")

    wrong = result[~result["correct"]]
    if len(wrong):
        print()
        print("틀린 영상 (원본 영상으로 다시 확인)")
        print(wrong.to_string(index=False))

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, colorbar=False)
    ax.set_title(f"LOSO  acc={accuracy:.3f}  F1={macro_f1:.3f}")
    fig.tight_layout()
    fig.savefig(SMOKE_REPORT_DIR / "confusion_matrix.png", dpi=150)

    with (SMOKE_REPORT_DIR / "metrics.txt").open("w", encoding="utf-8") as file:
        file.write(f"videos={len(X)} subjects={len(subjects)} labels={labels}\n")
        file.write(f"frames={args.frames} kp_conf={args.kp_conf} trees={args.trees}\n")
        file.write(f"accuracy={accuracy:.4f}\nmacro_f1={macro_f1:.4f}\n")

    print()
    print(f"Saved: {SMOKE_REPORT_DIR / 'loso_predictions.csv'}")
    print(f"Saved: {SMOKE_REPORT_DIR / 'confusion_matrix.png'}")
    print(f"Saved: {SMOKE_REPORT_DIR / 'metrics.txt'}")


if __name__ == "__main__":
    main()