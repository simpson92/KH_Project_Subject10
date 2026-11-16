from __future__ import annotations

# ==================================================
# day03_plot_feature_timeline.py
# 18번 CSV로 Feature 타임라인 그래프 + 튐(Spike) Frame 탐지
# ==================================================
# feature_design.md 6장의 그래프(team_feature_timeline.png)를
# 누구나 같은 결과로 다시 만들 수 있게 한 코드입니다.
#
# 하는 일
#   1. 18번이 만든 *_frame_features.csv를 1~4개 읽음 (영상 1개 = 선 1개)
#   2. Feature마다 그래프 한 칸 (x축 = 시간, 모든 칸이 같은 시간축)
#      기본: wrist − nose(손목이 코보다 위면 음수) · 팔꿈치 각도 · 어깨 폭
#   3. 각도 Feature에서 바로 앞 Frame과의 차이 |Δ|가 기준(--spike-deg)
#      이상인 Frame을 "튐"으로 표시하고 CSV로 저장
#
# "튐"은 Pose 오류일 수도, 실제로 빠른 동작일 수도 있습니다.
# 이 코드는 확인할 Frame 목록을 만들 뿐이고, 판정은 영상을 보고 사람이 합니다.
# (4일차 sequence_qa.csv의 REVIEW 후보로 사용)
# 한 Frame만 튀면 "튀는 순간"과 "돌아오는 순간" 2개가 함께 잡힙니다.
#
# 실행 예
#   python scripts/day03_plot_feature_timeline.py \
#     --csv reports/day03/tables/S01_ufoul_T01_frame_features.csv \
#           reports/day03/tables/S01_pushing_T01_frame_features.csv
#
# 결과 (둘 다 Git 제외 경로)
#   reports/day03/images/feature_timeline.png
#   reports/day03/tables/feature_spikes.csv

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 PNG로 저장 (WSL·서버에서도 동작)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ==================================================
# [01] 색 · 글자 설정
# ==================================================
# 영상(=series) 색은 고정 순서로 배정합니다. 4개를 넘기면 거절합니다.
# (색 4개는 색각이상 시뮬레이션 검사 통과, 노랑·청록은 밝아서 선 끝 이름표를 함께 표시)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# 각도처럼 "Δ가 크면 튐"으로 볼 Feature (단위 °)
ANGLE_FEATURES = {
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
}

# CSV에 없는 파생 Feature (상대 위치) — 이름: (A열, B열) → A − B
DERIVED_FEATURES = {
    "left_wrist_minus_nose": ("left_wrist_y", "nose_y"),
    "right_wrist_minus_nose": ("right_wrist_y", "nose_y"),
    "left_wrist_minus_shoulder": ("left_wrist_y", "left_shoulder_y"),
    "right_wrist_minus_shoulder": ("right_wrist_y", "right_shoulder_y"),
}

PANEL_TITLES = {
    "left_wrist_minus_nose": "왼손목 y − 코 y (음수 = 손목이 코보다 위)",
    "right_wrist_minus_nose": "오른손목 y − 코 y (음수 = 손목이 코보다 위)",
    "left_wrist_minus_shoulder": "왼손목 y − 왼어깨 y (음수 = 손목이 어깨보다 위)",
    "right_wrist_minus_shoulder": "오른손목 y − 오른어깨 y",
    "left_elbow_angle": "왼팔꿈치 각도 (°)",
    "right_elbow_angle": "오른팔꿈치 각도 (°)",
    "left_knee_angle": "왼무릎 각도 (°)",
    "right_knee_angle": "오른무릎 각도 (°)",
    "shoulder_width": "어깨 폭 (px) — 정규화 기준값",
}

DEFAULT_FEATURES = ["left_wrist_minus_nose", "left_elbow_angle", "shoulder_width"]


# ==================================================
# [02] 실행 옵션
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="18번 *_frame_features.csv (1~4개)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="그릴 Feature (CSV 열 이름 또는 파생 이름)",
    )
    parser.add_argument(
        "--spike-deg",
        type=float,
        default=40.0,
        help="각도 Feature에서 앞 Frame 대비 |Δ| 이 값 이상이면 튐",
    )
    parser.add_argument("--output", default="feature_timeline.png")
    parser.add_argument("--spikes-output", default="feature_spikes.csv")
    return parser.parse_args()


# ==================================================
# [03] CSV 읽기 + 파생 Feature 추가
# ==================================================
def load_frames(path: Path) -> pd.DataFrame:
    """18번 CSV를 읽고 valid_pose = 1인 Frame만 남긴 뒤 파생 Feature를 붙입니다."""
    df = pd.read_csv(path)
    missing = {"frame_id", "time_sec", "valid_pose"} - set(df.columns)
    if missing:
        raise ValueError(
            f"18번 CSV 형식이 아닙니다 ({path.name}): {sorted(missing)} 없음"
        )

    valid = df[df["valid_pose"] == 1].copy()
    for name, (a, b) in DERIVED_FEATURES.items():
        if a in valid.columns and b in valid.columns:
            valid[name] = valid[a] - valid[b]
    return valid.reset_index(drop=True)


def series_label(path: Path) -> str:
    # S01_ufoul_T01_frame_features.csv → S01_ufoul_T01
    return path.stem.removesuffix("_frame_features")


# ==================================================
# [04] 튐(Spike) 탐지
# ==================================================
def detect_spikes(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """바로 앞의 유효 Frame과 비교해 |Δ| >= threshold 인 Frame을 반환합니다.

    - pandas diff()는 "현재 행 − 바로 앞 행"
    - valid_pose = 0 Frame은 이미 빠져 있으므로,
      앞 Frame이 몇 Frame 전인지(frame_gap)도 기록
      (gap이 크면 Pose가 끊겼다가 돌아온 순간일 수 있음)
    """
    if column not in df.columns:
        raise KeyError(column)

    delta = df[column].diff()
    gap = df["frame_id"].diff()
    mask = delta.abs() >= threshold

    out = df.loc[mask, ["frame_id", "time_sec", column]].copy()
    out = out.rename(columns={column: "value"})
    out.insert(0, "feature", column)
    out["prev_value"] = df[column].shift(1)[mask]
    out["delta"] = delta[mask]
    out["frame_gap"] = gap[mask].astype("Int64")
    return out.reset_index(drop=True)


# ==================================================
# [05] 그래프
# ==================================================
def setup_korean_font() -> None:
    # 한글 제목이 □로 깨지지 않도록 설치된 한글 글꼴을 찾습니다.
    from matplotlib import font_manager

    candidates = [
        "Malgun Gothic",  # Windows
        "NanumGothic",  # WSL: sudo apt install fonts-nanum
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "AppleGothic",  # macOS
    ]
    for name in candidates:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def place_end_labels(ax, end_labels, min_gap=0.09) -> None:
    """선 끝 이름표를 그래프 오른쪽 바깥에 겹치지 않게 놓습니다.

    값이 비슷하면(예: 모든 영상이 팔을 내린 자세로 끝남) 위아래로 min_gap씩 벌립니다.
    위치는 축 비율(0~1) 기준이라 그림 크기와 상관없이 같은 간격이 됩니다.
    """
    if not end_labels:
        return
    low, high = ax.get_ylim()
    items = sorted(((v - low) / (high - low), label) for v, label in end_labels)
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):  # 아래에서 위로 밀어 올리기
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - 1.0
    if overflow > 0:  # 맨 위가 넘치면 전체를 내림
        ys = [y - overflow for y in ys]
    for y, (_, label) in zip(ys, items):
        ax.text(
            1.005,
            y,
            label,
            transform=ax.transAxes,
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )


def plot_timeline(series, features, spikes, threshold, output_path: Path) -> None:
    setup_korean_font()
    fig, axes = plt.subplots(
        len(features),
        1,
        figsize=(11, 2.8 * len(features)),
        sharex=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)

    for ax, feature in zip(axes[:, 0], features):
        ax.set_facecolor(SURFACE)
        end_labels = []
        for (label, df), color in zip(series, SERIES_COLORS):
            if feature not in df.columns:
                continue
            ax.plot(df["time_sec"], df[feature], color=color, linewidth=2, label=label)

            # 선 끝 이름표 (색만으로 영상을 구분하지 않도록)
            # 겹치지 않게 아래 place_end_labels()에서 한꺼번에 배치
            end_labels.append((float(df[feature].dropna().iloc[-1]), label))

            # 튐 Frame: 같은 색 점 + 바탕색 테두리
            hit = spikes[(spikes["series"] == label) & (spikes["feature"] == feature)]
            if not hit.empty:
                ax.scatter(
                    hit["time_sec"],
                    hit["value"],
                    s=64,
                    color=color,
                    edgecolors=SURFACE,
                    linewidths=2,
                    zorder=3,
                )

        if feature.endswith("_minus_nose") or feature.endswith("_minus_shoulder"):
            # 0 선 = 손목이 기준 관절과 같은 높이 (부호가 바뀌는 곳)
            ax.axhline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")

        place_end_labels(ax, end_labels)

        title = PANEL_TITLES.get(feature, feature)
        if feature in ANGLE_FEATURES:
            title += f"  ·  점 = 앞 Frame 대비 |Δ| ≥ {threshold:g}°"
        ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    axes[-1, 0].set_xlabel("시간 (초)", color=TEXT_SECONDARY)
    if len(series) >= 2:
        axes[0, 0].legend(
            loc="upper left",
            bbox_to_anchor=(0, 1.35),
            ncol=len(series),
            frameon=False,
            fontsize=9,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ==================================================
# [06] 실행
# ==================================================
def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.csv]
    if len(paths) > len(SERIES_COLORS):
        raise SystemExit(f"영상은 한 번에 {len(SERIES_COLORS)}개까지 비교합니다.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    series = [(series_label(p), load_frames(p)) for p in paths]

    for feature in args.features:
        if not any(feature in df.columns for _, df in series):
            raise SystemExit(f"Feature를 찾을 수 없습니다: {feature}")

    # 각도 Feature만 튐 탐지
    spike_tables = []
    for label, df in series:
        for feature in args.features:
            if feature in ANGLE_FEATURES and feature in df.columns:
                table = detect_spikes(df, feature, args.spike_deg)
                table.insert(0, "series", label)
                spike_tables.append(table)
    columns = ["series", "feature", "frame_id", "time_sec", "value"]
    columns += ["prev_value", "delta", "frame_gap"]
    spikes = (
        pd.concat(spike_tables, ignore_index=True)
        if spike_tables
        else pd.DataFrame(columns=columns)
    )

    image_path = ROOT / "reports" / "day03" / "images" / args.output
    spikes_path = ROOT / "reports" / "day03" / "tables" / args.spikes_output
    plot_timeline(series, args.features, spikes, args.spike_deg, image_path)
    spikes_path.parent.mkdir(parents=True, exist_ok=True)
    spikes.round(3).to_csv(spikes_path, index=False, encoding="utf-8-sig")

    # 요약 출력 (그래프를 못 보는 환경에서도 결과 확인)
    print(f"Spike threshold: |Δ| >= {args.spike_deg:g}°")
    for label, df in series:
        n = len(df)
        hit = spikes[spikes["series"] == label]
        print(f"{label:24s} valid frames {n:4d} · spikes {len(hit):3d}")
        for _, s in hit.iterrows():
            print(
                f"    {s['feature']:18s} frame {int(s['frame_id']):4d}"
                f" ({s['time_sec']:.2f}s)  {s['prev_value']:7.2f}° → {s['value']:7.2f}°"
                f"  Δ {s['delta']:+7.2f}"
            )
    print(f"Saved: {image_path}")
    print(f"Saved: {spikes_path}")


if __name__ == "__main__":
    main()
