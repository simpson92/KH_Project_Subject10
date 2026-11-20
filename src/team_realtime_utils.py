from __future__ import annotations

# ==================================================
# src/team_realtime_utils.py — 1조 7일차(day07_team_*) 공통 함수
# ==================================================
# 실시간 계산(Sliding Window · 30 Step Resampling · RF 예측 · 다수결)은
# 수업 src/realtime_utils.py를 그대로 씁니다 (수정 없음).
# 이 파일에는 "1조라서 달라지는 것"만 모았습니다.
#   - 설정: configs/team01_day07_realtime.json (수업 8개 값 + 1조 경로 · 대상 · 방향)
#   - 모델: 6일차 1조 38-D RF, 클래스 16개는 코드에 적지 않고 rf_training_info.json에서 읽음
#   - 영상: S05(Val)만. S02(Test)는 잠금
#   - 방향: 세로 1080×1920 (A5). 가로로 들어오면 회전
#   - Target: 16개를 숫자키 1~6에 다 못 넣으므로 --targets로 6개씩
# 무거운 라이브러리(ultralytics · torch)는 실제 실행할 때만 불러옵니다 → 테스트는 YOLO 없이 돌아감.
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "team01_day07_realtime.json"
CLASSES_PATH = ROOT / "configs" / "team01_action_classes.csv"
SUBJECT_SPLIT_PATH = ROOT / "configs" / "team01_subject_split.json"
BASELINE_CONFIG_PATH = ROOT / "configs" / "team01_day06_baseline.json"
SPLIT_DIR = ROOT / "reports" / "team_projects" / "team01" / "main" / "dataset" / "splits"

DAY07_DIR = ROOT / "reports" / "team_projects" / "team01" / "main" / "day07"
SAVED_VIDEO_DIR = DAY07_DIR / "saved_video"   # 저장영상 실행 결과 (MP4 · Frame 로그 CSV = Git 제외)
IP_WEBCAM_DIR = DAY07_DIR / "ip_webcam"       # IP Webcam Runtime 로그 (Git 제외)

FEATURE_LIST_FILE = "rf_feature_columns.json"
TRAINING_INFO_FILE = "rf_training_info.json"

# 수업 day07_realtime.json의 8개 값 (이름 · 뜻 그대로)
RUNTIME_KEYS = (
    "window_seconds",
    "min_window_fill_ratio",
    "min_valid_feature_ratio",
    "min_valid_features",
    "prediction_interval_sec",
    "confidence_threshold",
    "stabilization_buffer",
    "stabilization_min_votes",
)
# 1조에서 추가한 값 (경로 · 대상 · 방향)
TEAM_KEYS = ("model_dir", "model_file", "video_dir", "video_subject", "expect_portrait")

ROTATIONS = ("clockwise", "counterclockwise", "none")
MAX_TARGETS = 6                                   # 숫자키 1~6
DAILY_TO_N00 = {"D01", "D02", "D03"}              # 일상 동작 → 정답은 N00
UNTRAINED_TARGET = "X"                            # 학습 안 한 동작 (정답 없음)


# --------------------------------------------------
# 설정
# --------------------------------------------------
def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def test_subjects() -> set[str]:
    return set(load_json(SUBJECT_SPLIT_PATH).get("test_subjects", ["S02"]))


def validate_runtime_config(config: dict) -> list[str]:
    """설정 검사 (수업 41번 validate_runtime_config + 1조 항목). 문제 목록을 돌려줌 (빈 목록 = 통과)."""
    problems = []
    missing = [key for key in RUNTIME_KEYS + TEAM_KEYS if key not in config]
    if missing:
        return [f"설정에 필수 값이 없음: {', '.join(missing)}"]

    if float(config["window_seconds"]) <= 0:
        problems.append("window_seconds는 0보다 커야 함")
    if not 0 < float(config["min_window_fill_ratio"]) <= 1:
        problems.append("min_window_fill_ratio는 0 초과 1 이하")
    if not 0 < float(config["min_valid_feature_ratio"]) <= 1:
        problems.append("min_valid_feature_ratio는 0 초과 1 이하")
    if int(config["min_valid_features"]) < 2:
        problems.append("min_valid_features는 2 이상")
    if float(config["prediction_interval_sec"]) <= 0:
        problems.append("prediction_interval_sec는 0보다 커야 함")
    if not 0 <= float(config["confidence_threshold"]) <= 1:
        problems.append("confidence_threshold는 0~1")
    buffer_size = int(config["stabilization_buffer"])
    if buffer_size < 1:
        problems.append("stabilization_buffer는 1 이상")
    if not 1 <= int(config["stabilization_min_votes"]) <= max(buffer_size, 1):
        problems.append("stabilization_min_votes는 1 이상 · stabilization_buffer 이하")

    # Test 잠금: 저장영상 실험 · 설정 조정은 Val(S05)로만 (결정 기록 D7)
    if str(config["video_subject"]) in test_subjects():
        problems.append(f"video_subject={config['video_subject']}는 Test Subject — 저장영상 실험에 쓸 수 없음")
    if config.get("landscape_rotation", "clockwise") not in ROTATIONS:
        problems.append(f"landscape_rotation은 {ROTATIONS} 중 하나")
    return problems


def load_runtime_config(path: Path = CONFIG_PATH) -> dict:
    config = load_json(path)
    problems = validate_runtime_config(config)
    if problems:
        raise RuntimeError("Day07 설정 오류: " + " / ".join(problems))
    return config


def resolve_settings(config: dict, window_seconds=None, confidence_threshold=None,
                     stabilization_buffer=None) -> dict:
    """명령줄 옵션으로 값 하나만 바꿔 실행할 때 최종 값을 정함 (수업 42 · 44번과 같은 규칙).

    Buffer를 바꾸면 min_votes = buffer // 2 + 1 (과반수) — 3→2, 5→3, 7→4
    """
    window = float(window_seconds) if window_seconds is not None else float(config["window_seconds"])
    confidence = (float(confidence_threshold) if confidence_threshold is not None
                  else float(config["confidence_threshold"]))
    if stabilization_buffer is not None:
        buffer_size = int(stabilization_buffer)
        min_votes = buffer_size // 2 + 1
    else:
        buffer_size = int(config["stabilization_buffer"])
        min_votes = int(config["stabilization_min_votes"])
    return {
        "window_seconds": window,
        "confidence_threshold": confidence,
        "stabilization_buffer": buffer_size,
        "stabilization_min_votes": min_votes,
        "min_window_fill_ratio": float(config["min_window_fill_ratio"]),
        "min_valid_feature_ratio": float(config["min_valid_feature_ratio"]),
        "min_valid_features": int(config["min_valid_features"]),
        "prediction_interval_sec": float(config["prediction_interval_sec"]),
    }


def option_name(settings: dict) -> str:
    """결과 폴더 · 파일 이름: w5_c0.45_s5"""
    return (f"w{settings['window_seconds']:g}_c{settings['confidence_threshold']:g}"
            f"_s{settings['stabilization_buffer']}")


# --------------------------------------------------
# 모델 (6일차 1조 38-D RF)
# --------------------------------------------------
def model_paths(config: dict) -> dict:
    model_dir = ROOT / config["model_dir"]
    return {
        "model": model_dir / config["model_file"],
        "feature_columns": model_dir / FEATURE_LIST_FILE,
        "training_info": model_dir / TRAINING_INFO_FILE,
    }


def check_model_bundle(model, feature_columns: list, training_info: dict,
                       expected_classes: list[str] | None = None) -> list[str]:
    """모델 · Feature 목록 · 학습 정보가 서로 맞는지 (수업 41번 검사, 클래스는 학습 정보에서 읽음)."""
    from src.sequence_utils import sequence_feature_names

    problems = []
    for key in ("classes", "feature_count", "sequence_steps"):
        if key not in training_info:
            problems.append(f"rf_training_info.json에 {key} 없음 — day06_team_train_rf.py 다시 실행")
    if problems:
        return problems

    steps = int(training_info["sequence_steps"])
    expected_count = steps * 38
    if list(feature_columns) != sequence_feature_names(steps):
        problems.append("Feature 이름 · 순서가 sequence_utils.py와 다름")
    if int(training_info["feature_count"]) != expected_count:
        problems.append(f"feature_count {training_info['feature_count']} ≠ {steps} × 38")
    if int(getattr(model, "n_features_in_", -1)) != expected_count:
        problems.append(f"모델 입력 Feature {getattr(model, 'n_features_in_', '?')} ≠ {expected_count}")

    model_classes = [str(c) for c in getattr(model, "classes_", [])]
    info_classes = [str(c) for c in training_info["classes"]]
    if sorted(model_classes) != sorted(info_classes):
        problems.append(f"모델 클래스 {model_classes} ≠ 학습 정보 클래스 {info_classes}")
    if expected_classes is not None and sorted(info_classes) != sorted(expected_classes):
        problems.append(f"학습 정보 클래스가 1조 16클래스와 다름: {info_classes}")
    return problems


def expected_classes() -> list[str]:
    """configs/team01_day06_baseline.json의 labels (A01~A15 + N00)."""
    return [str(c) for c in load_json(BASELINE_CONFIG_PATH)["labels"]]


def load_model_bundle(config: dict, check: bool = True) -> dict:
    import joblib

    paths = model_paths(config)
    for name, path in paths.items():
        if not path.exists():
            hint = " — 0단계: day06_team_train_rf.py로 모델을 먼저 만드세요" if name == "model" else ""
            raise FileNotFoundError(f"{path}{hint}")
    model = joblib.load(paths["model"])
    feature_columns = load_json(paths["feature_columns"])
    training_info = load_json(paths["training_info"])
    if check:
        problems = check_model_bundle(model, feature_columns, training_info, expected_classes())
        if problems:
            raise RuntimeError("6일차 모델 검사 실패: " + " / ".join(problems))
    return {
        "model": model,
        "feature_columns": feature_columns,
        "training_info": training_info,
        "classes": [str(c) for c in training_info["classes"]],
        "sequence_steps": int(training_info["sequence_steps"]),
        "paths": paths,
    }


# --------------------------------------------------
# 클래스 이름 · 혼동 쌍
# --------------------------------------------------
def load_class_table(path: Path = CLASSES_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def english_names(path: Path = CLASSES_PATH) -> dict[str, str]:
    """화면 표시용 짧은 영어 이름 (OpenCV putText는 한글을 못 그림)."""
    names = {}
    for _, row in load_class_table(path).iterrows():
        names[str(row["action_id"])] = str(row["label_en"]).split("/")[0].split(":")[-1].strip()
    names[UNTRAINED_TARGET] = "Untrained action"
    return names


def display_label(label: str, names: dict[str, str]) -> str:
    """'A04' → 'A04 Technical foul', 'UNKNOWN' · 'WARMING_UP'은 그대로."""
    name = names.get(str(label))
    return f"{label} {name}" if name else str(label)


def load_confusable(path: Path = CLASSES_PATH) -> dict[str, set[str]]:
    """action_id → 설계 때 예상한 혼동 상대 (confusable 열, 양방향)."""
    pairs: dict[str, set[str]] = {}
    for _, row in load_class_table(path).iterrows():
        for other in str(row.get("confusable", "")).split():
            pairs.setdefault(str(row["action_id"]), set()).add(other)
            pairs.setdefault(other, set()).add(str(row["action_id"]))
    return pairs


# --------------------------------------------------
# Target (IP Webcam 숫자키)
# --------------------------------------------------
def expected_label_for(target_action: str) -> str:
    """Target 동작 → 모델 기준 정답. D01~D03 → N00, X(학습 안 한 동작) → X (정답 없음)."""
    target_action = str(target_action).strip().upper()
    return "N00" if target_action in DAILY_TO_N00 else target_action


def parse_targets(text: str, classes: list[str]) -> list[dict]:
    """'A01,A02,D03' → [{key:'1', target_action:'A01', expected_label:'A01'}, ...] (최대 6개)."""
    items = [part.strip().upper() for part in str(text).split(",") if part.strip()]
    if not items:
        raise ValueError("--targets가 비어 있음 (예: --targets A01,A02,A03,A04,A05,A06)")
    if len(items) > MAX_TARGETS:
        raise ValueError(f"--targets는 한 번에 {MAX_TARGETS}개까지 (숫자키 1~6). 지금 {len(items)}개")
    if len(set(items)) != len(items):
        raise ValueError(f"--targets에 중복이 있음: {items}")
    allowed = set(classes) | DAILY_TO_N00 | {UNTRAINED_TARGET}
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"알 수 없는 Target {unknown} — 허용: 모델 클래스 · D01~D03 · X")
    return [
        {"key": str(index + 1), "target_action": item, "expected_label": expected_label_for(item)}
        for index, item in enumerate(items)
    ]


# --------------------------------------------------
# 세로 영상 (A5: 1080×1920)
# --------------------------------------------------
def ensure_portrait(frame: np.ndarray, expect_portrait: bool = True,
                    rotation: str = "clockwise") -> tuple[np.ndarray, bool]:
    """높이 > 너비가 아니면 회전. (frame, 회전했는지)를 돌려줌."""
    import cv2

    if not expect_portrait or rotation == "none":
        return frame, False
    height, width = frame.shape[:2]
    if height > width:
        return frame, False
    code = cv2.ROTATE_90_CLOCKWISE if rotation == "clockwise" else cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(frame, code), True


def fit_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    """화면 표시용 축소 (1920 높이는 모니터에 안 들어감). YOLO에는 원본을 넣음."""
    import cv2

    height, width = frame.shape[:2]
    if target_height <= 0 or height <= target_height:
        return frame
    scale = target_height / height
    return cv2.resize(frame, (round(width * scale), int(target_height)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------
# S05 Val 저장영상 목록
# --------------------------------------------------
def val_clips(config: dict, split_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    """splits/val.csv에서 video_subject(S05) clip 목록. 정답은 train_label."""
    split = pd.read_csv(Path(split_dir) / "val.csv", encoding="utf-8-sig", keep_default_na=False)
    subject = str(config["video_subject"])
    clips = split[split["subject_id"].astype(str) == subject].copy()
    video_dir = ROOT / config["video_dir"]
    clips["video_abspath"] = [str(video_dir / name) for name in clips["video_file"]]
    columns = ["clip_id", "subject_id", "action_id", "train_label", "take_id", "video_file", "video_abspath"]
    return clips[columns].sort_values(["action_id", "take_id"]).reset_index(drop=True)


def select_first_take(clips: pd.DataFrame) -> pd.DataFrame:
    """동작마다 첫 Take 1개 → 신호 15 + D01 · D02 · D03 = 18개 (3단계 B)."""
    return clips.groupby("action_id", sort=True).head(1).reset_index(drop=True)


# --------------------------------------------------
# 로그 요약 공통
# --------------------------------------------------
def label_changes(series: pd.Series) -> int:
    """Label이 바뀐 횟수 (수업 43 · 45번과 같은 계산)."""
    clean = series.fillna("").astype(str)
    if clean.empty:
        return 0
    return int((clean != clean.shift(1)).sum() - 1)


def load_pose_backend(project_config: dict):
    """YOLO Pose를 불러와 frame → Result 함수를 돌려줌 (수업 42 · 44번과 같은 설정)."""
    from ultralytics import YOLO

    from src.common import resolve_device

    device = resolve_device(project_config["device"])
    pose_model = YOLO(project_config["pose_model"])

    def predict(frame):
        return pose_model.predict(
            source=frame,
            imgsz=int(project_config["image_size"]),
            conf=float(project_config["person_confidence"]),
            device=device,
            verbose=False,
        )[0]

    return predict
