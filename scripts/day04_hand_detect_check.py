from __future__ import annotations

# ==================================================
# day04_hand_detect_check.py
# 1조 스모크 영상 → MediaPipe Hand Landmarker 손 검출률 측정 (D1 재검토 근거)
# ==================================================
# 질문
#   지금 촬영 조건(세로 · 전신)에서 손 21점을 얼마나 찾는가?
#   화면을 줄이면(= 멀리서 찍으면) 검출률이 얼마나 떨어지는가?
#   → 01 문서 D1 재검토 안건(YOLO만 ↔ YOLO + Hand Landmarker)과
#     03 문서 W1~W4(촬영 거리·해상도)를 숫자로 정하기 위한 실험
#
# 방법 (몸은 지금처럼 YOLO, 손만 MediaPipe — 01 문서 6.2 ②-b 구조)
#   1. 손목 위치 = STEP 4 Sequence CSV의 원본 좌표 raw_*_wrist_x·y (YOLO를 다시 돌리지 않음)
#      Sequence CSV가 없으면 그 clip만 YOLO로 손목을 구함 (--no-yolo 로 끌 수 있음)
#   2. --stride Frame마다 1장씩 꺼내서
#   3. 화면 배율(--scales)마다 영상을 줄이고 (1.0 = 원본, 0.5 = 2배 멀리서 찍은 효과)
#   4. crop : 손목 중심으로 (어깨 폭 × --crop-scale) 정사각형을 잘라 Hand Landmarker
#      full : 화면 전체에 Hand Landmarker (비교용, --modes 로 끌 수 있음)
#   5. 찾은 손의 0번 점(WRIST)이 YOLO 손목과 가까우면 "검출"로 셈
#      (U파울처럼 두 손이 붙어 있으면 crop 안에 손이 2개 → 손목에 더 가까운 손을 고름,
#       좌·우 crop이 같은 손 하나를 잡으면 더 가까운 손목만 인정 → shared = 1)
#
# 준비 (처음 한 번)
#   mkdir -p models
#   wget -O models/hand_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
#   - mediapipe는 pyproject에 넣지 않고 실행할 때만 붙입니다 (uv run --with)
#   - models/*.task 는 Git 제외 (.gitignore)
#
# 실행 순서 (팀 확장 문서 §6처럼 작게 → 크게)
#   uv run --with mediapipe python scripts/day04_hand_detect_check.py --clip-id T01_S01_A03_T01   # ① 1개
#   uv run --with mediapipe python scripts/day04_hand_detect_check.py --subject-id S01            # ② 한 사람 20개
#   uv run --with mediapipe python scripts/day04_hand_detect_check.py                             # ③ 100개 전체
#
# 출력 (Git 제외 — reports/team_projects/team01/tables/*.csv)
#   hand_detect_frames.csv : Frame × 배율 × 방식 × 손(좌/우)마다 한 줄
#   hand_detect_qa.csv     : clip × 방식 × 배율마다 왼손·오른손·양손 검출률
#   화면                    : 동작별·배율별 검출률 표 + 어깨 폭(px) 구간별 검출률 표
#
# 결과 읽는 법
#   - 검출률은 "손목이 YOLO에서 믿을 만할 때(conf ≥ --kp-conf)"만 분모로 셉니다.
#     손목 자체를 못 찾은 Frame은 wrist_status = low · missing 으로 따로 남김
#   - 어깨 폭 px 구간표가 촬영 기준이 됩니다.
#     예) "어깨 폭 80 px 이상이면 crop 검출률 90%" → 그만큼 크게 찍히도록 거리·해상도를 정함
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST_CSV = ROOT / "configs" / "team01_clip_manifest.csv"
VIDEO_DIR = ROOT / "data" / "team_projects" / "team01" / "rgb" / "videos"
SEQUENCE_DIR = ROOT / "reports" / "team_projects" / "team01" / "sequences"
TABLE_DIR = ROOT / "reports" / "team_projects" / "team01" / "tables"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

SIDES = ("left", "right")
MODES = ("crop", "full")
# 어깨 폭(px) 구간 — 화면 표 "어깨 폭이 이만큼 찍히면 검출률은?"
SHOULDER_BINS = [0, 40, 60, 80, 100, 140, 200, 10_000]

FRAME_COLUMNS = [
    "clip_id",
    "subject_id",
    "action_id",
    "action_label",
    "frame_id",
    "time_sec",
    "wrist_source",
    "mode",
    "scale",
    "side",
    "wrist_conf",
    "wrist_status",
    "shoulder_px",
    "crop_px",
    "hands_found",
    "detected",
    "hand_score",
    "match_dist",
    "shared",
]


# ==================================================
# [01] 실행 옵션
# ==================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--video-dir", default=str(VIDEO_DIR))
    parser.add_argument("--sequence-dir", default=str(SEQUENCE_DIR))
    parser.add_argument("--output-dir", default=str(TABLE_DIR))
    parser.add_argument("--model", default=str(MODEL_PATH), help="hand_landmarker.task")
    parser.add_argument("--clip-id", nargs="+", default=None)
    parser.add_argument("--subject-id", nargs="+", default=None)
    parser.add_argument("--action-id", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체")
    parser.add_argument("--stride", type=int, default=5, help="N Frame마다 1장")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.33, 0.25],
        help="화면 배율. 0.5 = 같은 사람을 2배 멀리서 찍은 크기",
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.5,
        help="crop 한 변 = 어깨 폭 × 이 값 (손 길이 ≈ 어깨 폭 × 0.75 → 어느 방향이든 손이 들어감)",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=0.75,
        help="손 0번 점과 YOLO 손목 거리 ≤ 어깨 폭 × 이 값이면 같은 손 "
        "(손 길이 ≈ 어깨 폭 × 0.75. 팔이 겹치면 YOLO 손목이 손에서 어긋남 — 스모크 U파울 실측)",
    )
    parser.add_argument(
        "--same-hand-gap",
        type=float,
        default=0.25,
        help="좌·우 손목이 잡은 손의 0번 점 거리 < 어깨 폭 × 이 값이면 같은 손으로 봄",
    )
    parser.add_argument("--kp-conf", type=float, default=0.50)
    parser.add_argument("--min-hand-conf", type=float, default=0.5)
    parser.add_argument("--no-yolo", action="store_true", help="Sequence CSV 없는 clip은 건너뜀")
    return parser.parse_args(argv)


def select_clips(manifest: pd.DataFrame, args) -> pd.DataFrame:
    selected = manifest
    if args.clip_id:
        unknown = sorted(set(args.clip_id) - set(manifest["clip_id"]))
        if unknown:
            raise SystemExit(f"manifest에 없는 clip_id: {unknown}")
        selected = selected[selected["clip_id"].isin(args.clip_id)]
    if args.subject_id:
        selected = selected[selected["subject_id"].isin(args.subject_id)]
    if args.action_id:
        selected = selected[selected["action_id"].isin(args.action_id)]
    if args.limit > 0:
        selected = selected.head(args.limit)
    return selected


# ==================================================
# [02] Hand Landmarker 감싸기 (테스트에서는 가짜 detector로 바꿔 끼움)
# ==================================================
class Hand:
    """손 1개: points = (21, 2) Pixel 좌표(넣은 이미지 기준), score = 손이라고 믿는 정도."""

    def __init__(self, points: np.ndarray, score: float):
        self.points = points
        self.score = score


class MediaPipeHandDetector:
    """MediaPipe Tasks HandLandmarker (IMAGE 모드). detect(BGR 이미지) → Hand 목록."""

    def __init__(self, model_path: Path, num_hands: int = 2, min_conf: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as error:
            raise SystemExit(
                "mediapipe가 없습니다 → uv run --with mediapipe python scripts/day04_hand_detect_check.py"
            ) from error
        if not Path(model_path).exists():
            raise SystemExit(
                f"모델 파일이 없습니다: {model_path}\n파일 맨 위 '준비' 칸의 wget 명령으로 받으세요."
            )
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr: np.ndarray) -> list[Hand]:
        height, width = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            # MediaPipe 좌표는 0~1 → Pixel로 바꿈
            points = np.array([[lm.x * width, lm.y * height] for lm in landmarks], np.float32)
            score = float(handedness[0].score) if handedness else float("nan")
            hands.append(Hand(points=points, score=score))
        return hands

    def close(self) -> None:
        self._landmarker.close()


# ==================================================
# [03] 손목 정보 (Sequence CSV 한 Row → 좌/우 손목 · 어깨 폭)
# ==================================================
def _number(value) -> float:
    """빈칸·문자 → NaN. Confidence 빈칸을 1.0으로 바꾸지 않기 위해 NaN 그대로 둠."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def shoulder_px(row: dict) -> float:
    """어깨 폭(px). STEP 4의 shoulder_width를 먼저 쓰고, 없으면 원본 어깨 좌표로 계산."""
    width = _number(row.get("shoulder_width"))
    if np.isfinite(width) and width > 0:
        return width
    left = np.array([_number(row.get("raw_left_shoulder_x")), _number(row.get("raw_left_shoulder_y"))])
    right = np.array([_number(row.get("raw_right_shoulder_x")), _number(row.get("raw_right_shoulder_y"))])
    distance = float(np.linalg.norm(left - right))
    return distance if np.isfinite(distance) and distance > 0 else float("nan")


def wrist_of(row: dict, side: str, kp_conf: float) -> tuple[np.ndarray | None, float, str]:
    """(손목 좌표, conf, 상태). 상태: ok · low(conf 낮음) · missing(좌표 없음)."""
    x = _number(row.get(f"raw_{side}_wrist_x"))
    y = _number(row.get(f"raw_{side}_wrist_y"))
    conf = _number(row.get(f"conf_{side}_wrist"))
    if not (np.isfinite(x) and np.isfinite(y)):
        return None, conf, "missing"
    if np.isfinite(conf) and conf < kp_conf:
        return np.array([x, y], np.float32), conf, "low"
    # conf가 빈칸(측정 안 됨)이면 좌표만 믿고 진행 — 표에는 빈칸 그대로 남김
    return np.array([x, y], np.float32), conf, "ok"


# ==================================================
# [04] crop · 같은 손 찾기
# ==================================================
def crop_square(image: np.ndarray, center: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """center 중심 size×size crop. 화면 밖은 검은색으로 채워서 손목이 항상 crop 가운데 오게 함.

    반환: (crop 이미지, crop 왼쪽 위의 원본 좌표)
    """
    size = max(math.floor(size + 0.5), 2)
    half = size // 2
    x0 = math.floor(float(center[0]) + 0.5) - half
    y0 = math.floor(float(center[1]) + 0.5) - half
    height, width = image.shape[:2]
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x0 + size - width), max(0, y0 + size - height)
    inside = image[max(0, y0) : min(height, y0 + size), max(0, x0) : min(width, x0 + size)]
    crop = cv2.copyMakeBorder(
        inside, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    return crop, np.array([x0, y0], np.float32)


def nearest_hand(hands: list[Hand], wrist: np.ndarray, radius: float) -> tuple[Hand | None, float]:
    """손 0번 점(WRIST)이 wrist에 가장 가까운 손. radius보다 멀면 다른 손 → None."""
    best, best_dist = None, float("inf")
    for hand in hands:
        dist = float(np.linalg.norm(hand.points[0] - wrist))
        if dist < best_dist:
            best, best_dist = hand, dist
    if best is None or best_dist > radius:
        return None, best_dist if best is not None else float("nan")
    return best, best_dist


# ==================================================
# [05] Frame 한 장 → 결과 Row들
# ==================================================
def frame_records(frame: np.ndarray, row: dict, base: dict, detector, args) -> list[dict]:
    """배율 × 방식 × 좌/우 손마다 한 Row."""
    records = []
    shoulder = shoulder_px(row)
    wrists = {side: wrist_of(row, side, args.kp_conf) for side in SIDES}
    for scale in args.scales:
        small = frame if scale == 1.0 else cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        shoulder_s = shoulder * scale
        full_hands = None
        for mode in args.modes:
            matched = {}  # side → (Hand, 거리) — 같은 손을 두 손목이 나눠 갖지 않게 확인용
            side_records = {}
            for side in SIDES:
                wrist, conf, status = wrists[side]
                record = dict(base)
                record.update(
                    mode=mode,
                    scale=scale,
                    side=side,
                    wrist_conf=round(conf, 4) if np.isfinite(conf) else "",
                    wrist_status=status,
                    shoulder_px=round(shoulder_s, 2) if np.isfinite(shoulder_s) else "",
                    crop_px="",
                    hands_found="",
                    detected="",
                    hand_score="",
                    match_dist="",
                    shared="",
                )
                side_records[side] = record
                if status == "ok" and not np.isfinite(shoulder_s):
                    record["wrist_status"] = status = "no_scale"
                if status != "ok":
                    continue
                wrist_s = wrist * scale
                radius = shoulder_s * args.match_radius
                if mode == "crop":
                    size = math.floor(shoulder_s * args.crop_scale + 0.5)
                    crop, origin = crop_square(small, wrist_s, size)
                    hands = detector.detect(crop)
                    # crop 좌표 → 줄인 화면 좌표
                    hands = [Hand(points=h.points + origin, score=h.score) for h in hands]
                    record["crop_px"] = crop.shape[0]
                else:
                    if full_hands is None:
                        full_hands = detector.detect(small)
                    hands = full_hands
                hand, dist = nearest_hand(hands, wrist_s, radius)
                record["hands_found"] = len(hands)
                record["detected"] = int(hand is not None)
                record["shared"] = 0
                if hand is not None:
                    record["hand_score"] = round(hand.score, 4)
                    matched[side] = (hand, dist)
                if np.isfinite(dist):
                    # 어깨 폭 단위 거리 (배율과 무관하게 비교 가능)
                    record["match_dist"] = round(dist / shoulder_s, 4)
            # U파울처럼 두 손목이 붙어 있으면 왼쪽·오른쪽 crop에 "같은 손"이 잡힐 수 있음
            # → 그 손은 더 가까운 손목만 가져가고, 다른 쪽은 미검출(shared = 1)로 기록
            if len(matched) == 2:
                (hand_l, dist_l), (hand_r, dist_r) = matched["left"], matched["right"]
                gap = float(np.linalg.norm(hand_l.points[0] - hand_r.points[0]))
                if gap < shoulder_s * args.same_hand_gap:
                    loser = "right" if dist_l <= dist_r else "left"
                    side_records[loser].update(detected=0, hand_score="", shared=1)
            records += [side_records[side] for side in SIDES]
    return records


# ==================================================
# [06] 손목 Row 가져오기 (Sequence CSV 우선, 없으면 YOLO)
# ==================================================
def load_sequence_rows(sequence_dir: Path, clip_id: str) -> dict[int, dict] | None:
    path = sequence_dir / f"{clip_id}_sequence.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return {int(r["frame_id"]): r for r in table.to_dict("records")}


class YoloWrists:
    """Sequence CSV가 없을 때만 사용. STEP 4의 frame_row를 그대로 써서 같은 열을 만듦."""

    def __init__(self, model=None):
        path = ROOT / "scripts" / "day04_build_team_sequences.py"
        spec = importlib.util.spec_from_file_location("day04_build_team_sequences", path)
        self._seq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._seq)
        if model is None:
            from ultralytics import YOLO

            from src.common import load_config, resolve_device

            config = load_config()
            model = YOLO(config["pose_model"])
            self._settings = {
                "imgsz": config["image_size"],
                "conf": config["person_confidence"],
                "device": resolve_device(config["device"]),
            }
        else:
            self._settings = {"imgsz": 640, "conf": 0.25, "device": "cpu"}
        self._model = model

    def row(self, frame: np.ndarray, kp_conf: float) -> dict:
        result = self._model.predict(source=frame, verbose=False, **self._settings)[0]
        return self._seq.frame_row(result, {}, kp_conf, "area")


# ==================================================
# [07] clip 한 개 처리
# ==================================================
def process_clip(clip: dict, detector, args, yolo_factory) -> tuple[list[dict], str]:
    sequence_rows = load_sequence_rows(Path(args.sequence_dir), clip["clip_id"])
    source = "sequence"
    yolo = None
    if sequence_rows is None:
        if args.no_yolo:
            return [], "skip"
        source = "yolo"
        yolo = yolo_factory()
    video_path = Path(args.video_dir) / clip["video_file"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    records = []
    frame_id = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_id % args.stride == 0:
            if yolo is not None:
                row = yolo.row(frame, args.kp_conf)
            else:
                row = sequence_rows.get(frame_id)
            if row is not None:
                base = {
                    "clip_id": clip["clip_id"],
                    "subject_id": clip["subject_id"],
                    "action_id": clip["action_id"],
                    "action_label": clip["action_label"],
                    "frame_id": frame_id,
                    "time_sec": round(frame_id / fps, 4),
                    "wrist_source": source,
                }
                records += frame_records(frame, row, base, detector, args)
        frame_id += 1
    capture.release()
    return records, source


# ==================================================
# [08] 요약
# ==================================================
def summarize(frames: pd.DataFrame) -> pd.DataFrame:
    """clip × 방식 × 배율마다 왼손·오른손·양손 검출률. 분모 = 손목 ok인 Frame."""
    keys = ["clip_id", "subject_id", "action_id", "action_label", "mode", "scale"]
    rows = []
    for key, group in frames.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record["sampled_frames"] = group["frame_id"].nunique()
        wide = {}
        for side in SIDES:
            part = group[group["side"] == side]
            ok = part[part["wrist_status"] == "ok"]
            detected = pd.to_numeric(ok["detected"], errors="coerce")
            record[f"{side}_wrist_ok"] = len(ok)
            record[f"{side}_detect_rate"] = round(float(detected.mean()), 4) if len(ok) else ""
            wide[side] = ok.set_index("frame_id")["detected"].astype(int)
        both = pd.concat(wide, axis=1, join="inner")
        record["both_wrist_ok"] = len(both)
        record["both_detect_rate"] = (
            round(float((both.min(axis=1) == 1).mean()), 4) if len(both) else ""
        )
        scores = pd.to_numeric(group["hand_score"], errors="coerce")
        record["mean_hand_score"] = round(float(scores.mean()), 4) if scores.notna().any() else ""
        shoulder = pd.to_numeric(group["shoulder_px"], errors="coerce")
        record["median_shoulder_px"] = round(float(shoulder.median()), 1) if shoulder.notna().any() else ""
        rows.append(record)
    return pd.DataFrame(rows)


def rate_table(frames: pd.DataFrame, index: str) -> pd.DataFrame:
    """손목 ok인 Row만 → index × (방식, 배율) 검출률 표 (%)."""
    ok = frames[frames["wrist_status"] == "ok"].copy()
    ok["detected"] = pd.to_numeric(ok["detected"])
    table = ok.pivot_table(index=index, columns=["mode", "scale"], values="detected", aggfunc="mean")
    return (table * 100).round(1)


def shoulder_bin_table(frames: pd.DataFrame) -> pd.DataFrame:
    """어깨 폭(px) 구간별 검출률(%)과 표본 수 — 촬영 거리·해상도 기준."""
    ok = frames[frames["wrist_status"] == "ok"].copy()
    ok["detected"] = pd.to_numeric(ok["detected"])
    ok["shoulder_px"] = pd.to_numeric(ok["shoulder_px"])
    ok["shoulder_bin"] = pd.cut(ok["shoulder_px"], SHOULDER_BINS, right=False)
    table = ok.groupby(["shoulder_bin", "mode"], observed=True)["detected"].agg(["mean", "count"])
    table["mean"] = (table["mean"] * 100).round(1)
    return table.rename(columns={"mean": "detect_%", "count": "n"}).unstack("mode")


# ==================================================
# [09] 실행
# ==================================================
def main(argv=None, detector=None, yolo_model=None) -> int:
    args = parse_args(argv)
    if args.stride < 1:
        raise ValueError("--stride는 1 이상이어야 합니다.")
    if any(not 0 < s <= 1 for s in args.scales):
        raise ValueError("--scales는 0보다 크고 1 이하여야 합니다.")
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not Path(args.video_dir).exists():
        raise FileNotFoundError(
            f"{args.video_dir}\n먼저 python scripts/day04_make_clip_manifest.py --apply --link"
        )
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    clips = select_clips(manifest, args)
    if clips.empty:
        raise SystemExit("처리할 clip이 없습니다. 옵션을 확인하세요.")

    own_detector = detector is None
    if own_detector:
        detector = MediaPipeHandDetector(Path(args.model), min_conf=args.min_hand_conf)
    yolo_cache = {}

    def yolo_factory():
        if "yolo" not in yolo_cache:
            yolo_cache["yolo"] = YoloWrists(yolo_model)
        return yolo_cache["yolo"]

    print(
        f"대상 {len(clips)}개 · stride {args.stride} · 배율 {args.scales} · "
        f"방식 {args.modes} · crop = 어깨 폭 × {args.crop_scale}"
    )
    records = []
    try:
        for count, clip in enumerate(clips.to_dict("records"), start=1):
            clip_records, source = process_clip(clip, detector, args, yolo_factory)
            if source == "skip":
                print(f"[SKIP {count}/{len(clips)}] {clip['clip_id']} Sequence CSV 없음 (--no-yolo)")
                continue
            records += clip_records
            part = pd.DataFrame(clip_records)
            ok = part[(part["wrist_status"] == "ok") & (part["mode"] == args.modes[0]) & (part["scale"] == args.scales[0])]
            rate = pd.to_numeric(ok["detected"]).mean() if len(ok) else float("nan")
            print(
                f"[{count}/{len(clips)}] {clip['clip_id']} 손목={source} "
                f"{args.modes[0]}@{args.scales[0]} 검출률 {rate:.0%}"
            )
    finally:
        if own_detector:
            detector.close()

    if not records:
        raise SystemExit("결과가 없습니다.")
    frames = pd.DataFrame(records, columns=FRAME_COLUMNS)
    summary = summarize(frames)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames.to_csv(output_dir / "hand_detect_frames.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "hand_detect_qa.csv", index=False, encoding="utf-8-sig")

    status = frames.drop_duplicates(["clip_id", "frame_id", "side"])["wrist_status"].value_counts()
    print("\n[손목 상태] (Frame × 좌/우)", dict(status))
    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print("\n[동작별 검출률 %] 행 = 동작, 열 = (방식, 화면 배율)")
        print(rate_table(frames, "action_label").to_string())
        print("\n[손별 검출률 %]")
        print(rate_table(frames, "side").to_string())
        print("\n[어깨 폭 px 구간별 검출률 %] → 촬영 거리·해상도 기준")
        print(shoulder_bin_table(frames).to_string())
    print(f"\nSaved: {output_dir / 'hand_detect_frames.csv'}")
    print(f"Saved: {output_dir / 'hand_detect_qa.csv'}")
    print("Team01 hand detect check: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
