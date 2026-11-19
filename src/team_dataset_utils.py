from __future__ import annotations

# ==================================================
# team_dataset_utils.py
# 1조 ②단계 공통 함수 — Keypoint QA → Human QA → Approved Manifest → Subject-wise Split → Leakage 검사
# ==================================================
# 5일차 실습 scripts/27~31을 1조 데이터에 맞게 옮긴 것입니다. (실습 코드는 그대로 둠)
#
# 실습과 다른 점
#   - 입력: 실습은 26번이 만든 Keypoint CSV, 1조는 STEP 4 Sequence CSV (<clip_id>_sequence.csv)
#       · pose_detected → pose_ok,  person_count → people,  <관절>_conf → conf_<관절>
#   - 동작: 실습 6개(wave 등) → 1조 18개 (A01~A15 신호 + D01~D03 일상 동작)
#   - 학습 라벨: D01~D03 → N00 (일상 동작 하나로 묶음, 03 문서 W2)
#   - manifest review_status = REVIEW 인 clip은 자동 QA 수치가 좋아도 CHECK (사람이 꼭 보게)
#   - Sequence CSV가 아직 없는 clip은 NO_SEQUENCE 로 남김 (STEP 4를 돌리면 채워짐)
#
# 판정 기준 (실습 27번과 같음)
#   pose_ratio < 0.90 · single_person_ratio < 0.90 · mean_keypoint_conf < 0.50 · low_critical_ratio > 0.20
#   → 하나라도 걸리면 CHECK, 아니면 AUTO_PASS

from pathlib import Path

import numpy as np
import pandas as pd

# ==================================================
# [01] 동작별 중요 관절 (COCO 17 이름)
# ==================================================
# 근거: docs/referee/02_라벨링_가이드.md 동작 정의 · 04 문서 5.2 추가 특징(G-01~G-10)
ARMS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
HIPS = ["left_hip", "right_hip"]
LEGS = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
RIGHT_ARM = ["right_shoulder", "right_elbow", "right_wrist"]

ACTION_CRITICAL_KEYPOINTS = {
    "A01": ARMS + HIPS,              # 블로킹: 양손을 허리에 (G-09 손목–골반 거리)
    "A02": ARMS,                     # 홀딩: 손목 잡고 아래로
    "A03": ARMS + ["nose"],          # U파울: 머리 위에서 손목 잡기 (손목–머리 높이)
    "A04": ARMS,                     # 테크니컬: 두 손으로 T
    "A05": ARMS,                     # 푸싱: 두 손을 앞으로 밀기
    "A06": ARMS,                     # 트래블링: 두 주먹 원 운동
    "A07": ARMS,                     # 더블 드리블: 두 손 위아래 교차
    "A08": ARMS,                     # 과도한 팔꿈치: 팔꿈치 휘두름 (어깨 높이)
    "A09": ARMS,                     # 샷클락: 손끝이 어깨에
    "A10": LEGS + RIGHT_ARM,         # 볼 차기: 발 내밀기 + 가리키기 (G-10 발목)
    "A11": ARMS + ["nose"],          # 머리 가격: 손이 머리
    "A12": ARMS + HIPS,              # 후킹: 골반 높이에서 팔 (A08과 높이로 구분)
    "A13": ARMS,                     # 백코트: 팔로 앞뒤 가리키기
    "A14": ARMS,                     # 슈팅 아닌 파울: 손목 내려치기
    "A15": ARMS,                     # 교체: 아래팔 X자 (G-08 손목 교차)
    "D01": ARMS + ["nose"],          # 휘슬: 손을 입으로
    "D02": ARMS,                     # 팔짱
    "D03": ARMS + ["nose"],          # 뒷머리 긁기: 손이 머리 뒤
}

# 신호 15개 + 일상 동작 3개 = clip 단위로 확인하는 18개
EXPECTED_ACTIONS = sorted(ACTION_CRITICAL_KEYPOINTS)

# 판정 기준 (실습 27번)
MIN_POSE_RATIO = 0.90
MIN_SINGLE_PERSON_RATIO = 0.90
MIN_MEAN_CONF = 0.50
MAX_LOW_CRITICAL_RATIO = 0.20

# 사람이 고르는 값 (실습 28번)
ALLOWED_DECISIONS = {"PASS", "RESHOOT", "EXCLUDE"}
HUMAN_COLUMNS = ["human_decision", "human_note"]


# 5일차 수업 Keypoint CSV 형식으로 내보낸 파일 · 영상 위치 (둘 다 data/ 아래라 Git 제외)
KEYPOINT_DIR_REL = "data/team_projects/team01/main/keypoints"
VIDEO_DIR_REL = "data/team_projects/team01/main/rgb/videos"


def keypoint_csv_path(subject_id: str, clip_id: str) -> str:
    """수업 26번처럼 Subject 폴더 아래 clip 하나 = CSV 하나."""
    return f"{KEYPOINT_DIR_REL}/{subject_id}/{clip_id}.csv"


def train_label(action_id: str) -> str:
    """학습에 쓰는 라벨. 일상 동작 D01~D03은 N00 하나로 묶습니다."""
    action_id = str(action_id).upper()
    return "N00" if action_id.startswith("D") else action_id


# ==================================================
# [02] clip 1개 자동 QA (실습 27번 inspect_keypoint_csv)
# ==================================================
def inspect_sequence(df: pd.DataFrame, action_id: str, threshold: float = 0.50) -> dict:
    """STEP 4 Sequence DataFrame 1개 → 자동 QA 수치와 판정.

    - pose_ratio          : pose_ok = 1 인 Frame 비율
    - single_person_ratio : people = 1 인 Frame 비율 (여러 사람이 잡히면 주인공이 바뀔 위험)
    - mean_keypoint_conf  : conf_<관절> 17개 전체 평균 (사람을 못 찾은 Frame의 빈 값은 빼고)
    - low_critical_ratio  : 그 동작의 중요 관절 conf < threshold 인 비율
    """
    critical = ACTION_CRITICAL_KEYPOINTS.get(str(action_id).upper())
    if critical is None:
        raise RuntimeError(f"중요 Keypoint 정의가 없는 Action: {action_id}")

    if df.empty:
        return {
            "qa_status": "CHECK", "qa_reason": "빈 CSV", "frame_count": 0,
            "pose_ratio": 0.0, "single_person_ratio": 0.0, "mean_keypoint_conf": 0.0,
            "low_critical_ratio": 1.0, "missing_pose_frames": 0,
            "critical_keypoints": "|".join(critical),
        }

    pose = pd.to_numeric(df["pose_ok"], errors="coerce").fillna(0)
    people = pd.to_numeric(df["people"], errors="coerce").fillna(0)
    pose_ratio = float(pose.mean())
    single_person_ratio = float((people == 1).mean())

    conf_cols = [c for c in df.columns if c.startswith("conf_")]
    conf = df[conf_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mean_conf = float(np.nanmean(conf)) if np.isfinite(conf).any() else 0.0

    critical_cols = [f"conf_{name}" for name in critical]
    missing = [c for c in critical_cols if c not in df.columns]
    if missing:
        raise RuntimeError("중요 Keypoint conf 열이 없습니다: " + ", ".join(missing))
    values = df[critical_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if valid.sum() > 0:
        low_ratio = float(((values < threshold) & valid).sum() / valid.sum())
    else:
        low_ratio = 1.0

    # 어떤 기준에 걸렸는지 같이 남김 → Human QA 때 무엇을 봐야 하는지 바로 보임
    reasons = []
    if pose_ratio < MIN_POSE_RATIO:
        reasons.append("pose_ratio")
    if single_person_ratio < MIN_SINGLE_PERSON_RATIO:
        reasons.append("single_person")
    if mean_conf < MIN_MEAN_CONF:
        reasons.append("mean_conf")
    if low_ratio > MAX_LOW_CRITICAL_RATIO:
        reasons.append("low_critical")

    return {
        "qa_status": "CHECK" if reasons else "AUTO_PASS",
        "qa_reason": "|".join(reasons),
        "frame_count": int(len(df)),
        "pose_ratio": round(pose_ratio, 4),
        "single_person_ratio": round(single_person_ratio, 4),
        "mean_keypoint_conf": round(mean_conf, 4),
        "low_critical_ratio": round(low_ratio, 4),
        "missing_pose_frames": int((pose == 0).sum()),
        "critical_keypoints": "|".join(critical),
    }


# ==================================================
# [03] manifest 전체 자동 QA
# ==================================================
def build_qa_table(manifest: pd.DataFrame, sequence_dir: Path, threshold: float = 0.50,
                   root: Path | None = None) -> pd.DataFrame:
    """manifest 행마다 Sequence CSV를 찾아 자동 QA. 없으면 NO_SEQUENCE."""
    sequence_dir = Path(sequence_dir)
    rows = []
    for _, item in manifest.iterrows():
        clip_id = str(item["clip_id"])
        csv_path = sequence_dir / f"{clip_id}_sequence.csv"
        if csv_path.exists():
            qa = inspect_sequence(pd.read_csv(csv_path), item["action_id"], threshold)
        else:
            qa = {"qa_status": "NO_SEQUENCE", "qa_reason": "STEP 4 Sequence CSV 없음"}

        # manifest에서 사람이 이미 REVIEW로 표시한 clip은 수치와 관계없이 CHECK
        if str(item.get("review_status", "")).upper() == "REVIEW" and qa["qa_status"] == "AUTO_PASS":
            qa["qa_status"] = "CHECK"
            qa["qa_reason"] = "manifest_review"

        shown = csv_path
        if root is not None:
            try:
                shown = csv_path.resolve().relative_to(Path(root).resolve())
            except ValueError:
                shown = csv_path
        rows.append({
            "clip_id": clip_id,
            "subject_id": item["subject_id"],
            "action_id": item["action_id"],
            "train_label": train_label(item["action_id"]),
            "action_label": item["action_label"],
            "take_id": item["take_id"],
            "capture_set": item.get("capture_set", ""),
            "video_file": item["video_file"],
            "sequence_csv": Path(shown).as_posix(),
            # 5 · 6일차 수업 코드가 읽는 이름 (subject · take · keypoint_csv · video_path)
            "subject": item["subject_id"],
            "take": item["take_id"],
            "keypoint_csv": keypoint_csv_path(item["subject_id"], clip_id),
            "video_path": f"{VIDEO_DIR_REL}/{item['video_file']}",
            "manifest_review": item.get("review_status", ""),
            "manifest_reason": item.get("review_reason", ""),
            **qa,
        })
    table = pd.DataFrame(rows)
    for col in ("frame_count", "missing_pose_frames"):
        if col in table:
            table[col] = table[col].astype("Int64")  # NO_SEQUENCE 빈칸이 있어도 134.0이 아니라 134로
    return table


def merge_previous_review(qa: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    """Human QA 파일을 다시 만들 때 이미 적은 human_decision · human_note를 지키기.

    실습 27번은 돌릴 때마다 UNREVIEWED로 덮어씀 → 1조는 450개라 다시 적기 어려움
    """
    review = qa.copy()
    review["human_decision"] = "UNREVIEWED"
    review["human_note"] = ""
    if previous is None or previous.empty:
        return review
    keep = previous.set_index("clip_id")[HUMAN_COLUMNS]
    for col in HUMAN_COLUMNS:
        mapped = review["clip_id"].map(keep[col])
        review[col] = mapped.where(mapped.notna() & (mapped.astype(str) != ""), review[col])
    return review


# ==================================================
# [04] Human QA 검사 (실습 28번)
# ==================================================
def normalize_decisions(df: pd.DataFrame) -> pd.Series:
    return df["human_decision"].fillna("").astype(str).str.strip().str.upper()


def validate_review(df: pd.DataFrame, allow_no_sequence: bool = False) -> list[str]:
    """문제 목록을 돌려줌 (빈 list = 통과)."""
    errors = []
    decisions = normalize_decisions(df)
    unreviewed = df.loc[(decisions == "") | (decisions == "UNREVIEWED"), "clip_id"].tolist()
    if unreviewed:
        errors.append(f"UNREVIEWED {len(unreviewed)}개: " + ", ".join(unreviewed[:10]))
    invalid_mask = ~decisions.isin(ALLOWED_DECISIONS | {"", "UNREVIEWED"})
    if invalid_mask.any():
        errors.append("PASS / RESHOOT / EXCLUDE 가 아닌 값: "
                      + ", ".join(df.loc[invalid_mask, "clip_id"].tolist()))
    # CHECK인데 이유 없이 PASS → 왜 통과시켰는지 남겨야 함
    no_note = df.loc[(df["qa_status"] == "CHECK") & (decisions == "PASS")
                     & (df["human_note"].fillna("").astype(str).str.strip() == ""), "clip_id"].tolist()
    if no_note:
        errors.append("CHECK인데 human_note 없이 PASS: " + ", ".join(no_note))
    if not allow_no_sequence:
        missing = df.loc[df["qa_status"] == "NO_SEQUENCE", "clip_id"].tolist()
        if missing:
            errors.append(f"NO_SEQUENCE {len(missing)}개 (STEP 4 먼저): " + ", ".join(missing[:10]))
    dup = df.loc[df["clip_id"].duplicated(), "clip_id"].tolist()
    if dup:
        errors.append("clip_id 중복: " + ", ".join(dup))
    return errors


# ==================================================
# [05] Approved Manifest (실습 29번)
# ==================================================
def build_approved(review: pd.DataFrame) -> pd.DataFrame:
    """human_decision = PASS 이고 Sequence CSV가 있는 clip만."""
    decisions = normalize_decisions(review)
    approved = review[(decisions == "PASS") & (review["qa_status"] != "NO_SEQUENCE")].copy()
    if approved.empty:
        raise RuntimeError("PASS 데이터가 없습니다.")
    return approved.reset_index(drop=True)


# ==================================================
# [06] Subject-wise Split (실습 30번)
# ==================================================
def validate_subject_sets(config: dict) -> list[str]:
    train, val, test = (set(config[k]) for k in ("train_subjects", "val_subjects", "test_subjects"))
    errors = []
    if train & val:
        errors.append(f"Train / Validation Subject 중복: {sorted(train & val)}")
    if train & test:
        errors.append(f"Train / Test Subject 중복: {sorted(train & test)}")
    if val & test:
        errors.append(f"Validation / Test Subject 중복: {sorted(val & test)}")
    return errors


def missing_actions(df: pd.DataFrame, expected: list[str] | set[str]) -> list[str]:
    actual = set(df["action_id"].astype(str).str.upper())
    return sorted(set(expected) - actual)


def split_by_subject(approved: pd.DataFrame, config: dict,
                     expected_actions=EXPECTED_ACTIONS) -> dict[str, pd.DataFrame]:
    """사람 단위로 train / val / test 나누기. 문제가 있으면 RuntimeError."""
    errors = validate_subject_sets(config)
    configured = set(config["train_subjects"] + config["val_subjects"] + config["test_subjects"])
    data_subjects = set(approved["subject_id"].astype(str))
    if data_subjects - configured:
        errors.append(f"Split에 배정되지 않은 Subject: {sorted(data_subjects - configured)}")
    if configured - data_subjects:
        errors.append(f"Approved 데이터에 없는 Subject: {sorted(configured - data_subjects)}")

    splits = {}
    for name, key in (("train", "train_subjects"), ("val", "val_subjects"), ("test", "test_subjects")):
        part = approved[approved["subject_id"].isin(config[key])].copy()
        part.insert(0, "split", name)
        miss = missing_actions(part, expected_actions)
        if miss:
            errors.append(f"{name}에 없는 Action: {', '.join(miss)}")
        splits[name] = part.reset_index(drop=True)
    if errors:
        raise RuntimeError(" / ".join(errors))
    return splits


def split_counts(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """split × 학습 라벨(16개) clip 수 표 — 보고서용."""
    rows = []
    for name, df in splits.items():
        counts = df.groupby("train_label").size()
        row = {"split": name, "subjects": "|".join(sorted(df["subject_id"].unique())),
               "clips": len(df)}
        row.update(counts.to_dict())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


# ==================================================
# [07] Leakage 검사 (실습 31번)
# ==================================================
def check_leakage(splits: dict[str, pd.DataFrame], expected_actions=EXPECTED_ACTIONS) -> list[str]:
    errors = []
    for name, df in splits.items():
        miss = missing_actions(df, expected_actions)
        if miss:
            errors.append(f"{name} Action 누락: {', '.join(miss)}")
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            subj = set(splits[a]["subject_id"]) & set(splits[b]["subject_id"])
            if subj:
                errors.append(f"{a}/{b} Subject Leakage: {sorted(subj)}")
            clips = set(splits[a]["clip_id"]) & set(splits[b]["clip_id"])
            if clips:
                errors.append(f"{a}/{b} Clip Leakage: {sorted(clips)[:5]}")
            videos = set(splits[a]["video_file"]) & set(splits[b]["video_file"])
            if videos:
                errors.append(f"{a}/{b} 같은 영상 파일: {sorted(videos)[:5]}")
    return errors
