from __future__ import annotations

# ==================================================
# team_label_utils.py
# 1조 구간 라벨(labels.csv) 자동 초안 + 검증 — 02_라벨링_가이드 5장 · 6장 · 7장
# ==================================================
# 신호 Take 1개 = 7구간 (02 문서 5.2)
#   N00(시작 차렷) → X(손 들기) → C01(시계 정지) → X(전환) → 신호(A##) → X(복귀) → N00(끝 차렷)
# 일상 동작 D01~D03 = clip 전체 N00 (R-11)
#
# 입력은 STEP 4 Sequence CSV의 정규화 좌표 (골반 중심 0, 어깨 폭 1, y는 아래로 +)
#   - 차렷     : 손목 2개 · 발목 2개가 clip 첫(끝) 5 Frame 자세에서 rest_th 안
#   - C01      : 오른손목이 머리(코·눈·귀 평균)보다 위 + 왼손목은 아래, 가장 높은 지점 근처 구간
#   - 신호     : 전환이 끝나 손이 멈춘 Frame ~ 차렷으로 돌아가는 움직임 직전
#                정지형 신호(A01 · A03 · A04 · A10 · A15)는 가운데 자세와 가까운 구간 (02 문서 5.3)
# 자동 초안이므로 labeler = auto, 이상하면 qa_status = REVIEW. 사람이 고치면 labeler를 S## 로 바꿈
#
# 영상 15개(S04 A01~A15) + 이상값 6개를 Frame 사진으로 대조한 결과 경계 차이 약 0.1~0.3초

import numpy as np
import pandas as pd

FPS_DEFAULT = 30
MIN_SEG_FRAMES = 6          # R-04: 0.2초 미만 구간 금지 (30 FPS)
MIN_SIGNAL_FRAMES = 15      # 0.5초보다 짧은 신호는 사람이 확인
STATIC_SIGNALS = {"A01", "A03", "A04", "A10", "A15"}   # 02 문서 5.3 정지형
SIGNALS = {f"A{i:02d}" for i in range(1, 16)}
ALLOWED_LABELS = SIGNALS | {"C01", "N00", "X"}
QA_VALUES = {"PASS", "REVIEW", "RESHOOT", "EXCLUDE"}
COLUMNS = ["clip_id", "source", "person_id", "label", "start_frame", "end_frame",
           "start_sec", "end_sec", "fps", "labeler", "qa_status", "note"]

HEAD_Y = ["nose_y", "left_eye_y", "right_eye_y", "left_ear_y", "right_ear_y"]
LIMBS = ["right_wrist", "left_wrist", "right_ankle", "left_ankle"]


# ==================================================
# [01] 작은 도구
# ==================================================
def smooth(values, k: int = 5) -> np.ndarray:
    """빈 값을 앞뒤로 채우고 k Frame 중앙값으로 튀는 값 줄이기."""
    s = pd.Series(np.asarray(values, dtype=float)).interpolate(limit_direction="both")
    return s.rolling(k, center=True, min_periods=1).median().to_numpy()


def runs(mask) -> list[tuple[int, int]]:
    """True가 이어지는 구간 [(시작, 끝)] (끝 포함)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def fill_gaps(mask, gap: int) -> np.ndarray:
    """True 사이의 짧은 False(gap Frame 이하)를 메움 — YOLO가 한두 Frame 튀는 것 무시."""
    mask = np.array(mask, dtype=bool)
    rr = runs(mask)
    for (_, e0), (s1, _) in zip(rr, rr[1:]):
        if s1 - e0 - 1 <= gap:
            mask[e0 + 1:s1] = True
    return mask


# ==================================================
# [02] 신호 clip 1개 → 구간 경계
# ==================================================
def segment_signal(df: pd.DataFrame, action_id: str = "", rest_th: float = 0.2,
                   still: float = 0.08, plateau: float = 0.6, cap_tr: int = 18,
                   cap_ret: int = 30, hold_th: float = 0.4) -> dict:
    """경계 Frame 번호를 dict로. C01을 못 찾으면 c01 = None."""
    n = len(df)
    head = smooth(df[HEAD_Y].mean(axis=1))
    P = {k: np.stack([smooth(df[f"{k}_x"]), smooth(df[f"{k}_y"])], 1) for k in LIMBS}
    wrists = np.concatenate([P["right_wrist"], P["left_wrist"]], 1)
    speed = smooth(np.r_[0, np.linalg.norm(np.diff(wrists, axis=0), axis=1)], 5)

    def dist_from(ref: slice) -> np.ndarray:
        return np.max(np.stack(
            [np.hypot(*(P[k] - np.median(P[k][ref], 0)).T) for k in LIMBS], 1), 1)

    d_start = dist_from(slice(0, 5))
    d_end = dist_from(slice(max(n - 5, 0), n))
    d_end_s = smooth(d_end, 9)

    a = 0
    while a < n and d_start[a] < rest_th:
        a += 1
    rest_end = a - 1                      # 시작 차렷 마지막 Frame (-1 = 없음)
    b = n - 1
    while b >= 0 and d_end[b] < rest_th:
        b -= 1
    rest_start = b + 1                    # 끝 차렷 첫 Frame (n = 없음)
    out = {"n": n, "rest_end": rest_end, "rest_start": rest_start, "c01": None, "signal": None}

    # C01: 오른손만 머리 위 → 가장 높은 지점 근처
    ry, ly = P["right_wrist"][:, 1], P["left_wrist"][:, 1]
    up = (ry < head - 0.3) & ~(ly < head - 0.3)
    up[:max(rest_end + 1, 0)] = False
    up = fill_gaps(up, 6)
    for s, e in runs(up):
        if e - s + 1 < MIN_SEG_FRAMES:
            continue
        seg = ry[s:e + 1]
        k = int(np.argmin(seg))
        near = fill_gaps(seg <= seg[k] + plateau, 4)
        lo = k
        while lo - 1 >= 0 and near[lo - 1]:
            lo -= 1
        hi = k
        while hi + 1 < len(seg) and near[hi + 1]:
            hi += 1
        if hi - lo + 1 >= MIN_SEG_FRAMES:
            out["c01"] = (s + lo, s + hi)
            break
    if out["c01"] is None or rest_start <= out["c01"][1] + 1:
        return out
    c01_end = out["c01"][1]

    # 신호 시작: 전환 움직임이 멈춘 첫 Frame (cap_tr Frame 안에서 못 찾으면 그 지점)
    t, moved, sig_s = c01_end + 1, False, None
    while t < rest_start and t <= c01_end + cap_tr:
        if speed[t] >= still:
            moved = True
        elif moved:
            sig_s = t
            break
        t += 1
    if sig_s is None:
        sig_s = min(c01_end + 1 + cap_tr, rest_start - 1)
    # 신호 끝: 끝 차렷에서 거꾸로, 차렷과의 거리가 계속 커지는 동안 = 복귀
    t = rest_start - 1
    lim = max(sig_s + 1, rest_start - cap_ret)
    while t - 1 >= lim and d_end_s[t - 1] > d_end_s[t] + 0.02:
        t -= 1
    sig_e = t - 1

    # 정지형 신호: 가운데 자세와 가까운 Frame 구간으로 다시 잡기
    if action_id in STATIC_SIGNALS and rest_start - c01_end > 12:
        lo, hi = c01_end + 1, rest_start - 1
        mid = (lo + hi) // 2
        ref = {k: np.median(P[k][max(lo, mid - 5):min(hi, mid + 5) + 1], 0) for k in LIMBS}
        d_hold = np.max(np.stack([np.hypot(*(P[k] - ref[k]).T) for k in LIMBS], 1), 1)
        near = d_hold < hold_th
        if near[mid]:
            a_ = mid
            while a_ - 1 >= lo and near[a_ - 1]:
                a_ -= 1
            b_ = mid
            while b_ + 1 <= hi and near[b_ + 1]:
                b_ += 1
            sig_s, sig_e = a_, b_
    if sig_e >= sig_s:
        out["signal"] = (sig_s, sig_e)
    return out


# ==================================================
# [03] 경계 → labels.csv 행
# ==================================================
def _merge_short(segs: list[list]) -> list[list]:
    """R-04: 6 Frame보다 짧은 구간 정리.

    - 짧은 N00 → X (차렷이 너무 짧으면 이동으로 봄)
    - 짧은 X → 옆 N00에 붙임 (없으면 앞 구간에). 신호 · C01은 늘이거나 줄이지 않음
    - 같은 라벨이 이어지면 하나로
    """
    segs = [list(x) for x in segs]
    for seg in segs:
        if seg[0] == "N00" and seg[2] - seg[1] + 1 < MIN_SEG_FRAMES:
            seg[0] = "X"
    segs = _join_same(segs)
    i = 0
    while i < len(segs):
        label, s, e = segs[i]
        if label == "X" and e - s + 1 < MIN_SEG_FRAMES and len(segs) > 1:
            prev = segs[i - 1] if i > 0 else None
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            if nxt is not None and (nxt[0] == "N00" or prev is None):
                nxt[1] = s
            elif prev is not None:
                prev[2] = e
            segs.pop(i)
            segs = _join_same(segs)
            i = 0
            continue
        i += 1
    return segs


def _join_same(segs: list[list]) -> list[list]:
    merged: list[list] = []
    for seg in segs:
        if merged and merged[-1][0] == seg[0]:
            merged[-1][2] = seg[2]
        else:
            merged.append(list(seg))
    return merged


def label_rows(clip_id: str, person_id: str, action_id: str, n_frames: int,
               seg: dict | None, fps: int = FPS_DEFAULT, qa_override: str = "",
               note_override: str = "", action_label: str = "") -> list[dict]:
    """clip 1개의 구간 행 목록 (빈틈 · 겹침 없이 0 ~ n_frames-1)."""
    last = n_frames - 1
    notes: list[str] = []
    if str(action_id).startswith("D"):
        segs = [["N00", 0, last]]
        notes.append(f"일상 동작 {action_id} {action_label}".strip())
        qa = "PASS"
    elif seg is None or seg["c01"] is None or seg["signal"] is None:
        segs = [["X", 0, last]]
        notes.append("자동 구간 찾기 실패 — 사람이 직접 라벨")
        qa = "REVIEW"
    else:
        (c_s, c_e), (s_s, s_e) = seg["c01"], seg["signal"]
        r_end, r_start = seg["rest_end"], seg["rest_start"]
        segs = [["N00", 0, r_end], ["X", r_end + 1, c_s - 1], ["C01", c_s, c_e],
                ["X", c_e + 1, s_s - 1], [action_id, s_s, s_e],
                ["X", s_e + 1, r_start - 1], ["N00", r_start, last]]
        segs = [s for s in segs if s[2] >= s[1]]
        segs = _merge_short(segs)
        qa = "PASS"
        if s_e - s_s + 1 < MIN_SIGNAL_FRAMES:
            qa = "REVIEW"
            notes.append(f"신호 {s_e - s_s + 1} Frame (0.5초 미만)")
        if c_e - c_s + 1 < MIN_SEG_FRAMES:
            qa = "REVIEW"
            notes.append("C01 짧음")
    if qa_override:
        qa = qa_override
    if note_override:
        notes.append(note_override)
    rows = []
    for label, s, e in segs:
        note = ""
        if label == "X":
            note = "자동 초안 · 전환/이동"
        elif label in SIGNALS or label == "C01" or label == "N00":
            note = "자동 초안"
        rows.append({
            "clip_id": clip_id, "source": "webcam", "person_id": person_id, "label": label,
            "start_frame": int(s), "end_frame": int(e),
            "start_sec": round(s / fps, 3), "end_sec": round(e / fps, 3), "fps": int(fps),
            "labeler": "auto", "qa_status": qa, "note": note,
        })
    if notes and rows:
        rows[0]["note"] = " · ".join([rows[0]["note"]] + notes).strip(" ·")
    return rows


def keep_human_rows(new: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    """다시 만들 때 사람이 고친 clip(labeler가 auto가 아닌 행이 있는 clip)은 그대로 둠."""
    if previous is None or previous.empty:
        return new
    human_clips = set(previous.loc[previous["labeler"].astype(str) != "auto", "clip_id"])
    kept = previous[previous["clip_id"].isin(human_clips)]
    rest = new[~new["clip_id"].isin(human_clips)]
    return (pd.concat([kept, rest], ignore_index=True)
            .sort_values(["clip_id", "start_frame"]).reset_index(drop=True))


# ==================================================
# [04] 자동 검증 V-01 ~ V-10 (02 문서 7장)
# ==================================================
def validate_labels(df: pd.DataFrame, frame_counts: dict[str, int] | None = None,
                    for_training: bool = False) -> tuple[list[str], list[str]]:
    """(오류 목록, 경고 목록). 오류가 하나라도 있으면 학습 투입 금지."""
    errors, warnings = [], []
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        return [f"V-01 컬럼 없음: {missing_cols}"], []
    required = [c for c in COLUMNS if c != "note"]
    blank = df[required].astype(str).apply(lambda s: s.str.strip() == "").any(axis=1)
    if blank.any():
        errors.append(f"V-01 빈 값 {int(blank.sum())}행: {df.loc[blank, 'clip_id'].unique()[:5].tolist()}")
    bad_label = ~df["label"].isin(ALLOWED_LABELS)
    if bad_label.any():
        errors.append(f"V-02 허용 밖 label: {sorted(df.loc[bad_label, 'label'].unique())}")
    bad_qa = ~df["qa_status"].isin(QA_VALUES)
    if bad_qa.any():
        errors.append(f"V-01 qa_status 값 오류: {sorted(df.loc[bad_qa, 'qa_status'].unique())}")

    sig_lengths = []
    for clip, g in df.sort_values(["clip_id", "start_frame"]).groupby("clip_id", sort=False):
        s = g["start_frame"].astype(int).to_numpy()
        e = g["end_frame"].astype(int).to_numpy()
        if (e < s).any():
            errors.append(f"V-03 {clip}: end < start")
        if (s[1:] <= e[:-1]).any():
            errors.append(f"V-03 {clip}: 구간 겹침")
        if (s[1:] > e[:-1] + 1).any():
            errors.append(f"V-04 {clip}: 구간 빈틈")
        if s[0] != 0:
            errors.append(f"V-05 {clip}: 첫 구간이 0에서 시작하지 않음")
        if frame_counts and clip in frame_counts and e[-1] != frame_counts[clip] - 1:
            errors.append(f"V-05 {clip}: 마지막 end {e[-1]} ≠ 마지막 Frame {frame_counts[clip] - 1}")
        short = (e - s + 1) < MIN_SEG_FRAMES
        if short.any() and len(g) > 1:
            errors.append(f"V-06 {clip}: 0.2초 미만 구간 {int(short.sum())}개")
        action = clip.split("_")[2] if clip.count("_") >= 3 else ""
        sig = g[g["label"].isin(SIGNALS)]
        if action in SIGNALS and sig.empty:
            warnings.append(f"V-07 {clip}: 신호 구간 없음")
        for _, r in sig.iterrows():
            sig_lengths.append((clip, r["label"], int(r["end_frame"]) - int(r["start_frame"]) + 1))
        fps = g["fps"].astype(float).to_numpy()
        if (np.abs(g["start_sec"].astype(float).to_numpy() - s / fps) > 0.001).any() or \
           (np.abs(g["end_sec"].astype(float).to_numpy() - e / fps) > 0.001).any():
            errors.append(f"V-09 {clip}: sec ≠ frame / fps")

    if sig_lengths:
        sl = pd.DataFrame(sig_lengths, columns=["clip_id", "label", "frames"])
        mean = sl.groupby("label")["frames"].transform("mean")
        odd = sl[(sl["frames"] < 0.5 * mean) | (sl["frames"] > 2 * mean)]
        for _, r in odd.iterrows():
            warnings.append(f"V-08 {r['clip_id']}: {r['label']} {r['frames']} Frame (평균의 0.5배 미만 / 2배 초과)")
    if for_training and (df["qa_status"] == "REVIEW").any():
        errors.append(f"V-10 REVIEW 남음 {df.loc[df['qa_status'] == 'REVIEW', 'clip_id'].nunique()} clip")
    return errors, warnings
