from __future__ import annotations

# ==================================================
# team_crosscheck_utils.py
# 구간 라벨 교차 검수 (02_라벨링_가이드 8장 5단계) — 표본 고르기 · 검수표 비교 · 고친 값 반영
# ==================================================
# 흐름
#   1) pick_sample      : 451 clip 중 10%(45개) — 자동 라벨이 의심스러운 clip을 먼저, 나머지는 동작별 고르게
#   2) assign_reviewers : 5명에게 9개씩. **자기 영상은 자기가 보지 않음** (교차)
#   3) 검수자는 라벨이 안 보이는 모아보기(blind)에서 C01 · 신호의 시작/끝 Frame을 직접 적음
#      verdict: OK = 동작 정상 · 4칸 적음 / RESHOOT = 동작이 틀림 · 잘림
#      → 비교 후 차이가 큰 clip은 PM이 라벨 보이는 모아보기로 누가 맞는지 보고, 검수자가 맞으면 FIX로 바꿈
#   4) compare          : 자동 라벨과 경계 차이(초) — 평균 0.2초 넘으면 기준 다시 맞추기 (02 문서 8장)
#   5) apply_fixes      : FIX로 적은 clip은 검수자 값으로 구간을 다시 만들고 labeler = 검수자 S##

import numpy as np
import pandas as pd

from src.team_label_utils import SIGNALS, label_rows

SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
BOUNDARIES = ["c01_start", "c01_end", "sig_start", "sig_end"]
REVIEW_COLUMNS = ["reviewer", "no", "clip_id", "subject_id", "action_id", "action_label", "frames",
                  "sheet_file", "why", "review_c01_start", "review_c01_end",
                  "review_sig_start", "review_sig_end", "verdict", "note"]
VERDICTS = {"OK", "FIX", "RESHOOT"}
THRESHOLD_SEC = 0.2


def auto_boundaries(labels: pd.DataFrame) -> pd.DataFrame:
    """labels.csv → clip별 자동 경계 (C01 · 신호 시작/끝 Frame, 없으면 NaN)."""
    rows = []
    for clip, g in labels.groupby("clip_id"):
        c01 = g[g["label"] == "C01"]
        sig = g[g["label"].isin(SIGNALS)]
        rows.append({
            "clip_id": clip,
            "c01_start": c01["start_frame"].min() if len(c01) else np.nan,
            "c01_end": c01["end_frame"].max() if len(c01) else np.nan,
            "sig_start": sig["start_frame"].min() if len(sig) else np.nan,
            "sig_end": sig["end_frame"].max() if len(sig) else np.nan,
            "qa_status": g["qa_status"].iloc[0],
            "frames": int(g["end_frame"].max()) + 1,
        })
    return pd.DataFrame(rows)


def pick_sample(labels: pd.DataFrame, warn_clips: list[str], n_total: int = 45,
                n_daily: int = 3, seed: int = 42) -> pd.DataFrame:
    """교차 검수 표본. why = 고른 이유.

    1) REVIEW · 검증 경고(V-07 · V-08) clip  2) 신호 15개 × 2  3) 일상 동작 D01~D03 각 1  4) 남은 칸은 헷갈리기 쉬운 동작
    """
    rng = np.random.default_rng(seed)
    info = auto_boundaries(labels)
    info = info[info["qa_status"] != "EXCLUDE"].copy()
    info["subject_id"] = info["clip_id"].str.split("_").str[1]
    info["action_id"] = info["clip_id"].str.split("_").str[2]
    picked: dict[str, str] = {}
    for clip in info.loc[info["qa_status"] == "REVIEW", "clip_id"]:
        picked[clip] = "자동 라벨 REVIEW"
    for clip in warn_clips:
        if clip in set(info["clip_id"]) and clip not in picked:
            picked[clip] = "검증 경고 (신호 길이가 평균과 많이 다름)"

    def take(pool: pd.DataFrame, k: int, why: str):
        pool = pool[~pool["clip_id"].isin(picked)]
        if pool.empty or k <= 0:
            return
        # 사람이 겹치지 않게: 사람별로 섞은 뒤 돌아가며 1개씩
        order = pool.sample(frac=1, random_state=int(rng.integers(1_000_000)))
        chosen, used = [], set()
        for _, r in order.iterrows():
            if r["subject_id"] not in used:
                chosen.append(r["clip_id"])
                used.add(r["subject_id"])
            if len(chosen) == k:
                break
        for clip in chosen:
            picked[clip] = why

    signals = sorted(a for a in info["action_id"].unique() if a in SIGNALS)
    for a in signals:
        have = sum(1 for c in picked if c.split("_")[2] == a)
        take(info[info["action_id"] == a], max(0, 2 - have), "동작별 고르게 (신호 15개 × 2)")
    for d in ["D01", "D02", "D03"][:n_daily]:
        take(info[info["action_id"] == d], 1, "일상 동작 — 신호가 섞이지 않았는지")
    # 헷갈리기 쉬운 동작 (03 문서 혼동 쌍 + 정지형 신호 경계)
    extra = ["A08", "A12", "A09", "A11", "A03", "A15", "A04", "A01", "A10", "A13"]
    i = 0
    while len(picked) < n_total and i < 50:
        take(info[info["action_id"] == extra[i % len(extra)]], 1, "헷갈리기 쉬운 동작 · 정지형 경계")
        i += 1
    out = info[info["clip_id"].isin(picked)].copy()
    out["why"] = out["clip_id"].map(picked)
    return out.sort_values("clip_id").reset_index(drop=True)


def assign_reviewers(sample: pd.DataFrame, reviewers=SUBJECTS, seed: int = 42) -> pd.DataFrame:
    """검수자 배정 — 자기 영상 제외, 사람마다 같은 개수(9), 같은 동작은 서로 다른 검수자에게.

    동작 순서로 줄 세운 뒤 검수자를 돌아가며 배정 (자기 영상 · 꽉 찬 사람은 건너뜀).
    마지막에 남는 칸이 자기 영상뿐이면 다른 검수자와 맞바꿈.
    """
    rng = np.random.default_rng(seed)
    per = int(np.ceil(len(sample) / len(reviewers)))
    clips = sample.sample(frac=1, random_state=seed).sort_values("action_id", kind="stable").reset_index(drop=True)
    load = {r: 0 for r in reviewers}
    who: list[str | None] = []
    turn = int(rng.integers(len(reviewers)))
    for _, r in clips.iterrows():
        pick = None
        for step in range(len(reviewers)):
            cand = reviewers[(turn + step) % len(reviewers)]
            if cand != r["subject_id"] and load[cand] < per:
                pick = cand
                turn = (turn + step + 1) % len(reviewers)
                break
        if pick is None:   # 남은 자리가 자기 영상 사람뿐 → 이미 배정된 것과 맞바꿈
            free = next(x for x in reviewers if load[x] < per)
            for k, other in enumerate(who):
                if other not in (None, r["subject_id"]) and clips.loc[k, "subject_id"] != free:
                    who[k], pick = free, other
                    load[free] += 1
                    load[other] -= 1
                    break
        who.append(pick)
        load[pick] += 1
    out = clips.copy()
    out["reviewer"] = who
    if (out["reviewer"] == out["subject_id"]).any() or out["reviewer"].isna().any():
        raise RuntimeError("자기 영상을 피해서 배정할 수 없음")
    return out.sort_values(["reviewer", "clip_id"]).reset_index(drop=True)


def blank_review_sheet(assigned: pd.DataFrame, action_labels: dict[str, str]) -> pd.DataFrame:
    """검수자에게 줄 빈 표 (자동 경계는 넣지 않음 = blind)."""
    rows = []
    for reviewer, g in assigned.groupby("reviewer"):
        for no, (_, r) in enumerate(g.iterrows(), 1):
            rows.append({
                "reviewer": reviewer, "no": no, "clip_id": r["clip_id"], "subject_id": r["subject_id"],
                "action_id": r["action_id"], "action_label": action_labels.get(r["action_id"], ""),
                "frames": int(r["frames"]), "sheet_file": f"{r['clip_id']}.jpg", "why": r["why"],
                "review_c01_start": "", "review_c01_end": "", "review_sig_start": "", "review_sig_end": "",
                "verdict": "", "note": "",
            })
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def compare(review: pd.DataFrame, labels: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """검수자 경계 ↔ 자동 경계 차이(초). 일상 동작 · 빈칸은 비교에서 뺌."""
    auto = auto_boundaries(labels).set_index("clip_id")
    rows = []
    for _, r in review.iterrows():
        clip = r["clip_id"]
        row = {"reviewer": r["reviewer"], "clip_id": clip, "action_id": r["action_id"],
               "verdict": str(r.get("verdict", "")).strip().upper()}
        diffs = []
        for b in BOUNDARIES:
            mine = pd.to_numeric(r.get(f"review_{b}", ""), errors="coerce")
            theirs = auto.loc[clip, b] if clip in auto.index else np.nan
            d = abs(mine - theirs) / fps if np.isfinite(mine) and np.isfinite(theirs) else np.nan
            row[f"diff_{b}_sec"] = round(float(d), 3) if np.isfinite(d) else np.nan
            if np.isfinite(d):
                diffs.append(d)
        row["mean_diff_sec"] = round(float(np.mean(diffs)), 3) if diffs else np.nan
        row["over_0_2s"] = bool(diffs) and float(np.mean(diffs)) > THRESHOLD_SEC
        rows.append(row)
    return pd.DataFrame(rows)


def validate_review_sheet(review: pd.DataFrame) -> list[str]:
    """검수표 기입 오류 (빈 verdict, 신호 clip인데 경계 빈칸, 순서가 거꾸로)."""
    errors = []
    for _, r in review.iterrows():
        v = str(r.get("verdict", "")).strip().upper()
        clip = r["clip_id"]
        if v not in VERDICTS:
            errors.append(f"{clip}: verdict가 OK / FIX / RESHOOT가 아님 ({v or '빈칸'})")
            continue
        if v in ("OK", "FIX") and r["action_id"] in SIGNALS:
            vals = [pd.to_numeric(r.get(f"review_{b}", ""), errors="coerce") for b in BOUNDARIES]
            if not all(np.isfinite(vals)):
                errors.append(f"{clip}: 신호 clip인데 Frame 번호 4칸 중 빈칸이 있음")
            elif not (vals[0] <= vals[1] < vals[2] <= vals[3] < int(r["frames"])):
                errors.append(f"{clip}: Frame 순서가 C01 시작 ≤ C01 끝 < 신호 시작 ≤ 신호 끝 < {r['frames']} 이 아님")
    return errors


def apply_fixes(labels: pd.DataFrame, review: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """verdict = FIX 인 신호 clip을 검수자 경계로 다시 만듦. OK는 qa_status만 PASS로."""
    out = labels.copy()
    changed = []
    for _, r in review.iterrows():
        v = str(r.get("verdict", "")).strip().upper()
        clip = r["clip_id"]
        g = out[out["clip_id"] == clip]
        if g.empty:
            continue
        if v == "OK":
            out.loc[g.index, "qa_status"] = "PASS"
            continue
        if v == "RESHOOT":
            out.loc[g.index, "qa_status"] = "RESHOOT"
            changed.append(f"{clip}: RESHOOT")
            continue
        if v != "FIX" or r["action_id"] not in SIGNALS:
            continue
        b = [int(float(r[f"review_{k}"])) for k in BOUNDARIES]
        n = int(g["end_frame"].max()) + 1
        first, last = g.iloc[0], g.iloc[-1]
        rest_end = int(first["end_frame"]) if first["label"] == "N00" else -1
        rest_start = int(last["start_frame"]) if last["label"] == "N00" else n
        rest_end = min(rest_end, b[0] - 1)
        rest_start = max(rest_start, b[3] + 1)
        seg = {"rest_end": rest_end, "rest_start": rest_start, "c01": (b[0], b[1]), "signal": (b[2], b[3])}
        rows = pd.DataFrame(label_rows(clip, first["person_id"], r["action_id"], n, seg,
                                       fps=int(first["fps"])))
        rows["labeler"] = r["reviewer"]
        rows["qa_status"] = "PASS"
        rows["note"] = ""
        rows.loc[rows.index[0], "note"] = f"교차 검수 {r['reviewer']} 수정"
        out = pd.concat([out.drop(index=g.index), rows], ignore_index=True)
        changed.append(f"{clip}: FIX by {r['reviewer']}")
    out = out.sort_values(["clip_id", "start_frame"]).reset_index(drop=True)
    return out, changed
