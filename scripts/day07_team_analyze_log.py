from __future__ import annotations

# ==================================================
# day07_team_analyze_log.py — 3단계 E. IP Webcam 로그 분석 (수업 45번 → 1조 버전)
# ==================================================
# 왜?  실시간 화면은 순식간에 지나가서 눈으로 본 인상만 남습니다.
#      저장된 Runtime 로그에서 "Target이 있는 구간 × Prediction이 갱신된 Frame"만 모아
#      동작별로 Stable 결과 · UNKNOWN 비율 · 흔들림 · FPS를 숫자로 남깁니다.
#      (Pose · RF를 다시 돌리지 않음 — 저장된 CSV만 읽음)
#
# 수업 45번과 다른 점
#   - 동작 기준: expected_label 대신 target_action (D01 · D02 · D03은 정답이 모두 N00이라 따로 봐야 함)
#   - 여러 로그를 한 번에: --all (Target 6개씩 3번 나눠 실행한 로그를 합침) 또는 --log 여러 개
#   - PM 기록표 형식: "수행 N회 중 마지막 Stable = 정답 _회" (segment = Target 키를 누른 한 번)
#   - 헷갈리는 쪽: 틀린 Stable 중 가장 많은 Label + 설계 때 예상한 혼동 쌍인지 (confusable 열)
#     6일차 Val에서 헷갈린 A04 → A06 · A07 → A05 · D03 → A11을 따로 출력
#   - protocol(with_c01 / no_c01)별로 나눠서 비교
#
# stable_match_ratio · final_correct는 Accuracy가 아님 — 사람이 누른 Target 기준 참고 지표.
#
# 입력: reports/team_projects/team01/main/day07/ip_webcam/ip_webcam_*.csv
# 출력: reports/team_projects/team01/main/day07/ip_webcam_summary.csv  (요약표 = 커밋)
#
# 실행
#   uv run python scripts/day07_team_analyze_log.py            # 가장 최근 분석 가능한 로그 1개
#   uv run python scripts/day07_team_analyze_log.py --all      # 폴더의 로그 전부 (16동작 + 일상 + X)
#   uv run python scripts/day07_team_analyze_log.py --log <파일1> <파일2>
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_realtime_utils import (
    DAY07_DIR,
    IP_WEBCAM_DIR,
    UNTRAINED_TARGET,
    label_changes,
    load_confusable,
)

REQUIRED_COLUMNS = {"target_action", "expected_label", "segment_id", "prediction_updated",
                    "threshold_label", "stable_label", "raw_label", "confidence", "processing_fps"}
NOT_A_LABEL = {"WARMING_UP", "UNKNOWN", ""}
WATCH_PAIRS = {"A04": "A06", "A07": "A05", "D03": "A11"}   # 6일차 Val에서 헷갈린 쌍


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    pick = parser.add_mutually_exclusive_group()
    pick.add_argument("--log", nargs="+", help="분석할 로그 (여러 개 가능)")
    pick.add_argument("--all", action="store_true", help="입력 폴더의 로그 전부")
    parser.add_argument("--input-dir", default=str(IP_WEBCAM_DIR))
    parser.add_argument("--output", default=str(DAY07_DIR / "ip_webcam_summary.csv"))
    return parser.parse_args(argv)


def resolve_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_runtime_log(path: Path) -> tuple[pd.DataFrame, dict]:
    """로그 → (Target 있음 & Prediction 갱신 Frame, 진단 숫자)."""
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("필수 열 없음: " + ", ".join(missing))
    df["target_action"] = df["target_action"].astype(str).str.strip()
    updated = pd.to_numeric(df["prediction_updated"], errors="coerce").fillna(0).astype(int) == 1
    has_target = df["target_action"] != ""
    usable = df[updated & has_target].copy()
    usable["log"] = path.name
    for column in ("confidence", "processing_fps"):
        usable[column] = pd.to_numeric(usable[column], errors="coerce")
    diagnostic = {"rows": len(df), "target_rows": int(has_target.sum()),
                  "prediction_rows": int(updated.sum()), "usable_rows": len(usable)}
    return usable, diagnostic


def log_candidates(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.glob("ip_webcam_*.csv")
             if "summary" not in path.name.lower() and "comparison" not in path.name.lower()]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def print_stop_help() -> None:
    print("day07_team_ip_webcam.py 실행에서 확인하세요.")
    print("1) 카메라 창을 클릭해야 키 입력이 들어감")
    print("2) 1~6 키로 Target 선택 → 화면 Target이 '-'가 아닌지")
    print("3) WARMING_UP이 끝날 때까지(약 4초 이상) 동작 수행")
    print("4) Prediction이 나온 뒤 Q로 종료 → 새 CSV 저장")


def collect(args) -> pd.DataFrame:
    if args.log:
        paths = [resolve_path(value) for value in args.log]
    else:
        input_dir = resolve_path(args.input_dir)
        paths = log_candidates(input_dir)
        if not paths:
            print(f"[STOP] IP Webcam 로그가 없음: {input_dir}")
            raise SystemExit(1)
        # --all이 아니면 최근 것부터 보고, 분석 가능한 첫 로그 1개만 씀 (수업 45번과 같음)
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            usable, diagnostic = read_runtime_log(path)
        except ValueError as exc:
            print(f"  SKIP {path.name}: {exc}")
            continue
        status = "OK  " if len(usable) else "SKIP"
        print(f"  {status} {path.name} | rows={diagnostic['rows']} target_rows={diagnostic['target_rows']}"
              f" prediction_rows={diagnostic['prediction_rows']} usable_rows={diagnostic['usable_rows']}")
        if len(usable):
            frames.append(usable)
            if not args.all and not args.log:
                break
    if not frames:
        print()
        print("[STOP] Target이 표시된 Prediction이 한 번도 없음")
        print_stop_help()
        raise SystemExit(1)
    return pd.concat(frames, ignore_index=True)


def summarize(usable: pd.DataFrame, confusable: dict[str, set[str]]) -> pd.DataFrame:
    """target_action × protocol 마다 1줄."""
    if "protocol" not in usable.columns:
        usable["protocol"] = ""
    rows = []
    for (target_action, protocol), group in usable.groupby(["target_action", "protocol"], sort=True):
        expected = str(group["expected_label"].iloc[0])
        has_answer = target_action != UNTRAINED_TARGET
        segments = list(group.groupby(["log", "segment_id"], sort=False))
        finals = [str(segment["stable_label"].iloc[-1]) for _, segment in segments]

        stable_labels = group["stable_label"].astype(str)
        decided = stable_labels[~stable_labels.isin(NOT_A_LABEL)]
        top_label = decided.value_counts().index[0] if len(decided) else ""
        top_share = round(float((decided == top_label).mean()), 4) if len(decided) else None
        wrong = decided[decided != expected] if has_answer else decided
        top_wrong = wrong.value_counts().index[0] if len(wrong) else ""
        raw_labels = group["raw_label"].astype(str)                  # Confidence · 다수결 전 모델 1등
        top_raw = raw_labels.value_counts().index[0] if len(raw_labels) else ""
        fps = group["processing_fps"][group["processing_fps"] > 0]

        rows.append({
            "target_action": target_action,
            "expected_label": expected if has_answer else "",
            "protocol": protocol,
            "segments": len(segments),
            "final_correct": sum(final == expected for final in finals) if has_answer else None,
            "final_stable": " ".join(finals),
            "prediction_updates": len(group),
            "unknown_ratio": round(float((group["threshold_label"] == "UNKNOWN").mean()), 4),
            "stable_match_ratio": round(float((stable_labels == expected).mean()), 4) if has_answer else None,
            "top_stable": top_label,
            "top_stable_share": top_share,
            "top_wrong_stable": top_wrong,
            "top_wrong_known_confusable": int(bool(top_wrong) and top_wrong in confusable.get(target_action, set())),
            "raw_match_ratio": round(float((raw_labels == expected).mean()), 4) if has_answer else None,
            "top_raw": top_raw,
            "top_raw_share": round(float((raw_labels == top_raw).mean()), 4) if len(raw_labels) else None,
            "raw_label_changes": sum(label_changes(segment["raw_label"]) for _, segment in segments),
            "stable_label_changes": sum(label_changes(segment["stable_label"]) for _, segment in segments),
            "mean_confidence": round(float(group["confidence"].mean()), 4),
            "mean_processing_fps": round(float(fps.mean()), 2) if len(fps) else 0.0,
            "window_seconds": group["window_seconds"].iloc[0] if "window_seconds" in group else "",
            "confidence_threshold": group["confidence_threshold"].iloc[0] if "confidence_threshold" in group else "",
            "stabilization_buffer": group["stabilization_buffer"].iloc[0] if "stabilization_buffer" in group else "",
            "logs": " ".join(sorted(group["log"].unique())),
        })
    return pd.DataFrame(rows)


def watch_pairs(usable: pd.DataFrame) -> list[str]:
    """6일차 Val에서 헷갈린 쌍이 실시간에서도 나오는지 (Raw = 모델 1등 기준, Stable은 UNKNOWN이 많아서)."""
    lines = []
    for target_action, confused in WATCH_PAIRS.items():
        group = usable[usable["target_action"] == target_action]
        if group.empty:
            continue
        raw = group["raw_label"].astype(str)
        stable = group["stable_label"].astype(str)
        decided = stable[~stable.isin(NOT_A_LABEL)]
        stable_text = (f"Stable {int((decided == confused).sum())} / {len(decided)}" if len(decided)
                       else "Stable 판정 없음(UNKNOWN)")
        lines.append(f"  {target_action} → {confused}: Raw {int((raw == confused).sum())} / {len(raw)}"
                     f" ({float((raw == confused).mean()):.0%}) · {stable_text}")
    return lines


def main(argv=None) -> int:
    args = parse_args(argv)
    print("[로그]")
    usable = collect(args)
    table = summarize(usable, load_confusable())

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")

    print()
    print("[동작별 기록표 — PM 보고 형식]")
    for _, row in table.iterrows():
        if row["target_action"] == UNTRAINED_TARGET:
            result = f"정답 없음 · Stable 대부분 {row['top_stable'] or '-'}"
        else:
            result = f"{row['segments']}회 중 {row['final_correct']}회"
        wrong = ""
        if row["top_wrong_stable"]:
            wrong = f" · 틀리면 → {row['top_wrong_stable']}{' (예상 혼동)' if row['top_wrong_known_confusable'] else ''}"
        print(f"  {row['target_action']:4s} [{row['protocol']}] {result} · UNKNOWN {row['unknown_ratio']:.0%}"
              f" · conf {row['mean_confidence']:.2f} · Raw 대부분 {row['top_raw']} ({row['top_raw_share']:.0%})"
              f" · Stable 변화 {row['stable_label_changes']}{wrong}")

    pairs = watch_pairs(usable)
    if pairs:
        print()
        print("[6일차 Val 혼동 쌍 — 실시간에서도?]")
        print("\n".join(pairs))

    print()
    print(f"평균 Processing FPS: {table['mean_processing_fps'].mean():.1f}")
    protocols = sorted(set(table["protocol"]) - {""})
    if len(protocols) > 1:
        print("C01 포함 / 없이 비교: 같은 동작의 with_c01 · no_c01 줄을 나란히 보세요 (C10 근거)")
    print("주의: final_correct · stable_match_ratio는 수동 Target 기준 참고값이며 Accuracy가 아님")
    print(f"Saved: {output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
