"""VICON 원본 파일명 → take_log.csv 대응표 생성.

규칙 (docs/referee/01_관절_매핑_명세.md 7.1, 7.4):
- 원본 이름: 261111_E002_V2_T001_KHM.fbx  (날짜_E동작[_V변형]_T테이크_배우)
- clip_id  : P001_A002_T03                (사람_동작_테이크)
- T 번호는 사람·동작마다 01부터 다시 매김. 순서 = V 없음 → V1 → V2 → V3 → ..., 같은 V 안에서는 원본 T 순서
- V(제자리/이동 등)는 clip_id에 넣지 않고 variant 칸에 기록
- E016·E017 = 대표 라벨 N00, E018~E024(펜싱) = use N

실행:
    uv run python scripts/vicon_00_make_take_log.py --dry-run
    uv run python scripts/vicon_00_make_take_log.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "vicon"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "take_log.csv"

# 원본 파일명: 261111_E002_V2_T001_KHM.fbx (V는 없을 수 있음)
NAME_RE = re.compile(
    r"^(?P<date>\d{6})_E(?P<e>\d{3})(?:_V(?P<v>\d+))?_T(?P<t>\d{3})_(?P<actor>[A-Za-z]+)$"
)

# 배우 약칭 → person_id. 새 배우가 생기면 여기에 추가
PERSON_IDS = {
    "KHM": "P001",
}

# E001~E015 = 신호 A01~A15 (촬영 순서표 번호와 같음)
SIGNAL_ACTIONS = range(1, 16)

# 현장에서 추가한 동작: 대표 라벨 N00으로 사용
EXTRA_ACTIONS = {
    16: ("N00", "선수 가리키며 걷기"),
    17: ("N00", "공격권 방향 지시 (대상 외 FIBA 신호)"),
}

# 사용하지 않는 동작
EXCLUDED_ACTIONS = {e: "펜싱 (제외)" for e in range(18, 25)}

VARIANT_DESC = {
    "": "",
    "V1": "제자리",
    "V2": "이동",
    "V3": "V3 (의미 확인 필요)",
}

COLUMNS = [
    "source_file",
    "clip_id",
    "person_id",
    "action_id",
    "take",
    "variant",
    "variant_desc",
    "orig_take",
    "shoot_date",
    "main_label",
    "use",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VICON 파일명 → take_log.csv")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="VICON 원본 폴더")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="출력 CSV 경로")
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 결과만 출력")
    parser.add_argument("--overwrite", action="store_true", help="기존 take_log.csv 덮어쓰기")
    return parser.parse_args()


def find_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".fbx")


def parse_file(path: Path) -> dict | None:
    m = NAME_RE.match(path.stem)
    if not m:
        return None
    date = m["date"]
    return {
        "source_file": path.name,
        "actor": m["actor"].upper(),
        "e": int(m["e"]),
        "variant": f"V{int(m['v'])}" if m["v"] else "",
        "orig_take": int(m["t"]),
        "shoot_date": f"20{date[:2]}-{date[2:4]}-{date[4:6]}",
    }


def variant_order(variant: str) -> int:
    return int(variant[1:]) if variant else 0


def build_rows(items: list[dict]) -> list[dict]:
    used: dict[tuple[str, int], list[dict]] = {}
    excluded: list[dict] = []

    for it in items:
        person_id = PERSON_IDS[it["actor"]]
        e = it["e"]
        row = {
            "source_file": it["source_file"],
            "clip_id": "",
            "person_id": person_id,
            "action_id": f"A{e:03d}",
            "take": "",
            "variant": it["variant"],
            "variant_desc": VARIANT_DESC.get(it["variant"], f"{it['variant']} (의미 확인 필요)"),
            "orig_take": f"T{it['orig_take']:03d}",
            "shoot_date": it["shoot_date"],
            "main_label": "",
            "use": "Y",
            "note": "",
        }
        if e in SIGNAL_ACTIONS:
            row["main_label"] = f"A{e:02d}"
        elif e in EXTRA_ACTIONS:
            row["main_label"], row["note"] = EXTRA_ACTIONS[e]
        else:
            row["use"] = "N"
            row["action_id"] = ""
            row["note"] = EXCLUDED_ACTIONS.get(e, f"E{e:03d}: 알 수 없는 동작 번호 (확인 필요)")
            excluded.append(row)
            continue
        used.setdefault((person_id, e), []).append({**row, "_v": variant_order(it["variant"]), "_t": it["orig_take"]})

    rows: list[dict] = []
    for (person_id, e) in sorted(used):
        group = sorted(used[(person_id, e)], key=lambda r: (r["_v"], r["_t"], r["source_file"]))
        for take, r in enumerate(group, start=1):
            r["take"] = take
            r["clip_id"] = f"{person_id}_A{e:03d}_T{take:02d}"
            rows.append({k: r[k] for k in COLUMNS})

    rows.extend(sorted(excluded, key=lambda r: r["source_file"]))
    return rows


def check_problems(items: list[dict]) -> list[str]:
    problems = []
    unknown = sorted({it["actor"] for it in items if it["actor"] not in PERSON_IDS})
    if unknown:
        problems.append(f"PERSON_IDS에 없는 배우: {', '.join(unknown)} → 스크립트 PERSON_IDS에 추가")
    keys = Counter((it["actor"], it["e"], it["variant"], it["orig_take"]) for it in items)
    for (actor, e, v, t), n in keys.items():
        if n > 1:
            problems.append(f"같은 테이크가 {n}개: {actor} E{e:03d} {v or '-'} T{t:03d}")
    return problems


def print_summary(rows: list[dict], skipped: list[str]) -> None:
    used = [r for r in rows if r["use"] == "Y"]
    print(f"\n[대응 결과] 사용 {len(used)}개 / 제외 {len(rows) - len(used)}개 / 이름 규칙 밖 {len(skipped)}개")
    print(f"{'source_file':<34} {'clip_id':<16} {'variant':<8} {'label':<6} note")
    for r in rows:
        print(f"{r['source_file']:<34} {r['clip_id'] or '-':<16} {r['variant'] or '-':<8} {r['main_label'] or '-':<6} {r['note']}")
    if skipped:
        print("\n[이름 규칙 밖 → take_log에 넣지 않음]")
        for name in skipped:
            print(f"  {name}")


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        print(f"[오류] 폴더가 없음: {args.input}")
        return 1

    items, skipped = [], []
    for path in find_files(args.input):
        info = parse_file(path)
        if info is None:
            skipped.append(path.name)
        else:
            items.append(info)

    if not items:
        print(f"[오류] 이름 규칙에 맞는 FBX가 없음: {args.input}")
        return 1

    problems = check_problems(items)
    if problems:
        print("[오류] 대응표를 만들 수 없음")
        for p in problems:
            print(f"  - {p}")
        return 1

    rows = build_rows(items)
    print_summary(rows, skipped)

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았음")
        return 0
    if args.output.exists() and not args.overwrite:
        print(f"\n[중단] 이미 있음: {args.output} (덮어쓰려면 --overwrite)")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[저장] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())