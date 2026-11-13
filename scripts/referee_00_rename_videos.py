from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
import csv
import re
from pathlib import Path


# ② 프로젝트 경로와 영상 폴더 지정
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "referee" / "smoke" / "raw"
SUBJECT_CSV = ROOT / "data" / "referee" / "smoke" / "subjects.csv"


# ③ 파일명 앞의 동작 번호 → 영문 폴더명
#    (촬영 순서표 번호: 3 = U파울, 5 = 푸싱)
ACTION_CODES = {
    "s03": "ufoul",
    "s05": "pushing",
}

# 원본 파일명 형식: "s03 u파울-김하민 (1).mp4"
PATTERN = re.compile(
    r"^(?P<code>s\d+)\s+(?P<action>.+?)-(?P<name>.+?)\s*\((?P<take>\d+)\)\.mp4$",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # 기본은 미리보기만 하고, --apply를 붙여야 실제로 이름을 바꿈
    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 파일 이름 변경",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"폴더가 없습니다: {RAW_DIR}")

    # ④ raw 폴더 아래의 모든 MP4 찾기
    videos = sorted(RAW_DIR.rglob("*.mp4"))

    items = []
    skipped = []

    for path in videos:
        match = PATTERN.match(path.name)

        # 이미 바꾼 파일(S01_ufoul_T01.mp4 등)이나 형식이 다른 파일은 건너뜀
        if match is None:
            skipped.append(path)
            continue

        code = match.group("code").lower()

        if code not in ACTION_CODES:
            skipped.append(path)
            continue

        items.append(
            {
                "src": path,
                "action": ACTION_CODES[code],
                # 이름 앞뒤 공백 제거 → "최용훈 "과 "최용훈"을 같은 사람으로
                "name": match.group("name").strip(),
                "take": int(match.group("take")),
            }
        )

    if not items:
        print("바꿀 파일이 없습니다.")
        return

    # ⑤ 사람 이름 → S01, S02 ... (가나다순, 두 동작에서 같은 번호)
    names = sorted({item["name"] for item in items})
    subject_ids = {
        name: f"S{index:02d}"
        for index, name in enumerate(names, start=1)
    }

    # ⑥ 새 파일 경로 만들기: raw/<action>/S01_<action>_T01.mp4
    plans = []
    targets = set()

    for item in items:
        subject = subject_ids[item["name"]]
        new_name = f"{subject}_{item['action']}_T{item['take']:02d}.mp4"
        dst = RAW_DIR / item["action"] / new_name

        # 같은 이름이 두 번 만들어지거나 이미 있으면 중단
        if dst in targets or (dst.exists() and dst != item["src"]):
            raise RuntimeError(f"이름이 겹칩니다: {dst}")

        targets.add(dst)
        plans.append((item["src"], dst))

    # ⑦ 미리보기 출력
    print("사람 번호")
    for name, subject in subject_ids.items():
        print(f"  {subject} = {name}")
    print()

    for src, dst in plans:
        print(f"{src.relative_to(RAW_DIR)}  ->  {dst.relative_to(RAW_DIR)}")

    print()
    print(f"변경 예정: {len(plans)}개 / 건너뜀: {len(skipped)}개")

    if not args.apply:
        print()
        print("미리보기만 했습니다. 실제로 바꾸려면 --apply를 붙여 다시 실행하세요.")
        return

    # ⑧ 실제 이름 변경 (필요하면 동작 폴더로 이동)
    for src, dst in plans:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

    # ⑨ 이름 ↔ S번호 대응표 저장 (기존 표가 있으면 덮어쓰지 않음)
    if SUBJECT_CSV.exists():
        print(f"대응표가 이미 있어 덮어쓰지 않았습니다: {SUBJECT_CSV}")
    else:
        with SUBJECT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["subject_id", "name"])
            for name, subject in subject_ids.items():
                writer.writerow([subject, name])
        print(f"대응표 저장: {SUBJECT_CSV}")

    print(f"완료: {len(plans)}개 이름 변경")


if __name__ == "__main__":
    main()