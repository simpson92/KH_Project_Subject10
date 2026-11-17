from __future__ import annotations

# ==================================================
# day04_make_clip_manifest.py
# 스모크 영상 → 4일차 이름 규칙(T01_S##_A##_T##) + clip manifest
# ==================================================
# 4일차 교재 STEP 1 "영상마다 고유한 이름과 정답 Label을 붙입니다"를
# 1조 스모크 영상 100개(5명 × U파울·푸싱 × 10회)에 적용합니다.
#
# 하는 일
#   1. data/referee/smoke/raw 의 영상 이름을 읽음
#      (원본 "s03 u파울-이름 (1).mp4" · 변환 후 "S01_ufoul_T01.mp4" 둘 다 인식)
#   2. 사람 → S##   : data/referee/smoke/subjects.csv (없으면 가나다순으로 새로 만듦)
#      동작 → A##   : configs/team01_action_classes.csv 의 smoke_action 칸
#                     (ufoul → A03, pushing → A05 — 02_라벨링_가이드 4장)
#      회차 → T##
#   3. clip_id = T01_S##_A##_T## (01_관절_매핑_명세 7.1)
#   4. configs/team01_clip_manifest.csv 작성 (실명 없음 → Git에 올려도 됨)
#      review_status = 3일차 video_qa.csv 결과(PASS/REVIEW), 없으면 PENDING
#   5. --apply 를 붙이면 영상을 clip_id 이름으로 복사
#      → data/team_projects/team01/rgb/videos/T01_S01_A03_T01.mp4 (4일차 교재 폴더 구조)
#      원본은 그대로 둠 (이동·삭제 안 함)
#
# 실행
#   python scripts/day04_make_clip_manifest.py            # 미리보기 + manifest만
#   python scripts/day04_make_clip_manifest.py --apply    # 영상 복사까지
#   python scripts/day04_make_clip_manifest.py --apply --link  # 하드링크(용량 절약)
#
# 실명이 들어간 원본 경로 ↔ clip_id 대응표는 Git 제외 경로에만 저장합니다.
#   data/referee/smoke/clip_id_map.csv

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.referee_utils import RAW_DIR, SMOKE_DATA_DIR, VIDEO_EXTENSIONS  # noqa: E402

TEAM = "T01"
ACTION_CLASSES_CSV = ROOT / "configs" / "team01_action_classes.csv"
MANIFEST_CSV = ROOT / "configs" / "team01_clip_manifest.csv"
DEST_DIR = ROOT / "data" / "team_projects" / "team01" / "rgb" / "videos"
CLIP_MAP_CSV = SMOKE_DATA_DIR / "clip_id_map.csv"
VIDEO_QA_CSV = ROOT / "reports" / "day03" / "tables" / "video_qa.csv"

MANIFEST_COLUMNS = [
    # 4일차 교재 manifest 기본 칸
    "clip_id",
    "subject_id",
    "action_id",
    "action_label",
    "take_id",
    "video_file",
    "source_type",
    "review_status",
    # 1조 추가 칸
    "capture_set",
    "review_reason",
    "fps",
    "frame_count",
    "duration_sec",
    "width",
    "height",
]


def load_script(filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==================================================
# [01] 실행 옵션
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(RAW_DIR))
    parser.add_argument("--dest-dir", default=str(DEST_DIR))
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--video-qa", default=str(VIDEO_QA_CSV))
    parser.add_argument(
        "--source-type", default="phone", help="촬영 장비 (phone · webcam)"
    )
    parser.add_argument("--capture-set", default="smoke", help="촬영 묶음 이름")
    parser.add_argument(
        "--apply", action="store_true", help="영상을 clip_id 이름으로 복사"
    )
    parser.add_argument("--link", action="store_true", help="복사 대신 하드링크")
    return parser.parse_args()


# ==================================================
# [02] 동작 정의표 → {smoke_action: (A##, 국문명)}
# ==================================================
def load_action_map(path: Path = ACTION_CLASSES_CSV) -> dict[str, tuple[str, str]]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        row["smoke_action"]: (row["action_id"], row["label_ko"])
        for _, row in table.iterrows()
        if row["smoke_action"]
    }


# ==================================================
# [03] 사람 번호표 (subjects.csv) — 한 번 정하면 고정
# ==================================================
# 가나다순 번호는 새 사람이 들어오면 밀릴 수 있으므로,
# 처음 만든 번호표를 저장해 두고 다음부터는 그 표를 씁니다.
def ensure_subjects_csv(videos: list[Path], rename, path: Path) -> bool:
    """번호표가 없으면 referee_00과 같은 규칙(가나다순)으로 만들고 True 반환."""
    if path.exists():
        return False
    names = set()
    for video in videos:
        match = rename.PATTERN.match(video.name)
        if match and match.group("code").lower() in rename.ACTION_CODES:
            names.add(match.group("name").strip())
    if not names:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"subject_id": f"S{i:02d}", "name": n} for i, n in enumerate(sorted(names), 1)
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return True


# ==================================================
# [04] 영상 정보 (FPS · Frame 수 · 해상도)
# ==================================================
def read_video_info(path: Path) -> dict:
    try:
        import cv2
    except ImportError:
        return {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    info = {
        "fps": round(fps, 2),
        "frame_count": frames,
        "duration_sec": round(frames / fps, 2) if fps else "",
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


# ==================================================
# [05] manifest 행 만들기
# ==================================================
def make_clip_id(subject: str, action_id: str, take: int) -> str:
    return f"{TEAM}_{subject}_{action_id}_T{int(take):02d}"


def build_manifest(videos, ids, action_map, qa_status, args, read_info=read_video_info):
    """반환: (manifest DataFrame, 원본 경로 → clip_id 대응 DataFrame, 경고 목록)"""
    rows, mapping, warnings = [], [], []
    for video in videos:
        info = ids[video]
        action = info.get("action")
        if action not in action_map:
            warnings.append(f"동작을 알 수 없어 제외: {info['video_id']}")
            continue
        action_id, label = action_map[action]
        clip_id = make_clip_id(info["subject"], action_id, info["take"])

        # 3일차 일괄 QA 결과 (video_qa.csv의 video = S01_ufoul_T01)
        status, reason = qa_status.get(info["video_id"], ("PENDING", ""))

        row = {
            "clip_id": clip_id,
            "subject_id": info["subject"],
            "action_id": action_id,
            "action_label": label,
            "take_id": f"T{int(info['take']):02d}",
            "video_file": f"{clip_id}.mp4",
            "source_type": args.source_type,
            "review_status": status,
            "capture_set": args.capture_set,
            "review_reason": reason,
        }
        row.update(read_info(video))
        rows.append(row)
        mapping.append({"source_path": str(video), "clip_id": clip_id})

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    duplicated = manifest["clip_id"][manifest["clip_id"].duplicated()]
    if not duplicated.empty:
        raise RuntimeError(f"clip_id가 겹칩니다: {sorted(set(duplicated))}")
    manifest = manifest.sort_values("clip_id").reset_index(drop=True)
    for column in ("frame_count", "width", "height"):  # 154.0 → 154
        manifest[column] = pd.to_numeric(manifest[column], errors="coerce").astype(
            "Int64"
        )
    return manifest, pd.DataFrame(mapping), warnings


def load_qa_status(path: Path) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    qa = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {r["video"]: (r["status"], r.get("reasons", "")) for _, r in qa.iterrows()}


# ==================================================
# [06] 같은 사람 = 같은 S번호 점검 (Subject-wise Split 누수 방지)
# ==================================================
# 01_관절_매핑_명세 T-01: VICON 배우 KHM = S01 (스모크 S01과 같은 사람, 2026-11-16 확정).
# 같은 사람이 스모크·웹캠에도 찍혔으면 VICON 쪽 번호를 맞춰야 함 (후속 작업 11).
VICON_ACTORS = {"KHM": "S01"}


def check_vicon_subjects(subjects_csv: Path) -> list[str]:
    if not subjects_csv.exists():
        return []
    table = pd.read_csv(subjects_csv, encoding="utf-8-sig", dtype=str)
    notes = []
    for actor, vicon_id in VICON_ACTORS.items():
        # 이름 대신 번호만 출력 (실명 노출 방지). 대응 여부는 사람이 확인
        notes.append(
            f"VICON 배우 {actor} = {vicon_id}. 이 배우가 스모크 촬영자"
            f"({', '.join(table['subject_id'])}) 중 한 명이면"
            " VICON 번호를 그 번호로 맞춰야 함"
            " (01 문서 T-01 · 후속 작업 11)"
        )
    return notes


# ==================================================
# [07] 영상 복사 (원본은 그대로)
# ==================================================
def copy_videos(mapping: pd.DataFrame, dest_dir: Path, link: bool) -> tuple[int, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for _, m in mapping.iterrows():
        src, dst = Path(m["source_path"]), dest_dir / f"{m['clip_id']}.mp4"
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            raise RuntimeError(f"다른 파일이 이미 있습니다: {dst}")
        if link:
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
        done += 1
    return done, skipped


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    videos = sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit(f"영상이 없습니다: {source_dir}")

    rename = load_script("referee_00_rename_videos.py")
    batch = load_script("day03_batch_video_qa.py")

    if ensure_subjects_csv(videos, rename, Path(rename.SUBJECT_CSV)):
        print(f"사람 번호표를 새로 만들었습니다 (Git 제외): {rename.SUBJECT_CSV}")

    ids = batch.identify_videos(videos, rename)
    manifest, mapping, warnings = build_manifest(
        videos, ids, load_action_map(), load_qa_status(Path(args.video_qa)), args
    )

    # 요약 출력 (실명 없이)
    print()
    print("사람 × 동작별 clip 수")
    print(pd.crosstab(manifest["subject_id"], manifest["action_id"]).to_string())
    print()
    print("review_status:", manifest["review_status"].value_counts().to_dict())
    for _, r in manifest[manifest["review_status"] == "REVIEW"].iterrows():
        print(f"  REVIEW {r['clip_id']}  {r['review_reason']}")
    for w in warnings:
        print("  [경고]", w)
    for note in check_vicon_subjects(Path(rename.SUBJECT_CSV)):
        print("  [확인]", note)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    CLIP_MAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(CLIP_MAP_CSV, index=False, encoding="utf-8-sig")
    print()
    print(f"Saved: {manifest_path}  ({len(manifest)} clips, 실명 없음)")
    print(f"Saved: {CLIP_MAP_CSV}  (원본 경로 대응표, Git 제외)")

    if not args.apply:
        print()
        print("미리보기만 했습니다. 영상을 clip_id 이름으로 복사하려면 --apply")
        return

    done, skipped = copy_videos(mapping, Path(args.dest_dir), args.link)
    how = "하드링크" if args.link else "복사"
    print(f"{how} {done}개, 이미 있어 건너뜀 {skipped}개 → {args.dest_dir}")


if __name__ == "__main__":
    main()
