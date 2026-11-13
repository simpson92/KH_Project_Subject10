from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import re
from pathlib import Path


# ② 조 프로젝트(농구 심판 수신호) 스모크 테스트 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SMOKE_DATA_DIR = PROJECT_ROOT / "data" / "referee" / "smoke"
RAW_DIR = SMOKE_DATA_DIR / "raw"

SMOKE_REPORT_DIR = PROJECT_ROOT / "reports" / "referee" / "smoke"
KEYPOINT_DIR = SMOKE_REPORT_DIR / "keypoints"

MANIFEST_CSV = SMOKE_REPORT_DIR / "video_manifest.csv"
POSE_QUALITY_CSV = SMOKE_REPORT_DIR / "pose_quality.csv"


# ③ 영상 확장자 (폰 영상은 .MP4처럼 대문자일 수 있음)
VIDEO_EXTENSIONS = {".mp4", ".mov"}

# 파일명 형식: S01_ufoul_T01.mp4
NAME_PATTERN = re.compile(
    r"^(?P<subject>S\d+)_(?P<action>[a-z0-9]+)_T(?P<take>\d+)$"
)


# ④ raw 폴더 아래의 영상 목록
def list_videos() -> list[Path]:
    """data/referee/smoke/raw 아래의 모든 영상을 이름순으로 반환합니다."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"폴더가 없습니다: {RAW_DIR}")

    return sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


# ⑤ 파일명에서 사람·동작·take 읽기
def parse_name(stem: str) -> dict | None:
    """S01_ufoul_T01 → {subject: S01, action: ufoul, take: 1}"""
    match = NAME_PATTERN.match(stem)

    if match is None:
        return None

    return {
        "subject": match.group("subject"),
        "action": match.group("action"),
        "take": int(match.group("take")),
    }