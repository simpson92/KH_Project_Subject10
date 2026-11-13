from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import platform
import sys
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    ensure_day01_dirs,
    load_config,
    resolve_device,
)


def main() -> None:

    # ③ 필요한 폴더 생성 및 프로젝트 설정 읽기
    ensure_day01_dirs()

    config = load_config()
    device = resolve_device(config["device"])

    # ④ Python·라이브러리·CUDA 실행환경 정보 수집
    lines = [
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"OpenCV: {cv2.__version__}",
        f"PyTorch: {torch.__version__}",
        f"Ultralytics: {ultralytics.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"PyTorch CUDA: {torch.version.cuda}",
    ]

    # GPU 사용 가능 여부와 GPU 이름 확인
    if torch.cuda.is_available():
        lines.append(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        lines.append("GPU: CPU mode")

    # 실제 선택된 실행 장치 기록
    lines.append(f"Selected device: {device}")

    # ⑤ 수집한 환경정보 출력
    print("\n".join(lines))
    print("\nPose model loading...")

    # ⑥ YOLO Pose 모델 로딩 확인
    YOLO(config["pose_model"])

    print("Pose model load: OK")

    # ⑦ 환경 점검 결과를 TXT 파일로 저장
    report_path = (
        ROOT
        / "reports"
        / "day01"
        / "environment.txt"
    )

    # 결과 저장 폴더가 없으면 생성
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 수집한 환경정보를 파일에 기록
    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"\nSaved: {report_path}")


# 프로그램 실행 시작
if __name__ == "__main__":
    main()