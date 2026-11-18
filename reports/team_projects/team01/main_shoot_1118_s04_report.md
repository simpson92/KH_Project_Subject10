# 본 촬영 3차 결과 — 2026-11-18 (S04, 90개)

- 작성: 용훈
- 앞 보고서: `main_shoot_1117_report.md` (1차 73개) · `main_shoot_1117_2_report.md` (2차 108개)
- manifest: `configs/team01_main_manifest.csv` (누적 **271행**, 이번 `capture_set = main_1118`)
- 영상 위치 (Git 제외): `data/team_projects/team01/main/rgb/videos/`

## 1. 무엇을 찍었나

| 항목 | 값 |
| --- | --- |
| 사람 | S04 (Train) |
| 반복 | 18동작 × T01~T05 = 90개. 체크표 순서대로 동작마다 5개 연속 촬영 (일상 동작은 D01 → D03 → D02 순) |
| 전송 | **원본 그대로 전송** (카카오톡 아님) → 촬영 시각이 파일 이름에 남아 있음 |
| 원본 형식 | **4K 3840×2160 HEVC + 세로 회전 정보**, 가변 FPS (기준 W7과 다름) |

## 2. 기준 형식으로 변환 (원본 보존)

W7(세로 1080×1920 · 30 FPS)에 맞추려고 원본은 두고 변환본을 만들었습니다. 1 · 2차 영상과 같은 해상도 · 고정 FPS가 되어야 Feature 비교가 공정합니다.

```bash
# 회전 정보는 ffmpeg가 자동 적용 → 세로 결과
ffmpeg -i <원본>.mp4 -vf "scale=1080:1920,fps=30" -c:v libx264 -preset veryfast -crf 18 \
       -pix_fmt yuv420p -an -movflags +faststart <clip_id>.mp4
```

- 다른 폰으로 찍어도 4K · 가변 FPS면 같은 명령으로 변환한 뒤 manifest에 넣습니다.
- 이름 붙이기: 촬영 시각 순서 = 체크표 순서이므로 5개씩 묶어 T01~T05. 90개 모두 대표 Frame 9장으로 동작이 맞는지 확인 (**판정 수정 0건**)
- 원본 파일 ↔ clip_id 대응표는 Git 제외 (`main_shoot_1118_s04_map.csv`)

## 3. 품질 검사 결과

| 검사 | 결과 |
| --- | --- |
| 해상도 · FPS (W7) | 변환 후 90개 모두 1080×1920 · 30 FPS · 3.9~6.7초 · 합계 13,690 Frame |
| 사람 수 (Q-01) | 13,690 Frame 모두 pose_ok · 여러 사람 Frame 0 ✅ |
| 관절 conf | 90개 모두 PASS (손목 · 발목 conf < 0.50 Frame 0) ✅ |
| 판정 | **PASS 90** |

### 3.1 YOLO Pose 관절 confidence (`scripts/day04_pose_conf_report.py`)

| 사람 | clip | 전체 평균 | 팔(5~10번) 평균 | 손목 최솟값 | 발목 최솟값 |
| --- | --- | --- | --- | --- | --- |
| S04 | 90 | 0.971 | 0.991 | 0.853 | 0.758 |
| (참고) S01 | 54 | 0.967 | 0.989 | 0.878 | 0.930 |
| (참고) S05 | 54 | 0.967 | 0.986 | 0.856 | 0.852 |

- 1 · 2차와 같은 수준. 가장 낮은 관절은 이번에도 **오른귀(평균 0.75, conf < 0.50 Frame 4.7%)**
- 발목 최솟값 0.76은 A10 킥처럼 다리가 움직이는 순간. conf < 0.50 Frame은 0

### 3.2 손 검출 (crop · 원본 크기, 손목 5,542개)

| | 왼손 | 오른손 |
| --- | --- | --- |
| 전체 | 95.9% | 92.1% |

| 동작 | 검출률 | 비고 |
| --- | --- | --- |
| A03 U파울 | 86.6% | 머리 위에서 손이 겹침 (1 · 2차와 같음) |
| A15 교체 | 90.6% | |
| **D02 팔짱** | **61.2%** | 의도대로 손이 안 보임 → A15와 구분 근거 유지 |
| D03 뒷머리 | 82.5% | |
| 그 밖의 신호 · D01 | 95~100% | |

- 어깨 폭 중앙값 244 px (기준 60 px의 4배)

## 4. 진행 상황

| 사람 | 역할 (제안) | 개수 | 상태 |
| --- | --- | --- | --- |
| S01 | Train | 90 | ✅ |
| S04 | Train | 90 | ✅ 이번 |
| S05 | Validation | 91 | ✅ (A03만 6개) |
| S03 | Train | 0 | ⬜ 90개 남음 |
| S02 | Test | 0 | ⬜ 90개 남음 |

누적 **271개** (목표 450 중 270개 = 60%).

## 5. 다시 돌리기

```bash
M=configs/team01_main_manifest.csv
V=data/team_projects/team01/main/rgb/videos
O=reports/team_projects/team01/main
uv run python scripts/day04_check_team_videos.py --manifest $M --video-dir $V --output $O/tables/video_inventory.csv
uv run python scripts/day04_build_team_sequences.py --manifest $M --video-dir $V --output-dir $O/sequences   # 이미 있는 clip은 건너뜀
uv run --with mediapipe python scripts/day04_hand_detect_check.py --manifest $M --video-dir $V \
  --sequence-dir $O/sequences --output-dir $O/tables --scales 1.0 --modes crop
uv run python scripts/day04_pose_conf_report.py
```
