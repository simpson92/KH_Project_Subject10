# 본 촬영 4차 결과 — 2026-11-18 (S02 90개 · S03 80개)

- 작성: 용훈
- 앞 보고서: `main_shoot_1117_report.md` · `main_shoot_1117_2_report.md` · `main_shoot_1118_s04_report.md`
- manifest: `configs/team01_main_manifest.csv` (누적 **441행**, 이번 `capture_set = main_1118`)
- 영상 위치 (Git 제외): `data/team_projects/team01/main/rgb/videos/`

## 1. 무엇을 찍었나

| 항목 | S02 (Test) | S03 (Train) |
| --- | --- | --- |
| 개수 | 90 (18동작 × T01~T05) | **80** (16동작 × T01~T05) |
| 빠진 것 | 없음 | **A12 후킹 5개 · A13 백코트 5개** (A11 다음에 바로 A14를 찍음) |
| 촬영 순서 | 체크표 순서, 동작마다 5개 연속 (일상 동작은 D01 → D03 → D02) | 같음 |
| 전송 | 원본 그대로 | 원본 그대로 |
| 원본 형식 | 1920×1080 HEVC + 세로 회전 정보, 가변 FPS | 같음 |

## 2. 기준 형식으로 변환 · 이름 붙이기

- S04와 같은 명령으로 1080×1920 · 30 FPS H.264로 변환 (원본 보존, `main_shoot_1118_s04_report.md` 2장)
- 촬영 시각 순서 = 체크표 순서 → 5개씩 묶어 T01~T05. 170개 모두 대표 Frame 9장으로 동작 확인 (**판정 수정 0건**)
- 원본 ↔ clip_id 대응표는 Git 제외 (`main_shoot_1118_s02_map.csv`, `main_shoot_1118_s03_map.csv`)

### ⚠ HDR 원본

- S02 · S03 원본은 **HDR (HLG · BT.2020 · 10비트)**. 변환 때 HDR → SDR 톤 매핑을 하지 않아 변환본이 **밝고 색이 옅게** 보임 (S01 · S04 · S05는 SDR)
- 영향: 관절 conf · 손 검출이 SDR 사람들과 같은 수준(3장) → Pose Feature(좌표 · 각도)에는 영향 없다고 판단해 **지금은 그대로 사용**
- 6일차에 S02(Test) · S03 결과만 유난히 나쁘면 이것부터 의심. 다시 변환할 때는 톤 매핑을 넣음:
  `-vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=1080:1920,fps=30"`
- **S03 보충 촬영과 7일차 실시간은 폰 HDR을 끄고** 촬영 (실시간 웹캠은 SDR)

## 3. 품질 검사 결과

| 검사 | S02 | S03 |
| --- | --- | --- |
| 해상도 · FPS (W7) | 90개 1080×1920 · 30 FPS · 4.1~8.3초 · 16,705 Frame | 80개 · 4.7~6.8초 · 12,796 Frame |
| 사람 수 (Q-01) | 모든 Frame pose_ok · 여러 사람 0 ✅ | 같음 ✅ |
| 관절 conf | 90개 PASS ✅ | 80개 PASS ✅ |
| 판정 | **PASS 90** | **PASS 80** |

### 3.1 YOLO Pose 관절 confidence (`scripts/day04_pose_conf_report.py`)

| 사람 | clip | 전체 평균 | 팔(5~10번) 평균 | 손목 최솟값 | 발목 최솟값 | 오른귀 평균 |
| --- | --- | --- | --- | --- | --- | --- |
| S02 | 90 | 0.968 | 0.989 | 0.877 | 0.860 | 0.705 |
| S03 | 80 | 0.969 | 0.990 | 0.858 | 0.791 | 0.728 |
| (참고) S04 | 90 | 0.971 | 0.991 | 0.853 | 0.758 | 0.748 |

- 다섯 사람 모두 팔 평균 0.99 수준 → 사람 · 폰이 달라도 Pose 품질이 고르게 나옴
- 손목 · 발목 conf < 0.50 Frame 0. 약한 관절은 이번에도 오른귀뿐 (S02 11% Frame이 0.50 미만)

### 3.2 손 검출 (crop · 원본 크기)

| | S02 (손목 6,754개) | S03 (손목 5,184개) |
| --- | --- | --- |
| 왼손 / 오른손 | 94.2% / 89.0% | 94.4% / 89.9% |
| A03 U파울 | 85.4% | 85.1% |
| A15 교체 | 84.9% | 90.6% |
| **D02 팔짱** | **48.1%** | **55.0%** |
| D03 뒷머리 | 82.7% | 87.9% |
| 어깨 폭 중앙값 | 243 px | 259 px |

- 1~3차와 같은 경향: A15(85~91%) ↔ D02 팔짱(48~55%)이 손 "찾음/못 찾음"으로 갈림

## 4. 진행 상황

| 사람 | 역할 (제안) | 개수 | 상태 |
| --- | --- | --- | --- |
| S01 | Train | 90 | ✅ |
| S02 | Test | 90 | ✅ 이번 |
| S03 | Train | 80 | 🟡 **A12 · A13 각 5개 보충 촬영 필요** |
| S04 | Train | 90 | ✅ |
| S05 | Validation | 91 | ✅ (A03만 6개) |

누적 **441개** (목표 450 중 440개 = 98%). 남은 것: `T01_S03_A12_T01~T05`, `T01_S03_A13_T01~T05`

### 보충 촬영 메모 (S03)
- 같은 장소 · 같은 옷 · 같은 폰 위치. A12(골반 높이 당기기)와 A13(손가락으로 가리키며 흔들기)은 서로 헷갈리기 쉬우니 **A12 5개 → A13 5개** 순서로 붙여서
- 원본 전송, 변환 · 이름 붙이기는 같은 방법

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
