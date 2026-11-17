# 본 촬영 1차 결과 — 2026-11-17 (S01 · S05, 72개)

- 작성: 용훈
- 관련: `03` 웹캠 촬영 계획서 v2.2 (W1·W2·W4·W7~W9), `02` 라벨링 가이드 v2.2 (R-11·R-12), `04` 8.2
- manifest: `configs/team01_main_manifest.csv` (72행, `capture_set = main_1117`)
- 영상 위치 (Git 제외): `data/team_projects/team01/main/rgb/videos/`

## 1. 무엇을 찍었나

| 항목 | 값 |
| --- | --- |
| 사람 | S01 (VICON 배우 KHM과 같은 사람 → 학습 고정), S05 |
| 동작 | 신호 A01~A15 + 일상 동작 D01 휘슬 · D02 팔짱 · D03 뒷머리 = 18개 |
| 반복 | 동작마다 2회 (T01·T02). 단, S05는 A02 1회 · A03 3회 (아래 4장) |
| 영상 | 72개 · 세로 1080×1920 · 30 FPS · 3.6~6.8초 · 합계 10,741 Frame |
| 촬영 | 정면, 무늬 없는 벽 앞, 반팔, 신호 앞 C01 손 들기 / 일상 동작은 C01 없음 |

## 2. 파일 이름을 붙인 방법

- 영상을 카카오톡으로 받아 파일 이름이 `KakaoTalk_20261117_HHMMSSmmm.mp4`(받은 시각)였음
- 카카오톡이 영상을 다시 압축(인코더 `Lavf`)하면서 **촬영 시각 정보가 사라졌고**, 받은 순서도 일부 섞여 있었음 (예: S01 A01 ↔ A02, 두 사람 모두 A15 사이에 D01)
- 그래서 받은 순서만 믿지 않고 **72개를 모두 대표 Frame으로 확인해 동작을 판정한 뒤** `T01_S##_A##_T##` 이름을 붙임
- 같은 사람·같은 동작 안에서의 Take 번호는 받은 순서를 따름
- 원래 카톡 파일 ↔ clip_id 대응표는 Git 제외 (`data/team_projects/team01/main/main_shoot_1117_map_qa.csv`)

## 3. 품질 검사 결과

| 검사 (`03` 6장 · 10장) | 결과 |
| --- | --- |
| 해상도·FPS (W7) | 72개 모두 1080×1920 · 30 FPS ✅ |
| 화면 속 사람 수 (Q-01) | 여러 사람 Frame 0 · 사람 못 찾은 Frame 0 ✅ |
| 손목·발목 conf (Q-02·Q-04) | 모든 영상에서 conf < 0.50 Frame 없음 ✅ |
| 크기 | 어깨 폭 중앙값 200~238 px (손 검출 실험 기준 60 px의 3배 이상) ✅ |
| C01 규칙 (W6) | 신호 Take 모두 C01 있음, 일상 동작은 C01 없음 ✅ |
| 판정 | **PASS 71 · REVIEW 1** |

YOLO Pose(`yolo11n-pose`), 3 Frame마다 1장 검사. 대표 인물 = 가장 큰 BBox (D6).

### 3.1 손 검출 (`scripts/day04_hand_detect_check.py`, crop · 원본 크기)

| | 왼손 | 오른손 |
| --- | --- | --- |
| 전체 (손목 4,350개) | 93.9% | 89.7% |

| 동작 | 검출률 | 비고 |
| --- | --- | --- |
| A03 U파울 | 84.4% | 스모크(세로·긴 소매) 73.1% → 반팔·새 촬영 조건에서 상승 |
| A15 교체 | 86.2% | 손을 펴서 교차 → 손이 보임 |
| **D02 팔짱** | **44.8%** | 손을 팔 아래로 숨김 → 의도대로 손이 안 보임 |
| D03 뒷머리 | 77.8% | 머리 뒤 손이 가려짐 |
| 그 밖의 신호 | 90~100% | |

- A15(86%)와 D02(45%)가 손 검출 여부로 뚜렷하게 갈림 → `04` E-06(손 Feature)의 "찾음/못 찾음" 표시가 A15 ↔ 팔짱 구분 근거가 될 수 있음
- 손 모델: mediapipe 패키지 내장 모델로 실행 (`hand_detect_report.md` 6장과 같은 한계)

## 4. 고칠 점과 다음 촬영

| # | 내용 | 처리 |
| --- | --- | --- |
| 1 | S05 A02 홀딩 1개, A03 U파울 3개 | 다음 촬영에서 S05 A02 4개, A03 2개 |
| 2 | `T01_S05_A03_T01` — C01 손 들 때 손끝이 화면 위 끝에 닿음 (손목은 안) | REVIEW. A03이 3개라 재촬영 불필요. 다음부터 손끝 위 여유 확인 |
| 3 | 손끝이 화면 옆 끝 근처 (S05 A14 왼쪽, S01 A03 오른쪽) | 손목은 안쪽 → PASS. 폰을 반 걸음 뒤로 |
| 4 | 카카오톡 전송 → 재압축 · 촬영 시각 손실 · 순서 섞임 | 다음부터 USB · 구글 드라이브로 원본 전송 (`03` 6장) |

### 남은 촬영 (동작별 5회 목표, W4)

| 사람 | 지금 | 더 찍을 것 | Take 시작 |
| --- | --- | --- | --- |
| S01 | 36 | 18동작 × 3 = 54 | T03 |
| S05 | 36 | A02 × 4, A03 × 2, 나머지 16동작 × 3 = 54 | A02 T02 · A03 T04 · 나머지 T03 |
| S02 (테스트 전용) · S03 · S04 | 0 | 18동작 × 5 = 각 90 | T01 |

합계 450개 중 72개 완료 (16%).

## 5. 다시 돌리기

```bash
# 4일차 팀 확장 STEP 2~5를 본 촬영 폴더로 (스모크 결과와 섞이지 않게 출력 폴더도 분리)
M=configs/team01_main_manifest.csv
V=data/team_projects/team01/main/rgb/videos
O=reports/team_projects/team01/main
uv run python scripts/day04_check_team_videos.py --manifest $M --video-dir $V --output $O/tables/video_inventory.csv
uv run python scripts/day04_build_team_sequences.py --manifest $M --video-dir $V --output-dir $O/sequences
uv run --with mediapipe python scripts/day04_hand_detect_check.py --manifest $M --video-dir $V \
  --sequence-dir $O/sequences --output-dir $O/tables --scales 1.0 --modes crop
```
