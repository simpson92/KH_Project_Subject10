# 1조 RGB 데이터 준비 결과 (4일차 ⑧ · 5일차 Summary)

> 4일차 `팀별 결과 보고서 작성 틀` + 5일차 `day05_summary.md` 항목을 1조 본 촬영 데이터로 채운 문서입니다.
> 작성 2026-11-18 · 기준 manifest `configs/team01_main_manifest.csv` (451행)
>
> ⚠ **73개 clip(S01 36 · S05 37, 11-17 1차 촬영분)은 STEP 4 Sequence CSV를 아직 다시 만들지 않아** 자동 QA가 `NO_SEQUENCE` 입니다.
> 아래 숫자 중 *(현재)* 표시는 378개 기준, *(완료 후)* 는 451개 기준 예상값입니다. 마무리 명령은 8장.

## 1. 프로젝트와 Action

| 항목 | 값 |
| --- | --- |
| 팀 · 프로젝트 | 1조 · 농구 심판 수신호 인식 → 전광판 |
| clip 동작 | 18개 = 신호 A01~A15 + 일상 동작 D01 휘슬 · D02 팔짱 · D03 뒷머리 |
| 학습 라벨 | 16개 = A01~A15 + **N00** (D01~D03을 하나로, `train_label` 열) |
| 촬영 Subject | 5명 (S01~S05) · 사람마다 18동작 × 5 Take |
| 전체 MP4 | **451** = 450 + S05 A03 대체 Take 1 (`T01_S05_A03_T06`) |
| 형식 | 세로 1080×1920 · 30 FPS (03 문서 W7). 폰 원본은 변환 후 사용, 원본은 각자 PC에 보관 |

## 2. 데이터 검사 (STEP 2 · 영상 확인)

| 항목 | 결과 |
| --- | --- |
| 정상 재생 MP4 | 451 / 451 (STEP 2 `READY`) |
| FPS 이상 | 1개 — `T01_S01_A10_T04` FPS 29.31 (Frame 빠짐) |
| 누락 | 1차: 카톡 전송 중 빠진 영상 1개 찾아서 넣음 · 4차: S03 A12 · A13 10개 빠짐 → 같은 날 보충 |
| 라벨 오류 | 2차 카톡 파일 `S01 A12_T03 ↔ A13_T05` 뒤바뀜 → 확대 비교 후 바로잡음 |
| 대표 Frame 검토 | **451개 전부** 영상마다 9장 strip으로 동작 · Take 순서 확인 (헷갈리는 쌍 A08↔A12, A09↔A11, A12↔A13은 확대) |
| HDR | S02 · S03 원본이 HDR(HLG) → 톤매핑 없이 변환(색이 옅음). conf는 SDR과 같아 그대로 사용 (`main_shoot_1118_s02_s03_report.md`) |

## 3. Pose / Feature / Sequence

| 항목 | 결과 |
| --- | --- |
| Pose 모델 | `yolo11n-pose.pt` (imgsz 640, person conf 0.25, keypoint conf 0.50 — `configs/project.json`) |
| 기본 38-D | 사용 (STEP 4 Sequence CSV: 정규화 좌표 34 + 팔꿈치 · 무릎 각도 4) |
| 팀 추가 Feature | G-01~G-10 (손목 거리 · 손목 높이 · 손목 교차 · 발목 내밈 …) — **명세는 04 문서 5.2에 있음, 코드는 아직** (9장) |
| 손 검출 실험 | YOLO 손목 crop → MediaPipe Hand. A15 교체 85~91% ↔ D02 팔짱 48~55% → 손 "찾음/못 찾음" 자체가 신호 (`hand_detect_report.md`) |
| Sequence CSV | *(현재)* 378개 · 61,822 Frame · clip 평균 164 Frame(117~250) · pose_ok 100% · 여러 사람 Frame 0 |
| 약한 관절 | 귀(`right_ear`)가 가장 낮음, 그 다음 발목 (clip 최솟값 0.76). 손목 최솟값 0.92 이상 |

사람별 YOLO 평균 conf *(현재)*: S01 0.967 · S02 0.968 · S03 0.970 · S04 0.971 · S05 0.967

## 4. QA 결과 (5일차 실습 27~29번 → `scripts/day05_team_*.py`)

### 4.1 자동 QA — 동작별 중요 관절 (`day05_team_keypoint_qa.py`)

실습 27번 기준을 그대로 사용: pose_ratio < 0.90 · single_person_ratio < 0.90 · mean_conf < 0.50 · 중요 관절 low 비율 > 0.20 → `CHECK`.
중요 관절은 18동작마다 따로 정함 (`src/team_dataset_utils.py` `ACTION_CRITICAL_KEYPOINTS`, 예: A10 = 다리 + 오른팔, A11 · D01 · D03 = 팔 + 코).

| qa_status | 개수 *(현재)* | 비고 |
| --- | --- | --- |
| AUTO_PASS | 377 | 중요 관절 low 비율 **0.0** (모든 clip), pose · 한 사람 비율 1.0 |
| CHECK | 1 | `T01_S01_A10_T04` — 수치는 정상, manifest `REVIEW`라서 CHECK (1조 추가 규칙) |
| NO_SEQUENCE | 73 | STEP 4 다시 실행 필요 |

> **해석** — 실습 기준은 1조 데이터에 아주 느슨합니다(모든 clip이 기준과 거리가 멂).
> 실제로 걸러야 했던 문제(파일 뒤바뀜 · Take 누락 · 손끝 잘림 · Frame 빠짐)는 **conf로는 안 보이고 영상을 봐야** 찾을 수 있었습니다.
> → 자동 QA는 "사람을 못 찾음 · 여러 사람" 같은 큰 사고 탐지용, 라벨 정확도는 Human QA가 책임.

### 4.2 Human QA (`team01_clip_review.csv`)

실습처럼 **CHECK 먼저 자세히 → AUTO_PASS도 전수 확인** (451개 모두 strip 확인을 끝낸 상태라 전부 기록).
다시 돌려도 적어 둔 결정은 지워지지 않음 (실습 27번과 다른 점).

| human_decision | 개수 | clip · 이유 |
| --- | --- | --- |
| PASS | 449 | 동작 · Take 순서 맞음. 바로잡은 2개(S01 A12_T03 · A13_T05)는 note에 기록 |
| RESHOOT | 1 | `T01_S01_A10_T04` — FPS 29.31 · 끝 차렷 잘림. **킥 동작은 끝까지 있음** → S01 재촬영이 어려우면 PASS로 바꿔도 됨 (완료 후 Train A10이 14 → 15) |
| EXCLUDE | 1 | `T01_S05_A03_T01` — C01 손끝이 화면 밖. `T06`이 대체 |

`day05_team_validate_review.py` 추가 규칙: CHECK인데 note 없이 PASS → 실패 · NO_SEQUENCE 남아 있으면 실패 (`--allow-no-sequence`로 임시 통과)

### 4.3 Approved Manifest (`team01_approved_manifest.csv`)

- *(현재)* 377개 — PASS 이면서 Sequence CSV가 있는 clip
- *(완료 후)* **449개** — 신호 A01~A15 각 25개(A10만 24) · N00 75개

## 5. 데이터 분할 (실습 30 · 31번 → `day05_team_subject_split.py` · `day05_team_check_leakage.py`)

설정 `configs/team01_subject_split.json`

| split | Subject | clip *(현재)* | clip *(완료 후)* | 이유 |
| --- | --- | --- | --- | --- |
| Train | S01 · S03 · S04 | 233 | 269 | S01 = VICON 배우 → 학습 고정 (04 문서 8.2). S03(HDR) · S04(SDR) 둘 다 넣어 두 화질을 학습 |
| Validation | **S05** (제안) | 54 | 90 | S03 · S04 · S05 중 1명(04 문서 8.2). S03을 빼면 HDR 영상이 Train에 하나도 없고 Test(S02)만 HDR이 됨 → S05 |
| Test | S02 | 90 | 90 | 03 문서 W9 — 마지막에 한 번만 |

- split × 라벨 clip 수 → `main/dataset/splits/split_counts.csv`
- Leakage 검사 *(현재)*: Subject overlap 없음 · Clip overlap 없음 · 같은 영상 파일 없음 · 각 split 18동작 모두 있음 → **Leakage check: OK**
- 좌우 반전 증강: 아직 만들지 않음. 만들 때 **Split 다음에 Train에만** (04 문서 8.2 · WS-05)

### 5.1 Random Frame Split 비교 (실습 32번 → `day05_team_frame_split_demo.py`)

| | Frame 무작위 (`train_test_split` 0.2 · seed 42) | 사람 단위 (1조) |
| --- | --- | --- |
| Train / Test Frame | 49,321 / 12,331 | — |
| Subject overlap | **5명 전부** | 0 |
| Clip overlap | **test clip 377개 중 377개** | 0 |

→ Frame을 섞으면 바로 옆 Frame(1/30초 차이)이 Train · Test에 같이 들어가 6일차 점수가 부풀려짐

### 5.2 6일차 전에 알아 둘 것

- **N00가 신호보다 3배 많음** (일상 동작 3개를 하나로 묶어서, Train *(현재)* 39 vs 13 · *(완료 후)* 45 vs 15) → 6일차 RF에 `class_weight="balanced"` 또는 N00 줄이기 검토
- Validation이 1명이라 점수가 그 사람에게 치우칠 수 있음 → 필요하면 S03 · S04 · S05를 돌아가며 Validation (04 문서 8.2 교차 검증)

## 6. VICON 활용 계획

- VICON 배우 = S01 (스모크 · 본 촬영과 같은 사람 번호) → VICON 데이터도 **Train에만** (WS-06)
- 관절 매핑 · 정규화: `01_관절_매핑_명세.md` (VICON → COCO 17 + 손등 2), 3D JSON 추출 결과 `reports/vicon/extract_3d_report.csv`
- RGB와 구분: manifest `source_type` 열 (본 촬영 = `phone`), VICON은 별도 폴더 · 별도 표

## 7. 4 · 5일차 빠진 것 점검

### 4일차 산출물 (교재 ①~⑧ · 완료 체크리스트)

| | 산출물 | 상태 | 위치 / 남은 일 |
| --- | --- | --- | --- |
| ① | Action 정의표 | ✅ | `configs/team01_action_classes.csv` |
| ② | Clip Manifest | ✅ | `configs/team01_main_manifest.csv` 451행 |
| ③ | 영상 검사표 | ✅ | STEP 2 `main/tables/video_inventory.csv` (Git 제외) · 촬영별 보고서 |
| ④ | Sequence CSV | 🟡 378 / 451 | 73개 STEP 4 (8장) |
| ⑤ | Sequence QA 표 | ✅ | `main/dataset/team01_clip_qa.csv` · `pose_conf_*` · `hand_detect_*` |
| ⑥ | 팀별 Feature 명세 | 🟡 | 명세는 04 문서 5.2 (G-01~G-10). **단위 · 결측 처리 칸과 코드(`src/team_feature_utils.py`) 없음** |
| ⑦ | Dataset Split 표 | ✅ | `main/dataset/splits/` (이번) |
| ⑧ | 결과 보고서 MD | ✅ | 이 문서 |
| — | 좌우 반전 증강 분리 | ⬜ | 아직 증강 안 함 — 만들 때 Train에만 |
| — | 촬영 로그 `webcam_log.csv` (카메라 높이 · 거리) | ⬜ | 03 문서 9.1 양식만 있음. 실제 값 기록 없음 → 기억나는 값이라도 남기기 권장 |

### 5일차 (교재 49장 최종 체크리스트)

| 항목 | 상태 | 비고 |
| --- | --- | --- |
| MP4 검사 · Keypoint CSV | 🟡 | 실습 26번 Keypoint CSV 대신 STEP 4 Sequence CSV 사용 (같은 `conf_*` · 좌표 포함). 73개 남음 |
| 자동 QA (AUTO_PASS / CHECK) | ✅ | 동작별 중요 관절 18개 정의 |
| Human QA | ✅ | 451개 결정 · note |
| Approved Manifest | 🟡 | 377 → STEP 4 후 449 |
| Subject-wise Split · Leakage | ✅ | Leakage check: OK |
| Random Frame Split 비교 | ✅ | 5.1 |
| Mini Challenge | ➖ | 1조는 본 촬영 451개에 같은 흐름을 적용한 것으로 대신 |
| 실습 데이터 `reports/day05/day05_summary.md` | ⬜ | 실습용 18개(S01~S03 · 6동작) 결과 요약은 레포에 없음 — 25~32번을 돌린 사람이 추가 |
| README Day 05 | ✅ | 이번 |
| `configs/day05_subject_split.json` | ➖ | 실습용 그대로 둠. 1조는 `team01_subject_split.json` |

## 8. 남은 73개 마무리 (각자 PC)

```bash
M=configs/team01_main_manifest.csv
V=data/team_projects/team01/main/rgb/videos
O=reports/team_projects/team01/main
# 1) 받은 Sequence 378개를 $O/sequences/ 에 풀기 → STEP 4는 없는 73개만 계산
uv run python scripts/day04_build_team_sequences.py --manifest $M --video-dir $V --output-dir $O/sequences
# 2) 5일차 흐름 (Human QA 결정은 그대로 남음)
uv run python scripts/day05_team_keypoint_qa.py
uv run python scripts/day05_team_validate_review.py          # NO_SEQUENCE 0이면 OK
uv run python scripts/day05_team_approved_manifest.py        # 449
uv run python scripts/day05_team_subject_split.py            # 269 / 90 / 90
uv run python scripts/day05_team_check_leakage.py
uv run python scripts/day05_team_frame_split_demo.py
```

## 9. 다음 단계

- **6일차 전**: 73개 STEP 4 → 449개 확정 · Validation S05 팀 확정 (04 문서 8.2 갱신) · A10_T04 PASS/재촬영 결정
- **6일차**: 30-step × 38 = 1140-D · RF 300 · Val Accuracy / Macro F1 / Confusion Matrix (Test S02는 마지막에 한 번)
- **그 다음**: G-01~G-10 코드 + 손 검출 Feature → 38-D 대비 개선 비교 (04 문서 F1) → 작은 모델로 FastAPI 업로드 테스트 1~2개
