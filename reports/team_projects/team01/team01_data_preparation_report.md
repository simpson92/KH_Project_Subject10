# 1조 RGB 데이터 준비 결과 (4일차 ⑧ · 5일차 Summary)

> 4일차 `팀별 결과 보고서 작성 틀` + 5일차 `day05_summary.md` 항목을 1조 본 촬영 데이터로 채운 문서입니다.
> 작성 2026-11-18 · 기준 manifest `configs/team01_main_manifest.csv` (451행)
>
> ✅ **451개 확정** (2026-11-18 저녁) — Sequence CSV 451개가 Git에 올라가 있음 (`main/sequences/`, 다시 만들 때 덮어쓰지 않는 확정본).
> 아래 숫자는 모두 451개 기준. 다시 만드는 명령은 8장.

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
| Sequence CSV | **451개** · 72,703 Frame · clip 평균 161 Frame = 5.4초 (109~250) · pose_ok 100% · 여러 사람 Frame 0 |
| 약한 관절 | 귀(`right_ear`)가 가장 낮음, 그 다음 발목 (clip 최솟값 0.76). 손목 최솟값 0.92 이상 |

사람별 YOLO 평균 conf: S01 0.966 · S02 0.968 · S03 0.970 · S04 0.971 · S05 0.967 (전체 0.968 · 팔 0.989 · conf 표 PASS 451, `main/pose_conf_summary.md`)

## 4. QA 결과 (5일차 실습 27~29번 → `scripts/day05_team_*.py`)

### 4.1 자동 QA — 동작별 중요 관절 (`day05_team_keypoint_qa.py`)

실습 27번 기준을 그대로 사용: pose_ratio < 0.90 · single_person_ratio < 0.90 · mean_conf < 0.50 · 중요 관절 low 비율 > 0.20 → `CHECK`.
중요 관절은 18동작마다 따로 정함 (`src/team_dataset_utils.py` `ACTION_CRITICAL_KEYPOINTS`, 예: A10 = 다리 + 오른팔, A11 · D01 · D03 = 팔 + 코).

| qa_status | 개수 | 비고 |
| --- | --- | --- |
| AUTO_PASS | 449 | 중요 관절 low 비율 **0.0** (모든 clip), pose · 한 사람 비율 1.0 |
| CHECK | 2 | `T01_S01_A10_T04` · `T01_S05_A03_T01` — 수치는 정상, manifest `REVIEW`라서 CHECK (1조 추가 규칙) → Human QA에서 PASS · EXCLUDE |
| NO_SEQUENCE | 0 | |

> **해석** — 실습 기준은 1조 데이터에 아주 느슨합니다(모든 clip이 기준과 거리가 멂).
> 실제로 걸러야 했던 문제(파일 뒤바뀜 · Take 누락 · 손끝 잘림 · Frame 빠짐)는 **conf로는 안 보이고 영상을 봐야** 찾을 수 있었습니다.
> → 자동 QA는 "사람을 못 찾음 · 여러 사람" 같은 큰 사고 탐지용, 라벨 정확도는 Human QA가 책임.

### 4.2 Human QA (`team01_clip_review.csv`)

실습처럼 **CHECK 먼저 자세히 → AUTO_PASS도 전수 확인** (451개 모두 strip 확인을 끝낸 상태라 전부 기록).
다시 돌려도 적어 둔 결정은 지워지지 않음 (실습 27번과 다른 점).

| human_decision | 개수 | clip · 이유 |
| --- | --- | --- |
| PASS | 450 | 동작 · Take 순서 맞음. 바로잡은 2개(S01 A12_T03 · A13_T05)는 note에 기록 |
| PASS (CHECK) | (위 450에 포함) | `T01_S01_A10_T04` — FPS 29.31 · 끝 차렷 잘림이지만 **킥 동작은 끝까지 있음** → 재촬영 대신 PASS (11-18 팀 결정) |
| RESHOOT | 0 | |
| EXCLUDE | 1 | `T01_S05_A03_T01` — C01 손끝이 화면 밖. `T06`이 대체 |

`day05_team_validate_review.py` 추가 규칙: CHECK인데 note 없이 PASS → 실패 · NO_SEQUENCE 남아 있으면 실패 (`--allow-no-sequence`로 임시 통과)

### 4.3 Approved Manifest (`team01_approved_manifest.csv`)

- **450개** — 신호 A01~A15 각 25개 · N00 75개 (EXCLUDE 1개 빠짐)

## 5. 데이터 분할 (실습 30 · 31번 → `day05_team_subject_split.py` · `day05_team_check_leakage.py`)

설정 `configs/team01_subject_split.json`

| split | Subject | clip | 이유 |
| --- | --- | --- | --- |
| Train | S01 · S03 · S04 | **270** (신호 15 · N00 45) | S01 = VICON 배우 → 학습 고정 (04 문서 8.2). S03(HDR) · S04(SDR) 둘 다 넣어 두 화질을 학습 |
| Validation | **S05** (11-18 확정, 04 문서 v2.3) | **90** (신호 5 · N00 15) | S03 · S04 · S05 중 1명(04 문서 8.2). S03을 빼면 HDR 영상이 Train에 하나도 없고 Test(S02)만 HDR이 됨 → S05 |
| Test | S02 | **90** (신호 5 · N00 15) | 03 문서 W9 — 마지막에 한 번만 |

- split × 라벨 clip 수 → `main/dataset/splits/split_counts.csv`
- Leakage 검사: Subject overlap 없음 · Clip overlap 없음 · 같은 영상 파일 없음 · 각 split 18동작 모두 있음 → **Leakage check: OK**
- 좌우 반전 증강: 아직 만들지 않음. 만들 때 **Split 다음에 Train에만** (04 문서 8.2 · WS-05)

### 5.1 Random Frame Split 비교 (실습 32번 → `day05_team_frame_split_demo.py`)

| | Frame 무작위 (`train_test_split` 0.2 · seed 42) | 사람 단위 (1조) |
| --- | --- | --- |
| Train / Test Frame | 58,036 / 14,509 | — |
| Subject overlap | **5명 전부** | 0 |
| Clip overlap | **test clip 450개 중 450개** | 0 |

→ Frame을 섞으면 바로 옆 Frame(1/30초 차이)이 Train · Test에 같이 들어가 6일차 점수가 부풀려짐

### 5.2 6일차 전에 알아 둘 것

- **N00가 신호보다 3배 많음** (일상 동작 3개를 하나로 묶어서, Train 45 vs 15) → 04 문서 기준(N00은 신호 1클래스의 1~2배, 넘치면 `class_weight` 또는 평가용 보관)을 넘음. 04 문서 7장 RF 설정에 이미 `class_weight="balanced"`가 있어 6일차는 그대로 시작하고, N00 Recall · Precision이 이상하면 N00을 줄여 비교
- **C01 지름길 주의** — 일상 동작은 "신호를 일상 동작과 헷갈리지 않게" 넣은 것(02 문서 5.5). 그런데 신호 clip에만 앞에 C01 손 들기가 있어서, clip 전체로 학습하면 모델이 **팔 모양이 아니라 "손을 들었나"로** 휘슬(D01) ↔ 머리 가격(A11)을 가를 수 있음. 그러면 Validation 점수는 좋아도 실시간(손 들기 없이 들어오는 윈도우)에서는 그 구분이 무너짐 → 5.3 구간 라벨로 **신호 구간만 잘라 학습한 결과와 비교**하면 확인 가능
- Validation이 1명이라 점수가 그 사람에게 치우칠 수 있음 → 필요하면 S03 · S04 · S05를 돌아가며 Validation (04 문서 8.2 교차 검증)

### 5.3 구간 라벨 `labels.csv` 자동 초안 (02 문서 5.2 · 6장 · 7장)

`scripts/day05_team_make_labels.py` → `main/labels/team01_labels.csv` (labeler = `auto`)

- 신호 clip = **N00 → X → C01 → X → A## → X → N00**, 일상 동작 clip = 전체 N00 (R-11)
- 차렷 = 손목 · 발목이 처음/끝 자세에서 0.2 안 · C01 = 오른손만 머리 위의 가장 높은 구간 · 정지형 신호(A01 · A03 · A04 · A10 · A15)는 가운데 자세와 가까운 구간
- **영상 21개(S04 A01~A15 + S02 · S03 이상값 6개)를 Frame 사진과 대조** → 경계 차이 대략 0.1~0.3초. 처음 규칙에서 놓친 것(C01 손이 흔들림, 왼손 고정 A14, 정지형 신호의 끝)을 고쳐 반영
- 결과 *(451 clip)*: 2,578구간 · PASS 448 · EXCLUDE 1 · **REVIEW 2** (`T01_S02_A14_T02` 신호를 못 찾음, `T01_S04_A15_T01` 신호 0.47초) · 검증 `day05_team_validate_labels.py` 오류 0 · 경고 5
- 전체 길이: N00 1,061초 · X 447초 · C01 346초 · 신호 570초 (동작당 약 38초)
- 남은 일: 02 문서 8장 5단계 **교차 검수 10%** — `day05_team_label_sheet.py --clip-id ...`로 모아보기를 만들어 눈으로 확인, 고친 clip은 labeler를 `S##`로

### 5.4 팀 Feature G-01 ~ G-10 (04 문서 5.2, `src/team_feature_utils.py`)

신호 구간 · C01 구간 · 일상 동작 clip의 평균 (`day05_team_signal_feature_check.py`, 451 clip 기준). 거리 = 어깨 폭 단위, 높이 + = 머리보다 위

| 라벨 | clip | G-01 손목 거리 | G-02 오른손목 높이 | G-04 손목–어깨 | G-05 팔꿈치–어깨 높이 | G-06 오른어깨 각도 | G-08 손목 교차 | G-09 손목–골반 | G-10 발목 간격 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | 25 | 1.40 | -1.57 | 0.99 | -0.50 | 46° | 1.00 | 0.63 | 0.84 |
| A02 | 25 | 0.36 | -1.66 | 1.05 | -0.70 | 21° | 1.00 | 0.64 | 0.94 |
| A03 | 25 | 0.68 | 0.95 | 1.54 | 0.72 | 159° | 1.00 | 3.85 | 1.11 |
| A04 | 25 | 0.86 | -0.34 | 0.25 | -0.26 | 62° | 1.00 | 1.52 | 0.88 |
| A05 | 25 | 1.27 | -1.02 | 0.40 | -0.55 | 39° | 1.00 | 1.29 | 0.93 |
| A06 | 25 | 0.48 | -1.02 | 0.54 | -0.46 | 33° | 0.91 | 1.26 | 0.88 |
| A07 | 25 | 1.41 | -1.47 | 0.80 | -0.66 | 27° | 1.00 | 0.97 | 0.96 |
| A08 | 25 | 2.04 | -0.73 | 0.37 | -0.29 | 61° | 1.00 | 0.97 | 0.91 |
| A09 | 25 | 2.66 | -0.13 | 0.64 | -0.07 | 92° | 1.00 | 1.27 | 0.90 |
| A10 | 25 | 1.89 | -2.06 | 1.72 | -0.79 | 21° | 1.00 | 0.60 | 1.26 |
| A11 | 25 | 3.03 | 0.43 | 0.91 | 0.28 | 128° | 1.00 | 1.53 | 0.95 |
| A12 | 25 | 1.80 | -1.74 | 1.25 | -0.56 | 45° | 1.00 | 0.61 | 0.88 |
| A13 | 25 | 1.96 | -1.57 | 1.03 | -0.72 | 32° | 1.00 | 0.80 | 1.00 |
| A14 | 24 | 2.33 | -1.79 | 1.58 | -0.66 | 40° | 1.00 | 0.86 | 0.93 |
| A15 | 25 | 0.15 | -0.80 | 0.38 | -0.53 | 12° | 0.76 | 1.72 | 0.98 |
| C01 | 374 | 4.07 | 1.35 | 1.82 | 0.74 | 155° | 1.00 | 2.12 | 0.96 |
| N00(D01) | 25 | 1.79 | -1.42 | 0.84 | -0.68 | 25° | 1.00 | 0.78 | 0.95 |
| N00(D02) | 25 | 0.78 | -1.65 | 1.09 | -0.72 | 16° | 0.28 | 0.84 | 0.91 |
| N00(D03) | 25 | 2.18 | -1.05 | 1.05 | -0.35 | 67° | 1.00 | 0.93 | 0.95 |

- **설계대로 갈리는 것**: G-02 손목 높이(A03 · A11 · C01 머리 위 ↔ 나머지), G-01 손목 거리(A15 0.15 · A02 0.36 ↔ A09 · A11 3배), G-06 어깨 각도(A09 92° ↔ A08 61°), G-10 발목 간격(A10 1.26 ↔ 나머지 0.84~1.11), G-05(A11 + ↔ A12 −0.56 ↔ A08 −0.29)
- **설계와 다른 것**: G-04는 A09보다 A08이 더 작음(팔꿈치를 접어 손목이 어깨 옆) · G-09는 A01(0.63)과 팔 내린 신호(A02 · A10 · A12 ≈ 0.6)가 비슷함 · **G-08 손목 교차는 A15(0.76)보다 팔짱 D02(0.28)가 더 많이 교차** → A15 ↔ 팔짱은 G-08이 아니라 G-01(0.15 vs 0.78)과 손 검출(B5)로 갈라야 함
- → 6일차 E-01(38-D ↔ 48-D)에서 48-D가 이기는지, 이긴다면 어느 Feature 때문인지(RF feature_importances_) 확인

### 5.5 5일차 수업 Keypoint CSV 형식 (`day05_team_export_keypoints.py`)

- 6일차 수업 코드는 5일차 Keypoint CSV(`<관절>_x_px · _y_px · _x_norm · _y_norm · _conf`)를 읽음 → Sequence CSV에서 같은 YOLO 결과를 그대로 옮김 (`x_norm = x_px ÷ 영상 너비`, YOLO 다시 안 돌림)
- 위치 `data/team_projects/team01/main/keypoints/<S##>/<clip_id>.csv` (Git 제외). Approved · split CSV에 수업 이름 열 `subject · take · keypoint_csv · video_path` 추가

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
| ④ | Sequence CSV | ✅ 451 | `main/sequences/` 확정본 (Git) |
| ⑤ | Sequence QA 표 | ✅ | `main/dataset/team01_clip_qa.csv` · `pose_conf_*` · `hand_detect_*` |
| ⑥ | 팀별 Feature 명세 | ✅ | 명세 04 문서 5.2 · 코드 `src/team_feature_utils.py` (단위 = 어깨 폭, 결측 = NaN, FR-01 · FR-05) · 확인표 5.4 |
| ⑦ | Dataset Split 표 | ✅ | `main/dataset/splits/` (이번) |
| ⑧ | 결과 보고서 MD | ✅ | 이 문서 |
| — | 좌우 반전 증강 분리 | ⬜ | 아직 증강 안 함 — 만들 때 Train에만 |
| — | 촬영 로그 `webcam_log.csv` | 🟡 | `main/webcam_log.csv` 451행 자동 채움. **카메라 높이 · 거리 · 진행자(S##) 3칸은 사람이 기입** |
| — | 구간 라벨 `labels.csv` (02 문서 5.2) | 🟡 | 자동 초안 451 clip (5.3). 교차 검수 10% · REVIEW 2개 남음 (7일차 전까지) |
| — | 05 문서 실시간 입력 세로(W7) 반영 | ✅ | 05 문서 v2.1: RP-1 방향 확인 · 회전, API-07 업로드 테스트(계획), ER-07 |

### 5일차 (교재 49장 최종 체크리스트)

| 항목 | 상태 | 비고 |
| --- | --- | --- |
| MP4 검사 · Keypoint CSV | ✅ | STEP 4 Sequence CSV 451 → 수업 형식으로 내보내기 (5.5) |
| 자동 QA (AUTO_PASS / CHECK) | ✅ | 동작별 중요 관절 18개 정의 |
| Human QA | ✅ | 451개 결정 · note |
| Approved Manifest | ✅ | 450 |
| Subject-wise Split · Leakage | ✅ | Leakage check: OK |
| Random Frame Split 비교 | ✅ | 5.1 |
| Mini Challenge | ➖ | 1조는 본 촬영 451개에 같은 흐름을 적용한 것으로 대신 |
| 실습 데이터 `reports/day05/day05_summary.md` | ⬜ | 실습용 18개(S01~S03 · 6동작) 결과 요약은 레포에 없음 — 25~32번을 돌린 사람이 추가 |
| README Day 05 | ✅ | 이번 |
| `configs/day05_subject_split.json` | ➖ | 실습용 그대로 둠. 1조는 `team01_subject_split.json` |

## 8. 다시 만들기 (각자 PC)

```bash
M=configs/team01_main_manifest.csv
V=data/team_projects/team01/main/rgb/videos
O=reports/team_projects/team01/main
# 1) Sequence는 Git의 확정본 451개를 씀 — 이미 있으면 STEP 4는 건너뜀 (--overwrite 금지)
uv run python scripts/day04_build_team_sequences.py --manifest $M --video-dir $V --output-dir $O/sequences
# 2) 5일차 흐름 (Human QA 결정은 그대로 남음)
uv run python scripts/day05_team_keypoint_qa.py
uv run python scripts/day05_team_validate_review.py          # NO_SEQUENCE 0이면 OK
uv run python scripts/day05_team_approved_manifest.py        # 450
uv run python scripts/day05_team_subject_split.py            # 270 / 90 / 90
uv run python scripts/day05_team_check_leakage.py
uv run python scripts/day05_team_frame_split_demo.py
# 3) 구간 라벨 · 수업 형식 · 확인표 · 촬영 로그 (사람이 고친 라벨 · 실측값은 그대로 남음)
uv run python scripts/day05_team_make_labels.py
uv run python scripts/day05_team_validate_labels.py
uv run python scripts/day05_team_export_keypoints.py
uv run python scripts/day05_team_signal_feature_check.py
uv run python scripts/day05_team_make_webcam_log.py
```

## 9. 다음 단계

- **5일차 끝 (11-18)**: Sequence 451 확정 · Approved 450 · Split 270 / 90 / 90 · Leakage OK · 테스트 145 passed
- **남은 사람 손 작업**: 라벨 교차 검수 10% (7일차 전까지) · `webcam_log.csv` 실측 3칸
- **6일차**: 30-step × 38 = 1140-D · RF 300 · Val Accuracy / Macro F1 / Confusion Matrix (Test S02는 마지막에 한 번)
- **6일차 비교 실험**: E-01 38-D ↔ 48-D(G-01~G-10, 코드 준비됨) · clip 전체 ↔ 신호 구간만(C01 지름길 확인, 5.3 라벨)
- **그 다음**: 작은 모델로 FastAPI 업로드 테스트 1~2개 (05 문서 API-07) · 손등 Feature는 2단계(E-06, 04 문서 F6 = 미사용)
