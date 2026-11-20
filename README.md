# RefSign — 농구 심판 수신호 실시간 인식 → 전광판

교과 10 자세추정 **1조 프로젝트**입니다.
농구 심판이 파울 후 보내는 수신호(FIBA) 15개를 웹캠 영상에서 YOLO Pose로 인식해 전광판에 표시하는 것을 목표로 합니다.

```text
웹캠 영상 → YOLO Pose (COCO 17) → 38-D Frame Feature (+ 팀 Feature G-01~G-10)
   → Sequence → Random Forest → 수신호 판정 → 전광판 (FastAPI → 화면)
```

> **2026-11-20 레포 정리** — 교과 10 수업 실습(교재 번호 코드·결과)은 조원 각자 개인 레포에서 진행합니다.
> 이 레포에는 1조 프로젝트와, 1조 코드가 실제로 사용하는 교과 공통 코드만 남겼습니다.

## 신호 클래스

| ID | 내용 |
| --- | --- |
| A01~A15 | 블로킹 · 홀딩 · U파울 · 테크니컬 · 푸싱 · 트래블링 · 더블드리블 · 팔꿈치 · 샷클락 · 킥볼 · 머리 가격 · 후킹 · 백코트 · 슈팅 동작 아닌 파울 · 교체 |
| C01 | 시계 정지 (각 신호 앞) |
| N00 | 신호 없음 — 일상 동작 D01 휘슬 · D02 팔짱 · D03 뒷머리 긁기는 학습 Label N00 |
| X | 전환 구간 (구간 라벨에서만 사용) |

정의 · 라벨링 규칙: [`docs/referee/02_라벨링_가이드.md`](docs/referee/02_라벨링_가이드.md) · `configs/team01_action_classes.csv`

## 개발환경

| 항목 | 값 |
| --- | --- |
| OS | Windows + WSL2 Ubuntu |
| Python | 3.12 (`uv` 가상환경) |
| PyTorch | 2.15.0+cu132 (CUDA 13.2) |
| Ultralytics | 8.4.203 (`yolo11n-pose.pt`) |
| OpenCV | 5.0.0 |
| GPU | NVIDIA GeForce RTX 3060 |

```bash
cd ~/ai_vision/subject10_pose_action
uv sync
source .venv/bin/activate
uv run pytest -q          # YOLO · GPU · data/ 없이 실행되는 단위 테스트
```

`data/`, 이미지 · 영상, 모델 가중치(`*.pt`, `*.joblib`), 실행 결과 CSV 대부분은 Git에 올리지 않습니다 (`.gitignore`).

## 폴더 구조

```text
KH_Project_Subject10/
├── configs/
│   ├── project.json                  # YOLO 모델 · 이미지 크기 · Confidence 기준 · 장치 (교과 공통)
│   ├── team01_action_classes.csv     # A01~A15 · C01 · N00 · D01~D03 정의
│   ├── team01_clip_manifest.csv      # 스모크 100개 clip_id
│   ├── team01_main_manifest.csv      # 본 촬영 clip_id (454행)
│   ├── team01_subject_split.json     # Train S01·S03·S04 / Val S05 / Test S02
│   ├── team01_day06_baseline.json    # 6일차 RF 38-D
│   ├── team01_day06_e01_48d.json     # E-01 48-D (38-D + G-01~G-10)
│   └── team01_day07_realtime.json    # 7일차 실시간 (Window · Confidence · Buffer + 경로 · S05 · 세로)
├── src/
│   ├── common.py · pose_utils.py · feature_utils.py · sequence_utils.py   # 교과 공통 (아래 표)
│   ├── team_feature_utils.py         # 팀 Feature G-01~G-10
│   ├── team_dataset_utils.py         # QA · Human QA · Approved · Split · Leakage
│   ├── team_label_utils.py           # 구간 라벨 초안 · V-01~V-10 검사
│   ├── team_crosscheck_utils.py      # 라벨 교차 검수
│   ├── realtime_utils.py             # 교과 7일차 실시간 공통 (수정 없이 재사용, 아래 표)
│   ├── team_realtime_utils.py        # 1조 7일차: 설정 · 16클래스 · S05 · 세로 회전 · Target
│   ├── referee_utils.py              # 스모크 테스트 공통
│   └── vicon_utils.py                # VICON GLB → 3D JSON
├── scripts/
│   ├── 17_referee_rule.py · 18_video_frame_features.py · 23_summarize_action_sequences.py   # 교과 공통 (아래 표)
│   ├── day03_* · day04_* · day05_team_* · day06_team_* · day07_team_*   # 1조 Pipeline (dayNN = 수업 진도에 맞춰 만든 날)
│   ├── referee_00~04_*.py            # 스모크 테스트 (간이 검증)
│   └── vicon_00~01_*.py              # VICON
├── tests/                            # 단위 테스트
├── docs/referee/                     # 설계 문서 00~05 · 결정 기록 · 버전 표
├── reports/
│   ├── team_projects/team01/         # 촬영 보고서 · main/ (Sequence · dataset · labels · day06 결과)
│   ├── referee/smoke/                # 스모크 결과
│   ├── vicon/                        # VICON 3D 추출 결과
│   └── day03/challenge/feature_design.md   # 3일차 Feature 설계 (04 설계서 근거)
├── artifacts/team01/day06/           # RF 학습 정보 json (joblib은 Git 제외)
└── daily_csv/                        # 날짜별 작업 기록 (한 줄 = 작업 하나, S 번호만)
```

### 남겨 둔 교과 10 공통 코드

1조 코드가 직접 import하거나 불러 쓰는 파일이라 지우면 안 됩니다.

| 파일 | 1조에서 쓰는 곳 |
| --- | --- |
| `configs/project.json` · `src/common.py` | YOLO 설정 읽기 · GPU/CPU 선택 (모든 Pose 추출 스크립트) |
| `src/pose_utils.py` | COCO 17 이름 · Skeleton · 품질 판정 · `select_main_person` (D6 = 가장 큰 BBox) |
| `src/feature_utils.py` | 관절각도 · 어깨 폭 정규화 · 좌우 반전 (38-D) |
| `src/sequence_utils.py` | 38-D Frame Feature · Resampling (`day06_team_build_sequences.py`) |
| `src/realtime_utils.py` | Sliding Window · 30 Step 예측 · Confidence · 다수결 · FPS (`day07_team_*`, 교재 7일차 그대로) |
| `scripts/17_referee_rule.py` | 1조 U파울 규칙 (`day03_batch_video_qa.py`, `tests/test_referee_rule.py`) |
| `scripts/18_video_frame_features.py` | 영상 → 38-D, 대표 인물 D6 반영본 (`day03_batch_video_qa.py`, `tests/test_main_person.py`) |
| `scripts/23_summarize_action_sequences.py` | Sequence 요약 (`tests/test_team_sequences.py`, `tests/test_team_sequence_qa.py`) |
| `reports/day03/challenge/feature_design.md` | 3일차 Feature 측정 · 결정 (04 설계서와 팀 스크립트 주석이 참조) |

### `data/` 폴더 구조 (Git 제외 — 각자 PC에 직접 넣기)

```text
data/
├── referee/smoke/
│   ├── raw/ufoul/   S01~S05_ufoul_T01~T10.mp4    # 조원 휴대폰 촬영 50개
│   ├── raw/pushing/ S01~S05_pushing_T01~T10.mp4  # 조원 휴대폰 촬영 50개
│   └── subjects.csv                         # S 번호 ↔ 이름 (실명 → 공유 금지)
├── raw/vicon/                               # Motion_Visol.zip 압축 해제 (FBX 61개)
│   └── glb/                                 # Glb.zip 압축 해제 (GLB 60개)
├── take_log.csv                             # vicon_00 실행 결과
├── vicon_3d/                                # vicon_01 실행 결과 (3D JSON 50개)
├── raw/webcam/                              # 본 촬영 원본 (폰에서 옮긴 그대로)
└── team_projects/team01/
    ├── rgb/videos/                          # 스모크 100개 (clip_id 이름)
    └── main/
        ├── rgb/videos/                      # 본 촬영 (clip_id 이름)
        ├── keypoints/S0x/                   # day05_team_export_keypoints.py 결과
        └── day06/sequences/                 # day06_team_build_sequences.py 결과
models/hand_landmarker.task                  # MediaPipe 손 모델 (hand_detect_report.md 7장)
```

- 스모크 영상 · VICON zip · 본 촬영 영상은 조 공유 폴더에서 받아요. 영상 · 사진에는 조원 얼굴이 있어서 레포나 외부에 올리지 않아요.
- 본 촬영 영상은 카카오톡 말고 USB · 구글 드라이브로 **원본** 그대로 옮겨요 (카톡은 다시 압축하고 촬영 시각을 지움).
- `take_log.csv`, `vicon_3d/`, `keypoints/`, `day06/sequences/`는 받는 게 아니라 스크립트를 실행하면 생겨요.

---

## Pipeline

### 1. 스모크 · Feature 확인 (3일차 진도)

U파울 · 푸싱 2동작 × 5명 × 10회 = 100개로 간이 검증 (본 Pipeline 아님).

| 파일 | 내용 |
| --- | --- |
| `scripts/referee_00~04_*.py` · `reports/referee/smoke/` | 이름 정리 → 영상 확인 → Pose 추출 → LOSO 학습 → 오답 렌더링 |
| `scripts/day03_batch_video_qa.py` | 스모크 전체에 18번(38-D) · 17번(U파울 규칙) · 튐 탐지 → `video_qa.csv` (PASS/REVIEW) |
| `scripts/day03_plot_feature_timeline.py` | 18번 CSV → `wrist − nose` · 팔꿈치 각도 · 어깨 폭 그래프, 튐 Frame 표 |

### 2. 촬영 영상 → Sequence (4일차 진도)

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

| 파일 | 내용 |
| --- | --- |
| `scripts/day04_make_clip_manifest.py` | 원본 이름 → `T01_S##_A##_T##` + manifest (`--apply --link`로 영상 배치) |
| `scripts/day04_check_team_videos.py` | 영상 존재 · 해상도 · FPS · 길이 표 |
| `scripts/day04_extract_team_keyframes.py` | clip별 keyframe (얼굴 → Git 제외) |
| `scripts/day04_build_team_sequences.py` | Frame → YOLO Pose → 38-D + 원본 Keypoint · conf |
| `scripts/day04_team_sequence_qa.py` | Sequence 성공률 · conf · 튐 → PASS/REVIEW |
| `scripts/day04_hand_detect_check.py` | 손 검출 가능성 (YOLO 손목 crop → MediaPipe Hand) |
| `scripts/day04_pose_conf_report.py` | 관절별 YOLO conf 표 |

- **Sequence CSV는 확정본** (`reports/team_projects/team01/main/sequences/`) — `--overwrite`로 다시 만들지 않음 (다시 만들면 전체가 바뀐 것으로 잡힘). 새 촬영분만 `--clip-id`로 추가
- 촬영 보고서: `reports/team_projects/team01/main_shoot_*.md`

### 3. QA · Split · 구간 라벨 (5일차 진도)

```
Sequence CSV ─▶ ① 자동 QA (동작별 중요 관절) ─▶ ② Human QA (PASS / RESHOOT / EXCLUDE)
   ─▶ ③ Approved Manifest ─▶ ④ Subject-wise Split ─▶ ⑤ Leakage 검사   (+ Random Frame Split 비교)
```

| 수업 번호 | 1조 스크립트 | 출력 (`reports/team_projects/team01/main/dataset/`) |
| --- | --- | --- |
| 27 | `day05_team_keypoint_qa.py` | `team01_clip_qa.csv` · `team01_clip_review.csv` (다시 돌려도 사람이 적은 결정 유지) |
| 28 | `day05_team_validate_review.py` | 검사만 (UNREVIEWED · 잘못된 값 · note 없는 CHECK PASS · NO_SEQUENCE) |
| 29 | `day05_team_approved_manifest.py` | `team01_approved_manifest.csv` (`train_label`: D01~D03 → N00) |
| 30 | `day05_team_subject_split.py` | `splits/train.csv · val.csv · test.csv · split_counts.csv` |
| 31 | `day05_team_check_leakage.py` | Subject · clip · 영상 파일 겹침 + 동작 coverage |
| 32 | `day05_team_frame_split_demo.py` | Frame 무작위 분할의 누수 확인 (학습 없음) |
| — | `day05_team_make_labels.py` · `day05_team_validate_labels.py` | `main/labels/team01_labels.csv` 구간 라벨 (N00 → X → C01 → X → 신호 → X → N00) · V-01~V-10 검사 |
| — | `day05_team_label_sheet.py` | 라벨 확인용 모아보기 (각자 PC, 얼굴이 보여 Git 제외) |
| — | `day05_team_make_crosscheck.py` · `day05_team_label_crosscheck.py` | 라벨 교차 검수 10% (표본 45개, 기준 0.2초) → `--apply`로 반영 |
| — | `day05_team_export_keypoints.py` | Sequence → 5일차 Keypoint CSV 형식 (`data/team_projects/team01/main/keypoints/`) |
| — | `day05_team_signal_feature_check.py` | 팀 Feature G-01~G-10 라벨별 평균표 |
| — | `day05_team_make_webcam_log.py` | 촬영 로그 `main/webcam_log.csv` |

- **분할 (`configs/team01_subject_split.json`)**: Train S01 · S03 · S04 (270) / Validation S05 (90) / **Test S02 (90) — 맨 마지막에 한 번만**
- 결과 요약: `reports/team_projects/team01/team01_data_preparation_report.md`

### 4. Random Forest 학습 · 평가 (6일차 진도)

```bash
uv run python scripts/day05_team_export_keypoints.py           # Keypoint CSV (처음 한 번)
uv run python scripts/day06_team_build_sequences.py            # 30 Step Sequence (38-D)
uv run python scripts/day06_team_train_rf.py
uv run python scripts/day06_team_evaluate.py                   # Validation S05
# E-01 48-D: 위 세 줄에 --config configs/team01_day06_e01_48d.json
uv run python scripts/day06_team_compare.py \
  --configs configs/team01_day06_baseline.json configs/team01_day06_e01_48d.json --seeds 10
# Test S02는 마지막에 한 번만: day06_team_evaluate.py --split test --final
```

| 파일 | 내용 | 출력 |
| --- | --- | --- |
| `day06_team_build_sequences.py` | 입력 검사 + clip → 30 Step Sequence (38-D = 1,140열, 48-D = 1,440열) | `data/.../main/day06/sequences/[run/]{split}.csv` |
| `day06_team_train_rf.py` | RF (Tree 300 · class_weight balanced · seed 42) | `artifacts/team01/day06/[run/]` |
| `day06_team_evaluate.py` | Accuracy · Macro F1 · 클래스별 · 오분류 · Confusion Matrix (`--split test`는 `--final` 필수) | `reports/.../main/day06/[run/]` |
| `day06_team_compare.py` | 두 설정 비교 · 바뀐 clip · seed 반복 평균 (Test 사용 안 함) | `reports/.../main/day06/compare/` |

**Validation S05 결과 (A04 재촬영 반영, 결정 기록 D6)**

| Feature | Accuracy | Macro F1 | seed 10회 평균 Acc |
| --- | --- | --- | --- |
| 38-D (Baseline) | 0.9333 | 0.9450 | 0.909 |
| 48-D (E-01) | 0.922 | — | 0.922 |

- 남은 오류: N00(D03 뒷머리 긁기) → A11 · A14, A04 T02 → A06, A07 → A05
- E-01 채택은 보류 (D7), 좌우 반전 규칙 미결정 (D8)

### 5. 실시간 연결 — Sliding Window · IP Webcam (7일차 진도)

6일차 38-D RF 모델을 끝없이 들어오는 Frame에 연결합니다. 점수를 올리는 게 아니라 실시간에서만 생기는 문제
(UNKNOWN · 흔들림 · 지연 · 세로 방향)를 숫자로 남기는 단계. 교재 41~45번을 참고해 1조 조건으로 새로 만들었어요.

```bash
# 0. 6일차 모델이 있어야 함 (Git에 joblib 없음 → 4번 섹션으로 각자 생성, Val 84 / 90 확인)
uv run python scripts/day07_team_check_inputs.py                                   # A. 입력 검사
uv run python scripts/day07_team_realtime_video.py --clip-id T01_S05_A06_T01       # B. 저장영상 1개 (MP4 + CSV)
uv run python scripts/day07_team_realtime_video.py --first-takes --no-video        # B. S05 18개 (동작마다 첫 Take)
uv run python scripts/day07_team_realtime_video.py --first-takes --no-video --confidence-threshold 0.35   # C. 값 하나만 바꿔 비교
uv run python scripts/day07_team_summarize_runs.py                                 # C. 설정 비교표
uv run python scripts/day07_team_ip_webcam.py --url "http://<폰 IP>:8080/video" --targets A01,A02,A03,A04,A05,A06   # D. 6개씩 3번
uv run python scripts/day07_team_analyze_log.py --all                              # E. 동작별 기록표
```

| 파일 | 내용 | 출력 |
| --- | --- | --- |
| `day07_team_check_inputs.py` | 모델 · Feature 목록 · 학습 정보 · 설정 · S05 Val 90개 영상(세로 · 길이) 검사 (교재 41번) | `reports/.../main/day07/tables/` |
| `day07_team_realtime_video.py` | S05 저장영상을 Stream처럼: Pose → 38-D → Window → 30 Step → RF → Confidence → 다수결 (교재 42번). S02 clip은 거부 | `reports/.../main/day07/saved_video/<설정>/` |
| `day07_team_summarize_runs.py` | 설정별 Warm-up 통과 · UNKNOWN 비율 · Stable 흔들림 · FPS 비교표 (교재 43번) | `reports/.../main/day07/runtime_comparison.csv` · `runtime_runs.csv` |
| `day07_team_ip_webcam.py` | IP Webcam 실시간. 세로 확인 · 가로면 회전 · `--targets` 6개 · `--protocol with_c01 / no_c01` (교재 44번) | `reports/.../main/day07/ip_webcam/` (로그 + `_meta.json`) |
| `day07_team_analyze_log.py` | Target 구간만 모아 동작별 "N회 중 _회" · UNKNOWN · Raw 분포 · 혼동 쌍 (교재 45번) | `reports/.../main/day07/ip_webcam_summary.csv` |

- 설정은 `configs/team01_day07_realtime.json`에서만 바꿔요 (코드에 숫자 직접 적지 않기). Confidence는 0.45로 시작 — 16클래스라 확률이 나뉨
- 대표 인물은 학습 데이터와 같은 **가장 큰 BBox (D6)** — 교재 `frame_feature_from_pose()`(BBox conf 최고) 대신 `team_frame_feature_from_pose()`
- S05 D01 T04 · T05 · D03 T01은 4초 미만이라 기본 Warm-up(5초 × 0.8)에서 Prediction 0번 → `--first-takes`는 D03 T01 대신 T02를 씀
- 한 번에 하나만 비교: ① Confidence 0.35 / 0.45 / 0.55 → ② Window 5 / 3초 → ③ Buffer 3 / 5 / 7
- 금지: S02(Test) 영상 사용 · 실시간 결과로 재학습 · MP4 / joblib / Frame 로그 커밋. 요약표(`day07/*.csv`)만 커밋
- `stable_match_ratio` · "N회 중 _회"는 Accuracy가 아님 (사람이 누른 Target 기준 참고값)

---

## 설계 문서 · 기록

| 경로 | 내용 |
| --- | --- |
| `docs/referee/00_배포노트.md` | 배포별 변경 요약 · 회의 안건 · 버전 규칙 |
| `docs/referee/01_관절_매핑_명세.md` | VICON → COCO 17(+손등 2) 관절 매핑, 결정 D1~D6 |
| `docs/referee/02_라벨링_가이드.md` | 신호 클래스 · 구간 라벨 · 교차 검수 규칙 |
| `docs/referee/03_웹캠_촬영_계획서.md` | 웹캠 촬영 계획 |
| `docs/referee/04_특징_모델_설계서.md` | Feature · 모델 설계, 실험 E-01~E-06 |
| `docs/referee/05_시스템_아키텍처.md` | 실시간 시스템 (웹캠 → FastAPI → 화면), API, 디렉터리 |
| `docs/referee/doc_versions.csv` | 설계 문서 버전 기록 |
| `docs/referee/team01_decision_log.md` | 1조 결정 기록 (A1~ · B1~ · C1~ · D1~) |
| `docs/rgb_video_capture_log.md` | RGB 영상 촬영 기록 |
| `scripts/vicon_00_make_take_log.py` · `vicon_01_extract_3d_json.py` · `reports/vicon/` | VICON 촬영 파일 정리, GLB → 3D JSON |
| `reports/team_projects/team01/hand_detect_report.md` | 손 검출 실험 (D1 재검토 근거) |
| `daily_csv/` | 날짜별 작업 기록 + `index.csv`. 쓰는 법은 `daily_csv/README.md` |

## 작업 방식

- 각자 브랜치에서 작업 → PR → `main` merge (main에 바로 push 하지 않기)
- 브랜치에 main 변경 받기: `git fetch origin && git merge origin/main`
- 커밋 메시지는 한글 (`feat:` · `fix:` · `docs:` · `test:` · `data:` · `label:` · `chore:`)
- 레포에는 S 번호만 사용 (실명 · 얼굴 나온 영상/사진 금지)