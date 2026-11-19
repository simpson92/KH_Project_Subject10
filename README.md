# Subject 10 - Pose Estimation & Action Analysis

교과 10 자세추정 이론·실습 **개인 레포**입니다.
사전학습된 YOLO Pose(`yolo11n-pose.pt`)로 사람의 관절을 찾고, 관절 좌표를 숫자 Feature로 바꾸어 동작을 인식하는 전체 과정을 일차별로 실습합니다.

> 1~6일차는 조 공용 레포 `KH_Project_Subject10`에서 진행한 뒤, 2026-11-19에 교과10 실습 파일과 커밋 기록만 이 레포로 옮겼습니다.
> 1조 농구 심판 수신호 프로젝트는 `KH_Project_Subject10`에서 계속합니다.

```text
Frame → Pose → Keypoint → Feature → Sequence → Action Recognition → Realtime → Tracking
```

## 개발환경

| 항목 | 값 (`reports/day01/environment.txt`) |
| --- | --- |
| OS | Windows + WSL2 Ubuntu |
| Python | 3.12.16 (`uv` 가상환경) |
| PyTorch | 2.15.0+cu132 (CUDA 13.2) |
| Ultralytics | 8.4.203 |
| OpenCV | 5.0.0 |
| GPU | NVIDIA GeForce RTX 3060 |

```bash
cd ~/ai_vision/KH_Project_Subject1010
uv sync
source .venv/bin/activate
python scripts/00_check_environment.py
uv run pytest -q
```

`data/`, 이미지·영상, 모델 가중치(`*.pt`), 학습 모델(`*.joblib`), 실행 결과 CSV는 Git에 올리지 않습니다 (`.gitignore`).

## 폴더 구조

```text
KH_Project_Subject1010/
├── configs/
│   ├── project.json               # 모델명 · 이미지 크기 · Confidence 기준 · 장치   (1일차)
│   ├── action_classes.csv         # 6개 Action ID · Label · MP4 이름                (4일차)
│   ├── day05_subject_split.json   # Train S01 · Validation S02 · Test S03           (5일차)
│   └── day06_baseline.json        # 30 Step · Random Forest 설정                    (6일차)
├── src/
│   ├── common.py                  # 설정 읽기 · GPU/CPU 선택      (1일차)
│   ├── pose_utils.py              # COCO 17 이름 · Skeleton · 품질 판정 (2일차)
│   ├── feature_utils.py           # 관절각도 · 정규화 · 좌우 반전   (3일차)
│   └── sequence_utils.py          # 38-D Frame Feature · Resampling (6일차)
├── scripts/                       # 00~40 수업 순서대로 실행
├── tests/                         # 단위 테스트 (YOLO·GPU 없이 실행)
├── reports/                       # 일차별 결과 (CSV·이미지는 Git 제외)
└── artifacts/day06/               # 6일차 모델 정보 json (joblib은 Git 제외)
```

### `data/` 폴더 구조 (Git 제외 — 각자 PC에 직접 넣기)

```text
data/
├── day01/
│   ├── coco8-pose.zip · coco8_pose/        # 강사님 제공 (압축 해제)
│   ├── video/solution_ci_pose_demo.mp4     # 강사님 제공
│   └── ip_webcam/                          # 직접 촬영 (front · arms_up · far.jpg)
├── day02/
│   ├── images/                             # 강사님 제공 6장 (front, arms_up, side …)
│   └── videos/pose_quality_demo.mp4        # day01 영상 복사
├── day03/
│   ├── images/                             # 강사님 제공 8장 (arms_up_person_a …)
│   ├── videos/feature_demo.mp4             # day02 영상 복사
│   └── team/                               # Mini Challenge용 Frame 4장
├── day04/videos/                           # 강사님 제공 6개 (bend_return.mp4 …)
├── day05/
│   ├── videos/S01~S03/S0x_A00x_T01.mp4     # 강사님 제공 18개
│   ├── keypoints/ · manifests/ · qa/ · splits/   # 26~30번 실행 결과
└── day06/sequences/                        # 34번 실행 결과
```

- 강사님 제공 파일은 수업 자료에서 받아요.
- `yolo11n-pose.pt`는 프로젝트 최상위 폴더에 둬요 (처음 실행할 때 자동 다운로드, Git 제외).
- 영상 · 사진에는 얼굴이 있어서 레포나 외부에 올리지 않아요.

---

## Day 01 - Pose 전체 구조·개발환경·실제 데이터 입력

### 사용한 입력

- COCO8-Pose
- Ultralytics Pose Demo MP4 (`solution_ci_pose_demo.mp4`)
- 스마트폰 IP Webcam 캡처 이미지 (정면 · 양팔 들기 · 먼 거리)

### 학습한 내용

- Pose Estimation과 Action Recognition의 차이
- WSL 프로젝트와 Python 가상환경(`uv`) 구성
- OpenCV / PyTorch / Ultralytics, GPU / CUDA 확인
- COCO8-Pose 이미지 입력과 YOLO Pose 추론
- Person Bounding Box · COCO 17 Keypoint · Skeleton
- 실제 MP4를 Frame 단위로 읽고 Frame마다 Pose 적용
- IP Webcam 연결 및 Frame 저장, 직접 촬영 이미지 Pose 추론

### Day 01 Pipeline

```text
JPG / MP4 / IP Webcam
        ↓
Frame
        ↓
YOLO Pose
        ↓
Person Bounding Box
        ↓
COCO 17 Keypoint
        ↓
Skeleton
```

### 주요 실행 파일

```text
scripts/
├── 00_check_environment.py
├── 01_image_input.py
├── 02_video_input.py
├── 03_pose_image.py
├── 04_pose_video.py
└── 05_ip_webcam_capture.py
```

### 주요 결과

- `reports/day01/environment.txt` — 실행환경 점검 결과
- `reports/day01/pose_000000000036.txt` — COCO8-Pose 첫 이미지 Keypoint
- `reports/day01/pose_front.txt` · `pose_arms_up.txt` · `pose_far.txt` — IP Webcam 촬영 이미지 Keypoint
- `reports/day01/pose_demo_output.mp4` — MP4 Frame별 Pose 결과 (Git 제외)

### Mini Challenge

- 미진행

---

## Day 02 - Keypoint 좌표와 Pose 품질 분석

### 사용한 입력

- 강사가 준비한 조건별 이미지 6장
  - `front.jpg` · `arms_up.jpg` · `side.jpg` · `partial_body.jpg` · `occlusion.jpg` · `distance_far.jpg`
- 1일차에서 사용한 실제 MP4 (`pose_quality_demo.mp4`로 복사)

### 학습한 내용

- Person BBox Confidence와 Keypoint Confidence 구분
- COCO 17 Keypoint Pixel 좌표와 Normalized 좌표(이미지 크기 기준, 0~1)
- Keypoint Confidence와 Threshold, Confidence 기준에 따른 Skeleton 표시
- 여러 이미지의 Pose 품질 CSV 비교 (BBox Area Ratio, 평균 Keypoint Confidence, LOW 관절)
- MP4의 Frame별 Pose 품질 분석, 불안정한 Frame을 숫자로 찾고 실제 영상과 연결

### 실습에서 확인한 결과

- `front.jpg` — 평균 Keypoint Confidence `0.9754`, LOW 관절 `0` → 정면 기준 이미지에서 안정적
- `side.jpg` — 평균 Keypoint Confidence `0.7541`, LOW 관절 `4` → 6장 중 가장 불안정
- `front.jpg` vs `distance_far.jpg` — 화면 속 사람 크기는 약 절반(`0.2996 → 0.1489`)으로 줄었지만, 평균 Keypoint Confidence는 `0.9754 → 0.9710`, LOW 관절은 둘 다 `0` → 이번 거리 범위에서는 품질 유지
- MP4 — 60개 Frame 모두 사람 검출. 약 `0.80초`(Frame 25~26)에서 바닥 자세·얼굴 방향 때문에 눈·귀 Keypoint Confidence가 낮아짐

### Day 02 Pipeline

```text
이미지 · MP4
        ↓
YOLO Pose
        ↓
Person BBox + COCO 17 Keypoint
        ↓
Pixel / Normalized 좌표
        ↓
Keypoint Confidence → Threshold
        ↓
조건별 · Frame별 Pose 품질 비교
        ↓
숫자와 실제 이미지·영상 연결
```

### 주요 실행 파일

```text
scripts/
├── 06_check_day02_data.py
├── 07_pose_image_detail.py
├── 08_draw_skeleton_quality.py
├── 09_compare_pose_images.py
└── 10_pose_video_quality.py

src/
└── pose_utils.py
```

### 주요 결과 파일 (Git 제외)

```text
reports/day02/tables/
├── front_keypoints.csv
├── image_quality_summary.csv
├── pose_quality_demo_frame_quality.csv
└── pose_quality_demo_keypoints.csv
```

### Mini Challenge

- 미진행 (`reports/day02/challenge/challenge_notes.md` 없음)

---

## Day 03 - Joint Angle, Normalization & Frame Feature

### 오늘 배운 내용

- 세 Keypoint를 이용한 관절각도 계산 (가운데 관절 기준 두 벡터)
- Hip Center 기준 몸 중심 이동, Shoulder Width 기준 크기 정규화
- Near / Far, Person A / Person B 정규화 전·후 비교
- COCO Left / Right 기준과 Keypoint 좌우 반전
- 34개 Normalized 좌표 + 4개 관절각도(양 팔꿈치 · 양 무릎) = 38-D Frame Feature
- 동작을 설명하는 Feature 선택, 규칙 기반 판정(Arms Up)의 한계
- 영상의 각 Frame에서 38-D Feature를 만들고 시간에 따른 변화 확인

### Day 03 핵심 흐름

```text
YOLO Pose
        ↓
COCO 17 Keypoint + Confidence
        ↓
관절각도
        ↓
Hip Center → Shoulder Width Normalization
        ↓
Normalized Keypoint 34개 + Joint Angle 4개
        ↓
38-D Frame Feature
        ↓
영상의 각 Frame에 반복
        ↓
시간에 따른 Feature 변화 → Sequence의 재료
```

### 오늘 추가한 주요 코드

```text
src/
└── feature_utils.py

scripts/
├── 11_check_day03_data.py
├── 12_joint_angle.py
├── 13_normalize_pose.py
├── 14_compare_normalization.py
├── 15_flip_keypoints.py
├── 16_make_frame_feature.py
├── 17_arms_up_rule.py
└── 18_video_frame_features.py
```

> `18_video_frame_features.py`: 교재 26절의 18번 코드 칸에 17번 코드가 그대로 들어 있어,
> 교재 설명(Frame마다 38-D, `frame_id · time_sec · valid_pose · mean_keypoint_conf · low_keypoint_count`)대로 다시 작성했습니다.
> 교재 데모 영상에서 `Total frames 60 / Valid pose frames 60 / Feature count 38`, Frame 1 왼팔꿈치 약 140°로 교재 결과와 일치합니다.

### 오늘 이해한 핵심

한 장의 Pose에서 얻은 Keypoint 좌표를 그대로 사용하는 것이 아니라,
몸의 위치와 크기 차이를 줄인 Normalized 좌표와 관절각도를 이용하여
한 Frame의 자세를 38개의 숫자로 표현했습니다.

또한 동작을 구분할 때는 38개의 숫자를 무조건 모두 보는 것이 아니라,
손목과 어깨의 상대 높이처럼 해당 동작을 설명하는 Feature를 선택하는 것이 중요하다는 것을 확인했습니다.

영상에서는 같은 38-D Feature를 Frame마다 반복해서 만들고,
Feature가 시간에 따라 변하는 모습을 확인했습니다.

### Mini Challenge — 1조 농구 심판 수신호 (U파울 vs 푸싱)

조원들이 휴대폰으로 촬영한 수신호 영상(U파울 · 푸싱) 중 2명 × 2동작 영상 4개로
교재의 관절각도 · 정규화 · 38-D · Feature 선택 · 영상 변화 과정을 다시 수행했습니다.
자세한 내용: [`reports/day03/challenge/feature_design.md`](reports/day03/challenge/feature_design.md)

- 정규화: 정규화 후 동작 차이(`0.554` · `0.872`)가 사람 차이(`0.113`)보다 5~8배 큼.
  단, 어깨 폭이 자세에 따라 변함 (같은 사람 `172.57 → 235.84 px`)
- 선택한 Feature: `left_wrist_y`, `left_elbow_angle`, `left_wrist_y − nose_y`
  (U파울은 손목이 머리 위라 음수, 푸싱은 양수)
- 무릎 각도는 서서 하는 두 신호를 구분하지 못함
- 영상의 시작·끝(팔을 내린 구간)은 두 동작이 같아, Frame 한 장이 아니라 시간 순서가 필요함
- 17번 1조 규칙 (`scripts/17_referee_rule.py`): U파울 = 양 손목이 코보다 위 + 양 손목 거리 < 1.0 어깨 폭 → 사진 4장 · 영상 4개 모두 예상대로 판정 (푸싱 오판 0)

### 추가 작업 — 테스트 · D6 대표 인물

| 파일 | 내용 | 실행 |
| --- | --- | --- |
| `tests/` | `feature_utils`(각도·정규화·좌우 반전·Near/Far 불변성), 17번 U파울 규칙, D6 대표 인물 단위 테스트. 가짜 Pose 좌표로 검사해서 YOLO·GPU·`data/` 없이 실행됨 | `uv run pytest -q` |
| `src/pose_utils.py` `select_main_person()` | 여러 사람이 보이면 대표 인물 선택 (`area` = 가장 큰 BBox · `conf` · `first`) | 18번 `--main-person area` (기본) |
| `scripts/18_video_frame_features.py` | 대표 인물 선택 반영. CSV 맨 뒤에 `people` · `person_index` 열 추가 (38-D 열 위치는 그대로). 교재 결과 재현은 `--main-person first` | |

### 다음 수업

4일차에는 여러 Frame의 Feature를 시간 순서로 묶어
하나의 동작을 표현하는 **Sequence**로 확장합니다.

---

## Day 04 - Action Clip · Sequence 데이터 만들기

수업 실습 `scripts/19~24` — 동작 영상 6개를 Action Clip으로 정리하고, Frame마다 38-D Feature를 만들어 시간 순서의 Sequence CSV로 저장했습니다.

### Input

- 강사님 제공 MP4 6개 (640×480 · 25 FPS · 총 824 Frame)
- bend_return 85 · leg_raise_lower 125 · walk_turn_walk 266 · drink_return 120 · sit_stand 102 · wave 126 Frame

### Day 04 Pipeline

```text
MP4 6개 + configs/action_classes.csv (A001~A006 정답표)
        ↓
영상 정보 · Action 정답표 검사
        ↓
시작 · 가운데 · 끝 Keyframe 확인
        ↓
Frame마다 YOLO Pose → 38-D Frame Feature
        ↓
Action별 Sequence CSV (한 행 = Frame 1개)
        ↓
Frame 수 · 인접 Frame 변화량 요약
```

### 주요 실행 파일

| 파일 | 내용 | 출력 |
| --- | --- | --- |
| `configs/action_classes.csv` | A001~A006 · action_label · MP4 이름 (직접 작성) | — |
| `scripts/19_check_day04_data.py` | MP4 6개 해상도 · FPS · Frame 수 | `reports/day04/tables/video_metadata.csv` |
| `scripts/20_check_action_classes.py` | 정답표와 실제 MP4 일치 검사 | — |
| `scripts/21_extract_clip_keyframes.py` | 영상별 시작 · 가운데 · 끝 Frame 저장 | `reports/day04/keyframes/*.jpg` |
| `scripts/22_build_action_sequences.py` | Frame별 38-D Sequence CSV | `reports/day04/sequences/{label}_sequence.csv` |
| `scripts/23_summarize_action_sequences.py` | Action별 Frame 수 · 인접 Frame 평균 변화량 | `reports/day04/tables/sequence_summary.csv` |
| `scripts/24_inspect_sequence_stages.py` | Sequence CSV 한 개의 구간별 값 확인 | 터미널 출력 |

### 오늘 이해한 핵심

- 영상 정답(`action_label`)은 YOLO가 예측하는 것이 아니라 미리 작성한 정답표(`action_classes.csv`)에서 붙입니다.
- 같은 38-D Feature라도 Frame 하나가 아니라 시간 순서로 묶어야 동작을 표현할 수 있습니다.
- walk_turn_walk의 인접 Frame 변화량이 가장 컸음 → 몸을 돌릴 때 어깨 폭이 작아져 정규화 값이 크게 흔들린 것으로 추정

### Mini Challenge

- 미진행

---

## Day 05 - Keypoint Dataset · Human QA · Subject-wise Split

수업 실습 `scripts/25~32` — 3명 × 6동작 MP4 18개에서 Keypoint 원본을 뽑고, 자동 QA → 사람 검토 → 승인 → 사람 단위 분할까지 진행했습니다.

### Input

- 강사님 제공 MP4 18개 (S01 · S02 · S03 × A001~A006 × T01, 640×480 · 25 FPS · 총 2,433 Frame)
- S01 = 4일차 영상 6개와 같은 영상

### Day 05 Pipeline

```text
MP4 18개 + project.json + action_classes.csv
        ↓
26 Keypoint CSV 18개 (한 행 = Frame 1개, 95열 = 메타 10 + 17관절 × 5)
        ↓
27 자동 QA (AUTO_PASS / CHECK)
        ↓
사람 Human QA (PASS / RESHOOT / EXCLUDE)
        ↓
28 검사 → 29 Approved Manifest (PASS만)
        ↓
30 Subject-wise Split (S01 Train · S02 Validation · S03 Test) → 31 Leakage 검사
        ↓
32 Frame 무작위 분할 비교 (누수 확인)
```

### 주요 실행 파일

| 파일 | 내용 | 출력 |
| --- | --- | --- |
| `scripts/25_check_day05_data.py` | 파일명 규칙 · 18개 · 해상도 · FPS | `reports/day05/tables/video_inventory.csv` |
| `scripts/26_extract_keypoint_dataset.py` | 모든 Frame의 Keypoint 원본 (`x_px · y_px · x_norm · y_norm · conf`) | `data/day05/keypoints/S0x/{label}/*.csv` · `manifests/keypoint_dataset_manifest.csv` |
| `scripts/27_keypoint_qa.py` | Pose 비율 · 1인 비율 · 평균 conf · 동작별 중요 관절 | `data/day05/qa/keypoint_clip_qa.csv` · `keypoint_clip_review.csv` |
| `scripts/28_validate_human_review.py` | 사람 판정이 다 채워졌는지 검사 | — |
| `scripts/29_build_approved_manifest.py` | PASS만 모은 승인 목록 | `data/day05/manifests/approved_clip_manifest.csv` |
| `configs/day05_subject_split.json` | Train S01 · Validation S02 · Test S03 (직접 작성) | — |
| `scripts/30_build_subject_split.py` | 사람 단위 분할 | `data/day05/splits/{train,val,test}.csv` |
| `scripts/31_check_split_leakage.py` | Subject · Clip 겹침, Action Coverage | — |
| `scripts/32_demo_frame_split_leakage.py` | Frame 무작위 분할의 누수 확인 | 터미널 출력 |

### 주요 결과 (내 실행, 2026-11-18)

- 자동 QA: AUTO_PASS 12 · CHECK 6 (S02 A001~A006 전부, 주로 여러 사람으로 검출돼 1인 검출 비율이 낮음)
- Human QA: 원본 MP4 확인 후 18개 모두 PASS → Approved 18 → Split 6 / 6 / 6 · `Leakage check: OK`
- Frame 무작위 분할(32번)은 Train · Test에 같은 사람 · 같은 Clip이 섞임 → 사람 단위 분할이 필요한 근거
- 교재 28 · 29번 설명(AUTO_PASS 자동 승인)과 실제 코드(PASS만 승인)가 달라, 18개 모두 사람이 PASS를 입력함

### Mini Challenge

- 미진행

---

## Day 06 - Action Recognition Baseline 성능평가

수업 실습 `scripts/33~40` — 5일차 Approved 18 clip으로 첫 Action Recognition 모델(Random Forest)을 학습하고 평가했습니다.

### Input

- 5일차 Approved Keypoint Dataset 18 clip (3 Subjects × 6 Actions × 1 Take)
- S01 → Train · S02 → Validation · S03 → Test
- Action: bend_return · leg_raise_lower · walk_turn_walk · drink_return · sit_stand · wave

### Day 06 Pipeline

```
Keypoint CSV ─▶ 38-D Frame Feature ─▶ Sequence Resampling (frame_id 기준 30 Step) ─▶ 1,140-D (Clip 1개 = 1행)
   ─▶ Random Forest (S01 Train) ─▶ S02 Validation ─▶ 오분류 추적 ─▶ Feature 비교 ─▶ Mini Challenge ─▶ S03 Test (마지막 1번)
```

### 주요 실행 파일

| 파일 | 내용 | 출력 |
| --- | --- | --- |
| `configs/day06_baseline.json` | 30 Step · 최소 유효 Frame 10 · Tree 300 · max_depth null · seed 42 | — |
| `src/sequence_utils.py` | Frame → 38-D, frame_id 기준 Resampling, 펼치기, 열 이름 | — |
| `scripts/33_check_day06_inputs.py` | 5일차 Split · Keypoint CSV · Leakage 검사 | `reports/day06/tables/day06_input_check.csv` |
| `scripts/34_build_sequence_dataset.py` | 18 clip → Sequence Dataset (1,148열 = 메타 8 + 1,140) | `data/day06/sequences/{train,val,test}.csv` |
| `scripts/35_train_rf_baseline.py` | S01 6행으로 Random Forest 학습 | `artifacts/day06/rf_action_baseline.joblib` · `rf_feature_columns.json` · `rf_training_info.json` |
| `scripts/36_evaluate_validation.py` | S02 Accuracy · Macro F1 · Confusion Matrix | `reports/day06/metrics/validation_*` |
| `scripts/37_analyze_misclassified.py` | 오분류 clip에 원본 MP4 · Keypoint 경로 연결 | `reports/day06/misclassified/` |
| `scripts/38_compare_static_vs_sequence.py` | 가운데 Frame 1개(38) vs 30 Step(1,140) | `reports/day06/experiments/static_vs_sequence.csv` |
| `scripts/39_compare_feature_sets.py` | 좌표+각도(1,140) vs 좌표만(1,020) | `reports/day06/experiments/feature_set_comparison.csv` |
| `scripts/40_evaluate_test.py` | 최종 조건으로 S03 Test 1번 | `reports/day06/metrics/test_*` |

### 주요 결과 (내 실행, 2026-11-19)

| 실험 | Feature | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| Baseline Validation (S02) | 1,140 | 0.5000 | 0.4444 |
| Static (가운데 Frame) | 38 | 0.6667 | 0.5556 |
| Coordinates Only | 1,020 | 1.0000 | 1.0000 |
| Mini Challenge 15 Step | 570 | 0.6667 | 0.5833 |
| **Final Test (S03)** | 1,140 | **1.0000** | **1.0000** |

- Baseline 오분류: walk_turn_walk → leg_raise_lower · drink_return → wave · wave → drink_return
- 교재 결과(0.6667 / 0.5556)와 다른 이유: 5일차 Keypoint 추출 값 차이. Validation 6개라 clip 1개 = 0.1667
- 최종 조건은 원래 Baseline(30 Step · 좌표+각도) 유지. Test 결과를 보고 조건을 다시 바꾸지 않음
- Confidence 0.20~0.37 — 아주 작은 교육용 Dataset이라 이 숫자를 현업 성능으로 해석하지 않음
- 정리: [`reports/day06/day06_summary.md`](reports/day06/day06_summary.md)

### Mini Challenge

`sequence_steps` 하나만 30 → 15로 바꿔 S02 Validation 비교 → 0.5000 / 0.4444 → 0.6667 / 0.5833 (wave 정답으로 바뀜, 맞다가 틀린 clip 없음).
clip 1개 차이라 일반화하지 않고, 실험 후 30 Step으로 되돌림.
자세한 내용: [`reports/day06/challenge/mini_challenge_result.md`](reports/day06/challenge/mini_challenge_result.md)

### Git 제외

`data/day06/` · `*.joblib` · `reports/day06/` 자동 생성 csv · Confusion Matrix png (`artifacts/`는 json 2개만 `git add -f`)

### 다음 수업

7일차에는 실시간 Frame을 계속 모아 Sliding Window를 만들고 실시간 Action Recognition으로 연결합니다.