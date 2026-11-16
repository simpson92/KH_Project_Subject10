# Subject 10 - Pose Estimation & Action Analysis

교과 10 자세추정 이론·실습 프로젝트입니다.
사전학습된 YOLO Pose(`yolo11n-pose.pt`)로 사람의 관절을 찾고, 관절 좌표를 숫자 Feature로 바꾸어 동작을 인식하는 전체 과정을 일차별로 실습합니다.

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
cd ~/ai_vision/subject10_pose_action
uv sync
source .venv/bin/activate
python scripts/00_check_environment.py
```

`data/`, 이미지·영상, 모델 가중치(`*.pt`), 실행 결과 CSV는 Git에 올리지 않습니다 (`.gitignore`).

## 폴더 구조

```text
subject10_pose_action/
├── configs/project.json        # 모델명 · 이미지 크기 · Confidence 기준 · 장치
├── src/
│   ├── common.py               # 설정 읽기 · GPU/CPU 선택      (1일차)
│   ├── pose_utils.py           # COCO 17 이름 · Skeleton · 품질 판정 (2일차)
│   └── feature_utils.py        # 관절각도 · 정규화 · 좌우 반전   (3일차)
├── scripts/                    # 00~18 수업 순서대로 실행
├── reports/                    # 일차별 결과 (CSV·이미지는 Git 제외)
└── docs/referee/               # 1조 프로젝트 설계 문서
```

### `data/` 폴더 구조 (Git 제외 — 각자 PC에 직접 넣기)

```text
data/
├── day01/
│   ├── coco8-pose.zip · coco8_pose/        # 강사님 제공 (압축 해제)
│   ├── video/solution_ci_pose_demo.mp4     # 강사님 제공
│   └── ip_webcam/                          # 각자 촬영 (front · arms_up · far.jpg)
├── day02/
│   ├── images/                             # 강사님 제공 6장 (front, arms_up, side …)
│   └── videos/pose_quality_demo.mp4        # day01 영상 복사
├── day03/
│   ├── images/                             # 강사님 제공 8장 (arms_up_person_a …)
│   ├── videos/feature_demo.mp4             # day02 영상 복사
│   └── team/                               # 스모크 영상 가운데 Frame 4장 (Mini Challenge)
├── referee/smoke/
│   ├── raw/ufoul/  S01~S05_ufoul_T01~T10.mp4     # 조원 휴대폰 촬영 50개
│   ├── raw/pushing/ S01~S05_pushing_T01~T10.mp4  # 조원 휴대폰 촬영 50개
│   └── subjects.csv                        # S번호 ↔ 이름 (실명 → 공유 금지)
├── raw/vicon/                              # Motion_Visol.zip 압축 해제 (FBX 61개)
│   └── glb/                                # Glb.zip 압축 해제 (GLB 60개)
├── take_log.csv                            # vicon_00 실행 결과
└── vicon_3d/                               # vicon_01 실행 결과 (3D JSON 50개)
```

- 강사님 제공 파일은 수업 자료에서, 스모크 영상 · VICON zip은 조 공유 폴더에서 받아요.
- `take_log.csv`, `vicon_3d/`는 받는 게 아니라 스크립트를 실행하면 생겨요.
- 영상 · 사진에는 조원 얼굴이 있어서 레포나 외부에 올리지 않아요.

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

### 다음 수업

4일차에는 여러 Frame의 Feature를 시간 순서로 묶어
하나의 동작을 표현하는 **Sequence**로 확장합니다.

---

## 1조 프로젝트 — 농구 심판 수신호 인식 → 전광판

수업 Pipeline을 농구 심판의 파울 수신호 15개 인식에 적용하는 조 프로젝트입니다.

| 경로 | 내용 |
| --- | --- |
| `docs/referee/01_관절_매핑_명세.md` | VICON → COCO 17(+손등 2) 관절 매핑, 결정 사항 D1~D6 |
| `docs/referee/02_라벨링_가이드.md` | 신호 클래스(A01~A15, C01, N00) 라벨링 규칙 |
| `docs/referee/03_웹캠_촬영_계획서.md` | 웹캠 촬영 계획 |
| `docs/referee/04_특징_모델_설계서.md` | Feature · 모델 설계 |
| `scripts/vicon_00_make_take_log.py` · `vicon_01_extract_3d_json.py` | VICON 촬영 파일 정리, GLB → 3D JSON 추출 |
| `scripts/referee_00~04_*.py` · `reports/referee/smoke/` | 간이 검증용 스모크 테스트 (본 Pipeline 아님) |