# 심판 수신호 스모크 테스트 (U파울 vs 푸싱)

> ⚠️ **간이 검증용 실험이야.** 본 파이프라인이 아니야.
> 교재(교과10) 코드 구조나 팀 문서(DOC-01~04) 규칙을 따르지 않았어. 결과를 다시 만들 수 있게 코드는 고치지 않고 그대로 둬.

## 1. 목적

직접 찍은 휴대폰 영상으로 아래 두 가지가 실제로 되는지 빠르게 확인하려고 했어.

1. YOLO Pose가 심판 수신호 동작의 키포인트를 안정적으로 잡는가
2. 처음 보는 사람의 영상에도 동작 분류가 되는가

## 2. 데이터

| 항목 | 내용 |
|---|---|
| 동작 | U파울 (`ufoul`, 순서표 3번), 푸싱 (`pushing`, 순서표 5번) |
| 인원 | 5명 (S01~S05) |
| 테이크 | 사람·동작마다 10개 → 총 **100개** |
| 영상 | 휴대폰 세로 1080×1920, 30fps, 3.1~7.7초 |
| 파일명 | `S##_<action>_T##.mp4` (예: `S01_ufoul_T01.mp4`) |
| 위치 | `data/referee/smoke/raw/{ufoul,pushing}/` (git 제외) |

- 원본 파일명을 `referee_00`으로 바꿨어.
- 사람과 번호 대응표는 `data/referee/smoke/subjects.csv`에 있고, 실명이 들어 있어서 git에서 뺐어.

## 3. 실행 순서

```bash
# 0) 파일명 변경 (기본은 dry-run, --apply 붙여야 실제로 바뀜)
uv run python scripts/referee_00_rename_videos.py
uv run python scripts/referee_00_rename_videos.py --apply

# 1) 영상 정보 점검 → video_manifest.csv
uv run python scripts/referee_01_check_videos.py

# 2) 키포인트 추출 + 품질 판정 → keypoints/*.csv, pose_quality.csv
uv run python scripts/referee_02_extract_pose.py

# 3) RF 학습 + 사람 단위 교차검증(LOSO) → loso_predictions.csv, metrics.txt
uv run python scripts/referee_03_train_smoke.py

# 4) 오버레이 영상 (예: 틀린 영상만)
uv run python scripts/referee_04_render_pose.py --only-wrong
```

## 4. 방법 요약

| 단계 | 내용 |
|---|---|
| 포즈 | `yolo11n-pose.pt`, BBox 면적이 가장 큰 사람을 주인공으로 선택 |
| 품질 판정 | 사람 없는 프레임 > 10% 또는 팔(5~10번) 낮은 conf 비율 > 30%이면 CHECK, 아니면 PASS |
| 전처리 | conf < 0.5 → 빈 값 → 시간축 보간 |
| 정규화 | 엉덩이 중심을 원점으로, 어깨너비로 스케일 |
| 프레임 특징 | 좌표 34 + 관절각 4 (좌/우 팔꿈치, 좌/우 어깨) = 38-D |
| 시퀀스 | 클립 전체를 32프레임으로 맞춤 → 펼친 값 + 평균 + 표준편차 = 1292-D |
| 모델 | RandomForest 300, random_state 42 |
| 평가 | LOSO: 한 사람을 빼고 학습, 그 사람으로 테스트 (5번 반복) |

## 5. 결과

### 5.1 키포인트 품질
- **100/100 PASS**, 사람 없는 프레임 0
- 팔 관절 평균 conf **0.985** (푸싱 0.987, U파울 0.984, 최저 영상 0.958)
- 가장 약한 관절은 항상 **손목**
- `S02_pushing_T06`, `T07`: 화면에 2번째 사람이 나왔는데 면적 기준으로 제대로 골라냈어

### 5.2 분류 (LOSO)

| 테스트 인원 | S01 | S02 | S03 | S04 | S05 | 전체 |
|---|---|---|---|---|---|---|
| Accuracy | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | **0.99** |

- Macro F1 **0.99**

| 실제 \ 예측 | pushing | ufoul |
|---|---|---|
| pushing | 50 | 0 |
| ufoul | 1 | 49 |

### 5.3 오분류 1건
- `S04_ufoul_T01` → pushing으로 예측
- 원인: 클립이 7.5초로 길어. 손목이 코보다 위에 있는 프레임이 36%뿐이었어 (다른 영상은 53~69%). 그래서 32프레임으로 맞추면 서 있는 구간이 대부분을 차지해.
- 포즈 문제가 아니라 **클립 단위 라벨의 한계**야. 구간 라벨이나 슬라이딩 윈도우로 해결할 수 있어.

## 6. 의미와 한계

**확인된 것**
- 휴대폰 촬영 조건에서 키포인트가 안정적으로 나와
- 처음 보는 사람에게도 분류가 돼 (사람 사이 일반화)
- 이후 실험의 기준선(baseline)으로 쓸 수 있어

**증명하지 못한 것**
- 클래스가 2개뿐이야. 신호없음(N00)과 헷갈리기 쉬운 신호 쌍이 없어.
- 모두 같은 장소·같은 조건에서 연기한 영상이고, 클립도 잘라서 썼어
- 그래서 **실제 경기 심판 인식 성능을 뜻하지 않아**

## 7. 본 파이프라인과 다른 점

| 항목 | 스모크 테스트 | 본 파이프라인 (팀 문서 기준) |
|---|---|---|
| 파일명/폴더 | `S01_ufoul_T01`, `data/referee/smoke/raw/` | `S##_A##_R##`, `data/raw/webcam/` |
| 라벨 | 영상 하나에 라벨 하나 | `labels.csv` 구간 라벨 |
| 클래스 | 2개 | A01~A15 + C01 + N00 |
| 스케일 | 어깨너비 | 미정 (D2: 몸통 길이 vs 어깨너비) |
| 낮은 conf | 보간 | 미정 (D5: 마지막 유효값 유지) |
| 시퀀스 | 클립 전체를 32프레임으로 | 1.5초 윈도우, 통계 특징 |

## 8. 다시 쓸 만한 부분

- `referee_00`: 원본 파일명 변경 정규식
- `referee_01` / `referee_02`: 영상 점검, 키포인트 품질 PASS/CHECK
- `referee_04`: 키포인트 CSV 기반 오버레이 렌더러 (오분류 영상 확인용)

## 9. 파일

| 파일 | git | 설명 |
|---|---|---|
| `video_manifest.csv` | ✅ | 영상별 fps, 프레임 수, 해상도 |
| `pose_quality.csv` | ✅ | 영상별 키포인트 품질, PASS/CHECK |
| `loso_predictions.csv` | ✅ | 영상별 실제/예측 라벨 |
| `metrics.txt` | ✅ | Accuracy, F1, 혼동행렬 |
| `confusion_matrix.png` | ❌ | `*.png`라서 git 제외 |
| `keypoints/` | ❌ | 영상별 키포인트 CSV (용량 때문에 제외) |
| `videos/` | ❌ | 오버레이 mp4 |