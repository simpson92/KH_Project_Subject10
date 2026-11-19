# Day 06 Mini Challenge

## 1. 사용한 데이터

- Train Subject: S01
- Validation Subject: S02
- Train Clip 수: 6
- Validation Clip 수: 6
- Action 수: 6 (A001~A006)
- 5일차 본수업 Approved Dataset 18개 사용 (S03 Test는 사용하지 않음)

## 2. Sequence Dataset

- Sequence Steps: 30
- Frame Feature 수: 38 (정규화 좌표 34 + 관절각도 4)
- Sequence Feature 수: 1,140 (30 × 38)
- Train CSV 행 수: 6
- Validation CSV 행 수: 6
- CSV 열 수: 1,148 (Sequence Feature 1,140 + 메타 8)

## 3. Random Forest 학습

- Train Clips: 6
- Feature Count: 1140
- Classes: bend_return, drink_return, leg_raise_lower, sit_stand, walk_turn_walk, wave
- 생성된 Model: `artifacts/day06/rf_action_baseline.joblib`

## 4. Baseline Validation

- Accuracy: 0.5000
- Macro F1: 0.4444
- 정답 Clip 수: 3 (bend_return, leg_raise_lower, sit_stand)
- 오답 Clip 수: 3
- 교재 결과(0.6667 / 0.5556)와 다른 이유: 5일차에 추출한 Keypoint 값이 교재와 조금 달라 Feature와 Tree가 달라짐. Validation 6개라 Clip 1개 차이 = 0.1667

## 5. 오분류

| Clip | 실제 Action | 예측 Action | Confidence |
|---|---|---|---|
| S02_A003_T01 | walk_turn_walk | leg_raise_lower | 0.2133 |
| S02_A004_T01 | drink_return | wave | 0.2267 |
| S02_A006_T01 | wave | drink_return | 0.2200 |

- 원본 MP4에서 확인한 내용: `S02_A003_T01.mp4`(walk → leg)를 확인했고 걷기·회전 동작과 화면에 특별한 문제는 없었다. 데이터(촬영) 문제보다는 Feature 표현이나 작은 Train 데이터 쪽 원인일 가능성이 있다.
- drink_return과 wave는 서로 바뀌어 예측되었다. 둘 다 팔을 얼굴·머리 쪽으로 올리는 동작이라 숫자 패턴이 비슷했던 것으로 보인다.

## 6. 본수업 Feature 비교 결과

- Static VS Sequence: Static(38) 0.6667 / 0.5556, Sequence(1,140) 0.5000 / 0.4444 → 내 실행에서는 Static이 Clip 1개 더 맞힘
- Coordinates + Angles VS Coordinates Only: 좌표+각도(1,140) 0.5000 / 0.4444, 좌표만(1,020) 1.0000 / 1.0000
- 결과를 일반화하면 안 되는 이유: Validation이 한 사람(S02)의 6개 Clip뿐이라 Clip 1개만 달라져도 Accuracy가 0.1667씩 바뀐다. 다른 사람이나 더 많은 데이터에서는 결과가 달라질 수 있다.

## 7. 한 변수 실험

- 바꾼 조건: `sequence_steps` (나머지 n_estimators 300, max_depth null, random_state 42는 그대로)
- 변경 전: 30 Step (Feature 1,140)
- 변경 후: 15 Step (Feature 570)
- 가설: 15 Step으로 줄이면 Feature가 570개로 줄어서, 6개짜리 작은 데이터에서는 Validation 결과가 달라질 것 같다.

## 8. 실험 결과

- Accuracy: 0.6667 (Baseline 0.5000)
- Macro F1: 0.5833 (Baseline 0.4444)
- 오분류 Clip 수: 2 (Baseline 3)
- Baseline보다 좋아짐 / 비슷함 / 나빠짐: 좋아짐 (Clip 1개 차이)
- 예측이 바뀐 Clip:
  - S02_A006_T01 wave: drink_return(오답) → wave(정답)
  - S02_A004_T01 drink_return: wave(오답) → leg_raise_lower(오답, 틀린 답만 바뀜)
  - S02_A003_T01 walk_turn_walk: 두 조건 모두 leg_raise_lower(오답)
  - 맞다가 틀린 Clip은 없음
- 이유에 대한 생각: Step을 줄이자 drink와 wave가 서로 바뀌던 것 중 wave는 맞혔다. 하지만 Clip 1개 차이이고 Confidence도 0.20~0.36으로 낮아서 "15 Step이 더 좋다"고 일반화하지 않는다. walk → leg는 Step 수와 상관없이 남아 있어 다른 원인을 더 확인해야 한다.
- 실험 후 설정을 30 Step으로 되돌리고 34 → 35 → 36을 다시 실행해 Baseline(0.5000 / 0.4444)을 복구했다.

## 9. Test를 사용하지 않은 이유

- S03 Test는 조건을 모두 정한 뒤 마지막에 한 번만 확인하는 데이터다. Challenge에서 Test 결과를 보고 조건을 고르면 S03이 더 이상 "처음 보는 사람"이 아니게 되어 공정한 최종 평가가 되지 않는다.

## 10. 오늘 다시 이해한 핵심

- 길이가 다른 Clip도 Sequence Resampling으로 같은 30 Step(1,140개 숫자)으로 맞추면 한 행의 학습 데이터가 된다.
- 비교는 Validation(S02)으로만, 한 번에 한 가지 조건만 바꿔야 결과 차이의 원인을 알 수 있다.
- 6개짜리 작은 데이터에서는 점수 차이보다 어떤 Clip이 어떻게 바뀌었는지를 보는 것이 더 중요하다.
- 같은 코드여도 Keypoint 추출 결과가 조금 다르면 결과가 달라질 수 있다.
