# Day 06 Action Recognition Summary

## 1. Dataset

- Train Subject: S01
- Validation Subject: S02
- Test Subject: S03
- Train Clips: 6
- Validation Clips: 6
- Test Clips: 6
- Action Classes: 6
- Total Clips: 18

## 2. Frame Feature

- Normalized Coordinate: 34
- Joint Angle: 4
- Frame Feature Count: 38

## 3. Original Baseline

- Sequence Steps: 30
- Sequence Feature Count: 1140
- min_valid_frames: 10
- n_estimators: 300
- max_depth: null
- random_state: 42

## 4. Baseline Validation

- Accuracy: 0.5000
- Macro F1: 0.4444
- Correct Clips: 3
- Misclassified Clips: 3
- walk_turn_walk → leg_raise_lower
- drink_return → wave
- wave → drink_return

교재 결과(0.6667 / 0.5556)와 다른 이유:

- 5일차에 추출한 Keypoint 값이 교재와 조금 달라 Feature와 Random Forest Tree가 달라졌습니다.
- Validation이 6개라 Clip 1개 차이가 0.1667입니다.

## 5. Static VS Sequence

- Static Feature Count: 38
- Static Accuracy: 0.6667
- Static Macro F1: 0.5556

- Sequence Feature Count: 1140
- Sequence Accuracy: 0.5000
- Sequence Macro F1: 0.4444

해석:

- 내 실행에서는 Static이 Clip 1개를 더 맞혔습니다.
- 현재의 작은 Validation Dataset에서는 시간 정보가 도움이 되었는지 판단할 수 없습니다.

## 6. Feature Set Comparison

- Coordinates + Angles
  - Feature Count: 1140
  - Accuracy: 0.5000
  - Macro F1: 0.4444

- Coordinates Only
  - Feature Count: 1020
  - Accuracy: 1.0000
  - Macro F1: 1.0000

해석:

- 현재 S02 Validation에서는 Coordinates Only가 더 높은 결과를 보였습니다.
- 하지만 Validation Clip이 6개뿐이므로 일반적인 결론으로 확대하지 않습니다.

## 7. Mini Challenge

- 변경한 조건: sequence_steps
- 변경 전: 30 Step (Feature 1,140)
- 변경 후: 15 Step (Feature 570)
- 실험 전 가설: 15 Step으로 줄이면 Feature가 570개로 줄어서, 6개짜리 작은 데이터에서는 Validation 결과가 달라질 것 같다.

- Baseline Accuracy: 0.5000
- Baseline Macro F1: 0.4444

- 변경 후 Accuracy: 0.6667
- 변경 후 Macro F1: 0.5833

- 예측이 바뀐 Clip:
  - S02_A006_T01 wave: drink_return → wave (정답으로 바뀜)
  - S02_A004_T01 drink_return: wave → leg_raise_lower (계속 오답)
- 결과 해석: Clip 1개 차이이고 Confidence가 0.20~0.36으로 낮아 15 Step이 더 좋다고 일반화하지 않습니다. 실험 후 30 Step으로 되돌려 Baseline을 복구했습니다.
- 상세: `reports/day06/challenge/mini_challenge_result.md`

## 8. Final Selected Config

- Sequence Steps: 30
- Sequence Feature Count: 1140
- min_valid_frames: 10
- n_estimators: 300
- max_depth: null
- random_state: 42

최종 조건을 선택한 이유:

- Validation 비교(Static, Coordinates Only, 15 Step)는 모두 Clip 1~3개 차이라 6개짜리 Validation으로는 어느 쪽이 더 좋다고 확정할 수 없어 원래 Baseline을 유지했습니다.
- 40번 Test 코드는 35번에서 학습한 `rf_action_baseline.joblib`(Coordinates + Angles, 1,140 Feature)을 사용하므로, Coordinates Only는 코드를 고치지 않으면 최종 조건으로 선택할 수 없습니다.

## 9. Final Test

- Test Subject: S03
- Test Clips: 6
- Accuracy: 1.0000
- Macro F1: 1.0000
- Correct Clips: 6
- Misclassified Clips: 0
- Confidence: 0.23 ~ 0.37

Test 결과를 본 뒤 조건을 다시 변경하지 않은 이유:

- Test는 최종 조건을 정한 뒤 한 번만 확인하는 데이터입니다. Test 결과에 맞춰 조건을 바꾸면 S03이 더 이상 처음 보는 사람이 아니게 됩니다.
- Validation(0.5000)과 Test(1.0000)가 다른 것은 오류가 아니라, S02와 S03이 서로 다른 사람이고 Split마다 한 사람뿐이기 때문입니다.

## 10. 현재 Dataset의 한계

- 3 Subjects × 6 Actions × 1 Take의 작은 교육용 Dataset입니다.
- Action당 Subject별 Clip이 1개뿐입니다.
- 현재 Accuracy를 현업 성능으로 해석하지 않습니다.
- Confidence가 0.20~0.37 수준으로 낮아, Tree 다수가 확신을 가지고 고른 결과가 아닙니다.

## 11. 6일차에서 배운 핵심

- 길이가 다른 Clip을 frame_id 기준 Sequence Resampling으로 30 Step에 맞추면 Clip 1개가 1,140개 숫자의 한 행이 됩니다.
- Train으로 학습하고, Validation으로만 조건을 비교하고, Test는 마지막에 한 번 봅니다.
- 실험은 한 번에 한 가지 조건만 바꿔야 결과 차이의 원인을 알 수 있습니다.
- 작은 데이터에서는 점수보다 어떤 Clip이 어떻게 틀렸는지를 원본 영상까지 따라가 확인하는 것이 중요합니다.
- 같은 코드여도 Keypoint 추출 결과가 조금 다르면 결과가 달라질 수 있습니다.

## 12. 다음 수업

6일차에서는 완성된 Clip 전체를 이용하여 Action을 분류했습니다.

7일차에서는 실시간 Frame을 계속 모아 Sliding Window를 만들고
실시간 Action Recognition으로 연결합니다.
