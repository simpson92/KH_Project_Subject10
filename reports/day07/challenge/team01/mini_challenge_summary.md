# Day 07 Mini Challenge

## 1. 내가 선택한 입력

Action ID: A003

Action Label: walk_turn_walk

직접 촬영한 MP4: 직접 촬영하지 않음 — 5일차 S03 원본 `S03_A003_T01.mp4`를 `S91_A003_T01.mp4`로 복사해 사용 (6일차 모델이 학습하지 않은 사람 = 처음 보는 입력)

영상 길이: 약 10초 (6개 Action 중 가장 길어서 기본 Window 5초로도 Prediction 가능)


## 2. 저장영상 Baseline

Raw Prediction 경향: Prediction 24회, Raw Label이 8번 바뀜 — 하나의 Action으로 고정되지 않음

UNKNOWN이 많았는가: 예 — unknown_ratio 1.0 (전부 UNKNOWN)

Confidence 경향: 평균 0.2187로 낮음 (6 Class 균등 0.167보다 조금 높은 수준)

Stable 결과: Stable Label 없음 (stable_label_changes 0, stable_match_ratio 0.0)


## 3. 한 가지 Runtime 실험

변경한 값: Confidence Threshold

기본값: 0.55

실험값: 0.40

가설: Threshold를 낮추면 UNKNOWN이 줄어들 것이다. 하지만 잘못된 Label도 더 쉽게 통과할 수 있다.


## 4. Baseline VS Experiment

실제로 달라진 점: 없음 — unknown_ratio 1.0, stable_match_ratio 0.0, Raw 변화 8번 모두 같음 (Processing FPS만 79.97 → 78.87로 실행마다 조금 다름)

가설과 비슷했는가: 아니오 — 평균 Confidence 0.22가 0.40보다도 낮아서 Threshold를 넘는 Prediction이 없었음

한 문장 해석: Threshold는 Confidence가 그 근처에 있을 때만 효과가 있고, 모델 Confidence 자체가 낮으면 설정을 바꿔도 결과가 바뀌지 않는다.


## 5. IP Webcam 결과

Target: walk_turn_walk (키 3)

Raw Prediction 경향: Prediction 56회, Raw Label 6번 바뀜

UNKNOWN 경향: 전부 UNKNOWN (unknown_ratio 1.0, stable_match_ratio 0.0)

Confidence 경향: 평균 0.2622로 낮음

저장영상과 달랐던 점: 사람이 Action을 하지 않고 사물만 비춘 상태로 실행해서 동작 인식 결과는 참고용. Processing FPS는 저장영상 약 79 vs IP Webcam 34.7


## 6. 오늘 이해한 내용

- Sliding Window는: 최근 약 5초의 Frame Feature를 모아 30 Step으로 Resampling한 뒤 모델에 넣는 방식이라, 처음 약 4초는 Window를 채우느라 Prediction이 나오지 않는다.
- Confidence Threshold는: 모델이 가장 비슷하다고 본 Label이라도 Confidence가 기준보다 낮으면 UNKNOWN으로 바꾸는 안전장치다. 모델 Confidence가 전체적으로 낮으면 기준을 조금 낮춰도 효과가 없다.
- Stabilization은: 최근 Prediction 5개 중 같은 Label이 3표 이상일 때만 Stable로 인정해서 흔들림을 줄인다. 입력이 전부 UNKNOWN이면 Stable도 나오지 않는다.
- 저장영상과 실시간 입력의 차이는: 실시간은 카메라·네트워크 때문에 Processing FPS가 낮아지고(약 79 → 34.7), 사람 위치·거리·Pose 품질이 매번 달라서 같은 Runtime 설정이어도 결과가 달라질 수 있다.
