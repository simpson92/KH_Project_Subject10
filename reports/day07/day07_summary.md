# Day 07 Sliding Window & IP Webcam Realtime Summary

## 1. 사용한 모델

- Model: 6일차 Random Forest (`artifacts/day06/rf_action_baseline.joblib`, Tree 300 · max_depth null · seed 42)
- Action Classes: bend_return · drink_return · leg_raise_lower · sit_stand · walk_turn_walk · wave (6개)
- Sequence Steps: 30 (`rf_training_info.json`에서 읽음)
- Frame Feature: 38-D (정규화 좌표 34 + 관절각도 4)
- Final Feature Count: 1,140 (30 × 38)


## 2. 오늘 만든 Realtime Pipeline

Saved MP4 / IP Webcam
→ YOLO Pose
→ 38-D Frame Feature
→ Sliding Window
→ Sequence Resampling
→ Random Forest
→ Confidence Threshold
→ Stabilization
→ Runtime Result


## 3. 저장영상에서 확인한 핵심 결과

- Runtime Pipeline은 정상적으로 동작했는가: 예 — S02 영상에서 WARMING_UP 후 Prediction 발생 (walk_turn_walk 기본 설정 21회)
- Raw Prediction의 특징: Raw Label이 하나로 고정되지 않고 계속 바뀜
- UNKNOWN이 나타난 주요 이유: 평균 Confidence 약 0.23으로 Threshold 0.55보다 훨씬 낮음 — 모델을 S01 6 clip만으로 학습해서 처음 보는 사람에게 확신이 낮음
- Window / Threshold / Stabilization 중 가장 눈에 띈 변화: 셋 다 결과(UNKNOWN 100%)는 바뀌지 않음. Window는 줄일수록 짧은 wave 영상에서 Prediction 횟수가 늘어남 (3초 5회 → 2초 8회)


## 4. IP Webcam에서 확인한 핵심 결과

- Stream 연결: 정상 (1920×1080, 규칙적으로 살짝 끊김)
- Target 기록: 숫자 키로 Target 표시 확인
- Prediction 발생: 예 — 약 4초 WARMING_UP 후 발생
- Raw / Threshold / Stable에서 관찰한 특징: 사람 없이 화면·사물만 비춘 상태라 동작 인식 결과는 참고용. Confidence 약 0.23~0.26, 전부 UNKNOWN, Stable 없음
- 저장영상과 달랐던 점: Processing FPS가 저장영상 약 80 → IP Webcam 약 29~35로 낮음 (입력·네트워크 쪽 원인)


## 5. Mini Challenge

- 선택한 Action: A003 walk_turn_walk
- 직접 촬영한 MP4: 직접 촬영하지 않음 — 5일차 S03 영상을 `S91_A003_T01.mp4`로 복사해 사용
- 변경한 Runtime 변수: Confidence Threshold 0.55 → 0.40
- Baseline과 Experiment에서 달라진 점: 없음 — Confidence 0.22가 0.40보다도 낮아서 둘 다 UNKNOWN 100%
- IP Webcam 결과와 저장영상 결과의 차이: 둘 다 UNKNOWN 100%. IP Webcam은 사물만 비춰서 동작 비교는 못 함, FPS만 낮음


## 6. 현재 Pipeline의 한계

- 학습 데이터가 S01 6 clip뿐이라 처음 보는 사람에게 Confidence가 낮아 실시간 결과가 거의 UNKNOWN
- 화면에 사람이 한 명이라고 가정함 (여러 사람이면 누구를 따라갈지 모름)
- Runtime 설정(Threshold·Window·Buffer)은 결과 표시 방법만 바꿀 뿐 모델 성능을 올리지 못함


## 7. 한 문장 결론

- 저장영상과 IP Webcam 모두에서 Realtime Pipeline 자체는 정상적으로 동작했지만, 모델의 Confidence가 낮아 결과가 전부 UNKNOWN으로 나왔다.
