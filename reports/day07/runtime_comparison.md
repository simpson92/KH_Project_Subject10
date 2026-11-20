# Day 07 Runtime Check

## 기본 Runtime 설정

Window Seconds: 5.0

Confidence Threshold: 0.55

Stabilization Buffer: 5


## 저장영상에서 확인한 내용

사용한 영상: S02_A003_T01 (walk_turn_walk), S02_A006_T01 (wave)

Prediction 발생 여부: 예 — 기본 설정 A003에서 Prediction 21회

UNKNOWN이 많았는가: 예 — 모든 실행에서 unknown_ratio 1.0 (전부 UNKNOWN)

관찰한 특징:
- 평균 Confidence가 약 0.23~0.26으로, Threshold 0.55는 물론 0.4보다도 낮음
- Raw Label은 5번 바뀌었지만 Stable Label은 한 번도 나오지 않음 (stable_match_ratio 0.0)
- Threshold 0.4, Window 2·3초, Buffer 3·7로 바꿔도 UNKNOWN 1.0 그대로
- 원인 추정: 6일차 모델은 S01 6 clip만으로 학습해서 처음 보는 사람(S02)에게는 확신이 낮음 (6 class 균등이면 0.167)


## IP Webcam에서 확인한 내용

사용한 Action: wave (사람 없이 화면만 비추고 실행 — 실제 동작이 아니라서 동작 인식 결과는 참고용)

Raw Prediction: 112회 갱신, Raw Label 35번 바뀜

Confidence 경향: 평균 0.23으로 낮음

Threshold 결과: 전부 UNKNOWN (unknown_ratio 1.0)

Stable 결과: Stable Label 없음 (stable_match_ratio 0.0)


## 저장영상과 IP Webcam의 차이

- Processing FPS: 저장영상 약 77 vs IP Webcam 28.6
- 두 입력 모두 Confidence 약 0.23, UNKNOWN 100%
- IP Webcam은 Raw Label이 훨씬 자주 바뀜 (112회 중 35번 vs 저장영상 21회 중 5번)


## 한 줄 정리

- Pipeline(Frame → Pose → 38-D → Window → RF → Threshold → Stable)은 정상으로 동작했지만, 학습 데이터가 S01 6 clip뿐이라 Confidence가 낮아서 결과가 전부 UNKNOWN으로 나옴