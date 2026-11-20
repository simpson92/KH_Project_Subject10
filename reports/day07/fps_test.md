# Day 07 Processing FPS Check

## Saved Video

사용한 영상: S02_A003_T01 (walk_turn_walk), S02_A006_T01 (wave) — 640×480

평균 Processing FPS: 약 77 (실행 7회 범위 72.7 ~ 80.0, 기본 설정 w5_c0.55_s5 = 80.0)

느리다고 느껴졌는가: 아니오 — 원본 영상 25 FPS보다 처리 속도가 충분히 빠름


## IP Webcam

Stream Resolution: 1920×1080

평균 Processing FPS: 28.6

영상 자체의 지연이 느껴졌는가: 예 — 규칙적으로 살짝 끊김

Prediction 시작이 늦다고 느껴졌는가: 약 4초 대기 후 시작 (Window 5초 × fill ratio 0.8)


## 원인을 구분해 봅니다

카메라·네트워크 문제로 보이는가: 예 — 1920×1080 Stream을 Wi-Fi로 받아서 끊김이 생김. Processing FPS 28.6은 카메라 입력 속도(약 30 FPS)에 묶인 값으로 보임

AI Processing 문제로 보이는가: 아니오 — 같은 모델·GPU로 저장영상은 약 77 FPS 처리

Sliding Window 때문에 기다린 것으로 보이는가: 예 (시작 시점만) — 처음 약 4초는 Window를 채우는 정상 대기 시간


## 한 줄 정리

- AI 처리는 충분히 빠르고(저장영상 약 77 FPS), IP Webcam의 끊김과 낮은 FPS(28.6)는 1080p 입력·네트워크 쪽 원인으로 보임