# Day 08 Tracking Cases

## 1. Shaking Hands

- 전체 Frame: 30
- Unique Track IDs: 2
- Two-ID Frame Ratio: 1.0000
- Extra ID: 0
- ID 1: 30 / 30 Frame (mean box conf 0.9306 · keypoint 0.7009)
- ID 2: 30 / 30 Frame (mean box conf 0.9370 · keypoint 0.6981)
- Log 해석: 두 Track이 전체 영상에서 유지됨
- 결과영상 확인: 악수하며 가까워져도 두 사람이 처음부터 끝까지 같은 ID를 유지함
- 최종 판단: ID 유지 (Track Lost · ID Switch 없음)

## 2. Hugging

- 전체 Frame: 33
- Unique Track IDs: 4
- Two-ID Frame Ratio: 0.6364
- Extra ID: 2
- ID 1: 24 / 33 Frame (mean box conf 0.7344 · keypoint 0.6230)
- ID 2: 33 / 33 Frame (mean box conf 0.8600 · keypoint 0.6510)
- ID 4: 4 Frame, Frame 6~11 (mean box conf 0.3867 · keypoint 0.3287)
- ID 5: 3 Frame, Frame 20~23 (mean box conf 0.2981 · keypoint 0.3294)
- Log 해석: Occlusion 구간에서 Tracking이 불안정해짐
- ID 4 / ID 5가 나타난 장면: 두 사람이 안았을 때, 키가 더 큰 사람 쪽으로 BBox가 겹쳐 나타남
- 결과영상에서 본 현상: 안은 동안 키 큰 사람 쪽에 Track이 겹쳤고, 떨어진 뒤에는 두 사람이 원래 ID로 돌아감
- 최종 판단:
  - [ ] Track Lost + New ID
  - [ ] ID Switch 의심
  - [x] 중복 Track — 겹친 구간에 낮은 신뢰도(0.30~0.39)의 추가 Track이 잠깐 생김. 분리 후 원래 ID가 유지되어 ID Switch는 아님
  - [ ] 판단 어려움

## 3. Kicking

- 전체 Frame: 18
- Unique Track IDs: 2
- Two-ID Frame Ratio: 1.0000
- Extra ID: 0
- ID 1: 18 / 18 Frame (mean box conf 0.9306 · keypoint 0.7777)
- ID 2: 18 / 18 Frame (mean box conf 0.9346 · keypoint 0.7389)
- Log 해석: 빠른 움직임에서도 두 Track 유지
- 결과영상 확인: 발차기 전·순간·후 모두 같은 사람이 같은 ID를 유지함
- 최종 판단: ID 유지 (Track Lost · ID Switch 없음)

## 4. 세 영상 비교

- 기본 Tracking: Shaking Hands → 안정적
- Occlusion: Hugging → 가장 큰 Tracking 변화 발생
- 빠른 움직임: Kicking → 이번 영상에서는 안정적
- 가장 눈에 띈 문제: Hugging에서 ID 4와 ID 5가 추가로 발생 (안았을 때 키 큰 사람 쪽에 겹친 짧은 추가 Track, 분리 후 원래 ID로 복귀)