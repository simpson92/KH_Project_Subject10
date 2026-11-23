# Day 08 Tracking Cases

## 1. Shaking Hands

- 전체 Frame: 30
- Unique Track IDs: 2
- Two-ID Frame Ratio: 1.0000
- Extra ID: 0
- ID 1: 30 / 30 Frame
- ID 2: 30 / 30 Frame
- Log 해석: 두 Track이 전체 영상에서 유지됨
- 결과영상 확인:
- 최종 판단:

## 2. Hugging

- 전체 Frame: 33
- Unique Track IDs: 4
- Two-ID Frame Ratio: 0.6364
- Extra ID: 2
- ID 1: 24 / 33 Frame
- ID 2: 33 / 33 Frame
- ID 4: 4 Frame
- ID 5: 3 Frame
- Log 해석: Occlusion 구간에서 Tracking이 불안정해짐
- ID 4 / ID 5가 나타난 장면:
- 결과영상에서 본 현상:
- 최종 판단:
  - Track Lost + New ID
  - ID Switch 의심
  - 중복 Track
  - 판단 어려움

## 3. Kicking

- 전체 Frame: 18
- Unique Track IDs: 2
- Two-ID Frame Ratio: 1.0000
- Extra ID: 0
- ID 1: 18 / 18 Frame
- ID 2: 18 / 18 Frame
- Log 해석: 빠른 움직임에서도 두 Track 유지
- 결과영상 확인:
- 최종 판단:

## 4. 세 영상 비교

- 기본 Tracking: Shaking Hands → 안정적
- Occlusion: Hugging → 가장 큰 Tracking 변화 발생
- 빠른 움직임: Kicking → 이번 영상에서는 안정적
- 가장 눈에 띈 문제: Hugging에서 ID 4와 ID 5가 추가로 발생