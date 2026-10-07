# 성능 (5-fold Group CV, test segment 예측을 모두 합친 OOF)

- test segment: 정상 452 · 이상 13

| 판정 방식 | Precision | Recall | F1 | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| segment 평균 ≥ T* | 1.000 | 0.923 | 0.960 | 12 | 1 | 0 | 452 |
| **최종: segment 평균 OR 마르코프 연속경고** | 1.000 | 1.000 | 1.000 | 13 | 0 | 0 | 452 |

segment 평균 판정의 PR-AUC: 1.000
