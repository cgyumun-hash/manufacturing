# work — 프레스 유압펌프 이상 탐지 작업 폴더

발표 문서 `발표용_전체_1-6장.docx`의 내용을 **장별로 나눠 다시 실행할 수 있게** 정리한 폴더다.
모든 그림·표·성능 수치는 이 폴더의 코드로 원데이터에서 다시 계산되며, 문서의 숫자와 일치한다.

```
work/
├─ run_all.sh                 전체 재실행 (약 10분)
├─ common/                    공통 코드 — 모든 폴더가 같은 규칙을 쓴다
│   ├─ data.py                원데이터 로드, 중복 제거, 0.1초 아니면 segment 분리 (2장)
│   ├─ windows.py             9개 입력 구성, window 생성, segment 단위 5-fold Group CV (2.6·2.7)
│   ├─ features.py            Tabular 파생변수 19개 · Sequence 파생채널 12개 (3.4·3.5)
│   ├─ preprocess.py          입력 형식별 전처리 + 표준화 (train 정상으로만 fit)
│   ├─ evaluation.py          segment 평균 점수, 임곗값 T* 선택, Precision·Recall·F1·PR-AUC (3.6·3.7)
│   ├─ pipeline.py            모델 폴더의 train / valid / test / markov 공통 흐름
│   ├─ torch_models.py        1D-CNN · GRU · LSTM-AE 학습 루프
│   ├─ markov.py              보완 마르코프 연속경고: T_w, r, q, K (5장)
│   ├─ results.py             모델 결과 읽기 (3·4장 그림용)
│   └─ plot_style.py          그림 스타일 (matplotlib 기본, 정상 파랑 · 이상 빨강)
│
├─ data/                      ── 2장 데이터 설명과 전처리
│   ├─ raw/                   원데이터 2개 (press_data_normal.csv, outlier_data.csv)
│   ├─ 2.1_데이터_규모/        각 폴더 = make.py + 그림_*.png + 표_*.csv
│   ├─ 2.2_품질_점검/
│   ├─ 2.2_타임스탬프_간격/
│   ├─ 2.3_값_분포와_기술통계/
│   ├─ 2.3_크기와_모양_지표/
│   ├─ 2.3_센서_상관/
│   ├─ 2.4_대표_segment_파형/
│   ├─ 2.4_전류_반복_박자/
│   ├─ 2.4_센서_관계/
│   ├─ 2.4_PCA/
│   ├─ 2.5_segment_분리/
│   ├─ 2.6_입력_길이/
│   └─ 2.7_window_GroupCV/
│
├─ models/                    ── 3장 모델 학습·평가 + 5장 마르코프 (모델마다 따로 관리)
│   ├─ LogisticRegression/  LightGBM/  XGBoost/  Mahalanobis/  OneClassSVM/
│   ├─ CNN1D/  GRU/  LSTM_AE/
│   │   ├─ model.py           모델 정의와 하이퍼파라미터
│   │   ├─ train.py           9개 입력 × 5 fold 학습 → artifacts/<입력>/fold<k>/ (전처리기 + 모델)
│   │   ├─ valid.py           validation으로 임곗값 T* 선택 → results/valid_thresholds.csv
│   │   ├─ test.py            T*를 test에 적용 → results/performance.csv, 성능표.md
│   │   ├─ markov.py          5장: 최고 입력 하나에 마르코프 연속경고 → markov/
│   │   ├─ artifacts/         학습된 전처리기·모델 파일
│   │   ├─ results/           train_log, valid_thresholds, valid_segment_scores,
│   │   │                     test_predictions, fold_metrics, performance(성능표), 성능표.md
│   │   └─ markov/            markov_folds, markov_segments, markov_windows, markov_summary, 마르코프_결과.md
│   ├─ summarize.py           8개 모델 결과 모으기
│   └─ summary/               all_performance(72행), best_by_model, markov_summary_all, 전체_요약.md
│
├─ figures_ch3/               ── 3장 시각화 (각 폴더 = make.py + 그림 + 표)
│   ├─ 3.7_임곗값_선택_예시/
│   ├─ 3.8_F1_히트맵/
│   ├─ 3.8_모델별_최고_입력/
│   └─ 3.9_입력_형식별_F1/
├─ figures_ch4/               ── 4장 결과 해석
│   ├─ 4.1_LightGBM_입력길이별/
│   ├─ 4.2_단일시점_대비_최고/
│   └─ 4.3_SHAP_ablation/      (shap_ablation/ 결과로 그림)
└─ shap_ablation/             ── 4.3 SHAP · 그룹 ablation 실험
    ├─ ablation_setup.py      실험 설정 (분할 시드 20261005, 9개 모델, 0.5~2.0초)
    ├─ ablation_models.py     실험에 쓴 모델
    ├─ run_group_ablation.py  파생변수 그룹을 하나씩 빼고 F1 감소 측정
    ├─ run_shap.py            Tabular · Sequence 최고 모델의 SHAP
    └─ results/               cv_folds, cv_summary, group_drops, shap_folds, shap_summary
```

## 실행 순서

```bash
bash work/run_all.sh                       # 전부

python work/data/2.5_segment_분리/make.py   # 2장 폴더 하나만
python work/models/LightGBM/train.py       # 모델 하나: train → valid → test → markov 순서
python work/models/LightGBM/valid.py
python work/models/LightGBM/test.py
python work/models/LightGBM/markov.py
python work/models/summarize.py            # 8개 모델을 모두 돌린 뒤
```

필요 패키지: numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, torch, matplotlib, joblib, 나눔고딕 폰트.

## 공통 실험 규칙 (문서 2~3장)

| 항목 | 규칙 |
|---|---|
| segment | 이웃 기록 간격이 정확히 0.1초가 아니면 분리. 보간하지 않음 |
| 사용 segment | 모든 입력에서 2초(20시점) 이상인 같은 segment — 정상 452 · 이상 13 |
| 입력 9개 | 단일시점, 원데이터 0.5 · 1.0 · 1.5 · 2.0초, 파생 0.5 · 1.0 · 1.5 · 2.0초 |
| window | segment 안에서만, 이동 간격 = 시점 수 ÷ 2 (내림), segment당 최대 10개 |
| 분할 | segment 단위 5-fold Group CV, 남은 segment를 train 75% : valid 25% (시드 20261007) |
| 전처리 | 표준화·전류 회귀식은 train **정상** window로만 맞춤 |
| 판정 | segment 안 window 점수 평균 ≥ T* 이면 이상 |
| 임곗값 T* | validation segment에서 F1 최대 — test에는 그대로 적용 |
| 성능 | 5개 test fold 예측을 합친 OOF Precision · Recall · F1, PR-AUC |

5장 마르코프(`markov.py`)는 3장 성능표에서 가장 좋았던 입력 하나만 쓰고, 이미 학습된 모델과 T*를 그대로 쓴다.
validation 정상 window 점수의 99% 분위수를 T_w로, 정상 segment 안 전이로 r · q를 구해
N·π0·r·q^(K−1) ≤ 0.01 을 만족하는 가장 작은 K를 고른다. 판정 간격은 이동 간격(예: 1.0초 입력 = 0.5초)이다.

## 결과 확인 (기존 분석과 대조)

| 대상 | 비교 | 결과 |
|---|---|---|
| 8개 모델 × 9개 입력 성능 (72개) | `storytelling_analysis/08_model_comparison` | F1 · Precision · Recall · TP/FP/FN 모두 동일 |
| 5장 마르코프 (8개 모델 × 5 fold) | `storytelling_analysis/16_markov_run_alarm` | T*, T_w, 전이 수, K, F1 모두 동일 |
| SHAP · 그룹 ablation | `feature_ablation_shap_summary.csv`, `feature_ablation_group_drops.csv` | 모두 동일 |
| 2장 그림 속 수치 | 문서 표 3 ~ 18 | 동일 |

## 주요 결과

| 모델 | 최고 입력 | OOF F1 | + 마르코프 연속경고 F1 |
|---|---|---:|---:|
| LightGBM | 원데이터 1.0초 | **0.960** | **1.000** |
| XGBoost | 파생 1.5초 | 0.923 | 0.923 |
| 1D-CNN | 원데이터 1.5초 | 0.917 | 1.000 |
| Logistic Regression | 파생 2.0초 | 0.917 | 0.917 |
| One-Class SVM | 파생 1.5초 | 0.917 | 0.917 |
| GRU | 원데이터 1.0초 | 0.917 | 0.880 |
| Mahalanobis | 원데이터 0.5초 | 0.917 | 1.000 |
| LSTM-Autoencoder | 파생 1.5초 | 0.632 | 0.632 |

주의: 정상(2022-07-12)과 이상(2022-07-17)의 수집 날짜가 달라 날짜 효과와 고장 효과가 섞여 있고, 이상 segment가 13개뿐이다.
이 점수는 현재 데이터 안의 비교 결과이며 현장 성능으로 확정할 수 없다.
