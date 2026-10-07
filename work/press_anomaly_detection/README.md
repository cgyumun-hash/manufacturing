# 파인블랭킹 프레스 유압펌프 이상 탐지 — 제출 코드

센서 3개(상부 진동 AI0, 하부 진동 AI1, 전류 AI2)를 0.1초마다 기록한 데이터로 유압펌프 이상을 탐지한다.
8개 모델 × 9개 입력 비교에서 가장 성능이 높았던 **LightGBM + 원데이터 1.0초 window** 모델에,
segment가 끝나기 전에도 경보를 내는 **마르코프 연속경고**를 더한 최종 구성만 담았다.

| 판정 방식 | Precision | Recall | F1 | 헛경보(FP) | 놓침(FN) |
|---|---:|---:|---:|---:|---:|
| segment 평균 점수 ≥ T* | 1.000 | 0.923 | 0.960 | 0 | 1 |
| **최종: segment 평균 OR 마르코프 연속경고** | **1.000** | **1.000** | **1.000** | **0** | **0** |

(5-fold Group CV, 정상 452 · 이상 13 segment의 test 예측을 모두 합친 결과)

---

## 1. 폴더 구성

```
press_anomaly_detection/
├─ README.md
├─ requirements.txt        패키지 목록 (버전 고정, uv로 설치)
├─ run_all.py              전체 실행: 학습 → 기준값 결정 → test 예측 (모든 OS 공통)
├─ run_all.sh / run_all.bat  uv 설치 확인 + 가상환경 생성 + 설치 + 실행을 한 번에 (macOS·Linux / Windows)
├─ train.py                1단계: fold별 LightGBM 학습 → models/
├─ valid.py                2단계: validation으로 T*, T_w, K 결정 → results/valid_thresholds.csv
├─ predict_test.py         3단계: test segment 예측 · 성능 → results/
├─ src/
│   ├─ config.py           경로 · 시드 · 하이퍼파라미터 · 기준값 설정
│   ├─ data.py             데이터 로드, 중복 제거, segment 분리
│   ├─ windows.py          window 생성, segment 단위 5-fold Group CV
│   ├─ model.py            표준화 + LightGBM
│   ├─ evaluation.py       segment 평균 판정, 임곗값 선택, 지표
│   └─ markov.py           마르코프 연속경고 (T_w, r, q, K)
├─ data/                   학습용 데이터 (원본 그대로)
│   ├─ press_data_normal.csv   정상 20,000행 (2022-07-12)
│   └─ outlier_data.csv        이상 600행 (2022-07-17)
├─ models/                 학습된 모델 (fold1 ~ fold5: scaler.joblib, lightgbm.joblib, lightgbm.txt)
└─ results/                실행 결과 (제출 시점 결과가 미리 들어 있음, 다시 실행하면 같은 값으로 덮어씀)
    ├─ test_predictions.csv     ★ 테스트 데이터 예측 결과 (segment별)
    ├─ test_window_scores.csv   window별 점수 · 경고 · 연속 횟수
    ├─ performance.csv / .md    성능 요약
    ├─ performance_by_fold.csv  fold별 성능
    ├─ valid_thresholds.csv     fold별 기준값 T*, T_w, r, q, K
    └─ fold_splits.csv          각 segment가 어느 fold의 train / valid / test인지
```

## 2. 환경 설정 (uv)

[uv](https://docs.astral.sh/uv/)로 Python과 패키지를 설치한다. uv가 Python 3.12를 직접 내려받으므로 **컴퓨터에 Python이 없어도 된다.**
GPU는 필요 없고, 첫 실행 때 패키지를 내려받기 위해 인터넷 연결이 필요하다. 전체 실행 시간은 수 초다.

### 2-1. uv 설치 (이미 있으면 건너뜀)

설치 여부 확인:
```bash
uv --version
```
버전이 나오지 않으면(명령을 찾을 수 없다는 오류) 아래 명령으로 설치한다.

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS · Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**설치가 끝나면 터미널(PowerShell) 창을 닫고 새로 연 뒤** `uv --version`으로 확인한다.

### 2-2. 가상환경 생성 · 패키지 설치

`requirements.txt`와 `run_all.py`가 있는 이 폴더에서 실행한다.

```bash
cd press_anomaly_detection
uv venv --python 3.12              # Python 3.12가 없으면 uv가 자동으로 내려받아 .venv 생성
uv pip install -r requirements.txt # 버전 고정 패키지 설치
```

Python 3.10 ~ 3.12를 쓸 수 있다 (pandas 2.2.2가 3.13 이상용 설치 파일을 제공하지 않음).
Python 3.10 · 3.12 가상환경에서 각각 새로 설치해 실행했고, 두 경우 모두 `results/`와 `models/`가 제출 파일과 바이트 단위로 같았다.

> macOS에서 LightGBM이 `libomp` 오류를 내면 `brew install libomp`를 실행한다.

### 한 번에 실행 (2-1 ~ 3을 자동으로)

```bash
bash run_all.sh      # macOS · Linux
```
```bat
run_all.bat          :: Windows — 파일 탐색기에서 run_all.bat을 더블클릭해도 된다
```
uv가 없으면 설치하고, `.venv`가 없으면 만들어 패키지를 설치한 뒤 전체를 실행한다.

## 3. 실행

```bash
uv run run_all.py          # 전체 (학습 → 기준값 결정 → test 예측)

uv run train.py            # 단계별로 실행할 때
uv run valid.py
uv run predict_test.py
```
`uv run`은 이 폴더의 `.venv`를 자동으로 사용하므로 가상환경을 따로 활성화(activate)할 필요가 없다.

정상 실행 시 마지막 출력:
```
segment 평균만      : Precision 1.000  Recall 0.923  F1 0.960
최종 (OR 연속경고)  : Precision 1.000  Recall 1.000  F1 1.000
```
모든 시드와 LightGBM 스레드 수(1, deterministic)를 고정해, 다른 컴퓨터에서도 `results/`와 같은 값이 나온다.

## 4. 테스트 결과

별도의 test 파일이 없으므로 **segment 단위 5-fold Group CV**로 평가했다. 각 fold의 test segment는 학습과 기준값 결정에 쓰지 않았고,
5개 fold를 합치면 465개 segment(정상 452 · 이상 13)가 모두 한 번씩 test로 예측된다. 아래는 이 예측을 모두 합친 결과다 (양성 = 이상).

| 판정 방식 | Precision | Recall | F1 |
|---|---:|---:|---:|
| segment 평균 점수 ≥ T* | 1.000 | 0.923 | 0.960 |
| **최종: segment 평균 OR 마르코프 연속경고** | **1.000** | **1.000** | **1.000** |

- 혼동행렬 (최종): TP 13 · FN 0 · FP 0 · TN 452. segment 평균만 쓰면 이상 1개(anomaly_12)를 놓치고, 연속경고가 이를 segment 시작 후 3.4초에 잡는다.

### 결과 파일 — `results/`

**`test_predictions.csv`** — 테스트 데이터 예측 결과 (465개 segment 각 1행)

| 열 | 뜻 |
|---|---|
| fold | 이 segment가 test로 쓰인 fold (1~5) |
| segment | segment 이름 (normal_5 = 정상 데이터의 5번째 연속 구간) |
| start_time, end_time, samples | segment 시작·끝 시각, 기록 수 (0.1초 간격) |
| label | 정답 (0 = 정상, 1 = 이상) |
| segment_mean_score | segment 안 window 이상 점수의 평균 |
| T_star, pred_segment_mean | segment 평균 임곗값과 그 판정 (점수 ≥ T* → 1) |
| T_w, K, pred_consecutive | 경고 기준, 필요한 연속 횟수, 연속경고 판정 (경고 K번 연속 → 1) |
| first_alarm_seconds | 연속경고로 경보가 난 시점 (segment 시작 후 초) |
| **prediction** | **최종 예측** = pred_segment_mean OR pred_consecutive |
| correct | 정답과 같으면 1 |

| 파일 | 내용 |
|---|---|
| `test_window_scores.csv` | test window별 점수, 경고 여부(점수 ≥ T_w), 그 시점까지의 연속 경고 횟수 |
| `performance.csv` / `performance.md` | 전체 성능 요약 |
| `performance_by_fold.csv` | fold별 Precision · Recall · F1 · TP/FN/FP/TN |
| `valid_thresholds.csv` | fold별 T*, T_w, 정상 전이 수(n00 · n01 · n10 · n11), r, q, K |
| `fold_splits.csv` | 각 segment가 어느 fold의 train / valid / test에 들어갔는지 |

## 5. 방법 요약

1. **segment 분리** — 이웃 기록 간격이 정확히 0.1초가 아니면 끊긴 것으로 보고 그 자리에서 나눈다. 공백을 채우지(보간) 않는다.
   중복 1행은 제거한다. 2초(20시점) 이상 이어진 segment만 쓴다 → 정상 452 · 이상 13.
2. **입력** — 1.0초 window = 10시점 × 3센서 = 30개 값. segment 안에서만 0.5초씩 밀며 만들고 segment당 최대 10개를 고르게 고른다.
3. **누수 방지** — segment 단위 5-fold Group CV. test를 뺀 나머지를 train 75% · validation 25%로 나눈다.
   같은 segment의 window는 한 곳에만 들어간다. 표준화는 train 정상 window로만 맞춘다.
4. **모델** — LightGBM (트리 180개, 학습률 0.05, num_leaves 15, max_depth 5, class_weight balanced).
5. **segment 판정** — window 점수 평균 ≥ T*. T*는 validation segment에서 F1이 가장 높은 값이며 test에는 그대로 적용한다.
6. **마르코프 연속경고** — 0.5초마다 나오는 window 점수가 T_w(validation 정상 window 점수의 99% 분위수) 이상이면 '경고'.
   validation 정상 segment 안에서 경고가 새로 시작될 확률 r과 이어질 확률 q를 구하고,
   시간당 판정 수 N = 7,200에 대해 N·π0·r·q^(K−1) ≤ 0.01 을 만족하는 가장 작은 K를 고른다 (fold별 K = 5 ~ 7).
   segment 안에서 경고가 K번 연속되면 segment가 끝나기 전에 경보한다.
7. **최종 판정** = segment 평균 판정 OR 연속경고.

## 6. 주의

- 정상(2022-07-12)과 이상(2022-07-17)의 수집 날짜가 달라 날짜·운전조건 효과와 고장 효과가 섞여 있다.
- 독립적인 이상 segment가 13개뿐이라 이상 1개가 Recall을 약 7.7%p 바꾼다.
- 위 성능은 현재 데이터 안의 교차검증 결과이며, 새 날짜의 데이터로 외부 검증이 필요하다.
