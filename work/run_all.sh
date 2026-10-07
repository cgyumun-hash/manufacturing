#!/usr/bin/env bash
# work 폴더 전체를 처음부터 다시 만든다 (약 10분).
#   2장 데이터 그림·표 → 8개 모델 train/valid/test/markov → 요약 → SHAP·ablation → 3장·4장 그림
set -e
cd "$(dirname "$0")"

echo "== 2장: 데이터 전처리·특징 =="
for f in data/2.*/make.py; do python3 "$f" > /dev/null && echo "  $f"; done

echo "== 3장·5장: 8개 모델 =="
for m in LogisticRegression LightGBM XGBoost Mahalanobis OneClassSVM CNN1D GRU LSTM_AE; do
  for stage in train valid test markov; do
    python3 "models/$m/$stage.py" | tail -1
  done
done
python3 models/summarize.py > /dev/null

echo "== 4.3: SHAP · ablation =="
python3 shap_ablation/run_group_ablation.py | tail -2
python3 shap_ablation/run_shap.py > /dev/null

echo "== 3장·4장 그림 =="
for f in figures_ch3/*/make.py figures_ch4/*/make.py; do python3 "$f" > /dev/null && echo "  $f"; done
echo "완료"
