#!/usr/bin/env bash
# macOS · Linux: (uv가 없으면 설치) → Python 3.12 가상환경 생성 → 패키지 설치 → 전체 실행
set -e
cd "$(dirname "$0")"
if ! command -v uv > /dev/null 2>&1; then
  echo "uv가 없어 설치합니다..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d .venv ]; then
  uv venv --python 3.12
  uv pip install -r requirements.txt
fi
uv run run_all.py
