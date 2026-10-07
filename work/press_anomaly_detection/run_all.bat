@echo off
REM Windows: (uv가 없으면 설치) -> Python 3.12 가상환경 생성 -> 패키지 설치 -> 전체 실행
chcp 65001 > nul
cd /d "%~dp0"
where uv > nul 2>&1
if errorlevel 1 (
  echo uv가 없어 설치합니다...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
if not exist .venv (
  uv venv --python 3.12
  uv pip install -r requirements.txt
)
uv run run_all.py
pause
