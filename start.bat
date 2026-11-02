@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found.
  echo Create it with: py -3.12 -m venv .venv
  echo Then activate it and install requirements.txt.
  pause
  exit /b 1
)
.venv\Scripts\python.exe assistant.py
