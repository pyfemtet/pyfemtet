@echo off

REM Run all automated tests.
REM
REM Usage
REM =====
REM sample:
REM   .\tools\test-all      # Run all tests.
REM   .\tools\test-all -lf  # Run last failed.
REM
REM result:
REM   Run all automated tests.
REM

REM move to project root
cd %~dp0\..

REM オプション判定
set CACHE_CLEAR=1
if /i "%1"=="-lf" set CACHE_CLEAR=0

REM 自動化テストを実行
uv run python ".\tools\_impl\run_tests.py" "all" 1 1 0 %CACHE_CLEAR%

echo Test finished.
