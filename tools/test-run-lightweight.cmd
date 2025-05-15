@echo off

REM Run all lightweight tests.
REM
REM Usage
REM =====
REM sample:
REM   .\tools\test-lightweight      # Run all tests.
REM   .\tools\test-lightweight -lf  # Run last failed.
REM
REM result:
REM   Run all lightweight tests.
REM

REM move to project root
cd %~dp0\..

REM オプション判定
set CACHE_CLEAR=1
if /i "%1"=="-lf" set CACHE_CLEAR=0

REM 手動テストを実行
uv run python ".\tools\_impl\run_tests.py" "all" 0 0 1 %CACHE_CLEAR%

echo Test finished.
