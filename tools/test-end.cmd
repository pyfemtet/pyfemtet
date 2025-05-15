@echo off


REM move to project root
cd %~dp0\..

REM テスト実行中フラグがなければ何もしない
if not exist ".test-running" (
    echo Test is not running. Start `start-test.cmd` first.
    exit /b 0
)

REM 実際の処理
uv run python ".\tools\_impl\test_done.py" ".test-running"
