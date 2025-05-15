@echo off

REM move to project root
cd %~dp0\..

REM テスト実行中フラグがあれば何もしない
if exist ".test-running" (
    echo Test is already running. Exiting.
    exit /b 0
)

REM テスト実行中フラグを立てる
(
    echo %DATE% %TIME%
    echo Remove this file to cancel test.
) > ".test-running"
