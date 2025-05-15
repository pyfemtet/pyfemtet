@echo off

REM Usage
REM =====
REM sample:
REM   .\tools\notify "subject" "this is body\nsample"
REM "\n" is replaced in python code.


REM move to project root
cd %~dp0\..

REM check arguments
if "%~1"=="" (
    echo "Error: Subject (第一引数) を指定してください。"
    goto :eof
)
if "%~2"=="" (
    echo "Error: Body (第二引数) を指定してください。"
    goto :eof
)

REM mail notify
uv run python ".\tools\_impl\notify.py" "%~1" "%~2"
