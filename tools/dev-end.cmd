@echo off

REM Merge from main branch, run module test and merge to main.
REM
REM Usage
REM =====
REM sample:
REM   .\tools\end-dev
REM   .\tools\end-dev -na  # Don't abort merging if it failed. (No Abort)
REM result:
REM   If it fails to merge from main branch, mail notify and reset merging.
REM   If test fails, mail notify and don't merge to main.

REM move to project root
cd %~dp0\..

REM オプション判定
set ABORT_ON_FAIL=1
if /i "%1"=="-na" set ABORT_ON_FAIL=0

REM 現在のブランチ名を取得
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD') do set CURRENT_BRANCH=%%i

REM mainブランチを作業ブランチにマージ
echo === Merging main into %CURRENT_BRANCH%... ===
git merge main

REM マージ失敗時はマージの取り消し
if errorlevel 1 (

    echo === Merge from main failed. ===

    if %ABORT_ON_FAIL% EQU 1 (
        echo === Aborting merge... ===
        git merge --abort

        REM TODO: メール通知コマンド（例: powershell でメール送信など）
        call .\tools\notify "Merge failed!" "main cannot merge to %CURRENT_BRANCH%. Merging aborted. Please do it manually."

    )

    exit /b 1
)


REM モジュールテストを実行
echo === Running module tests... ===
call .\tools\run-tests.cmd "module" 0 0

REM テスト失敗時は終了
if errorlevel 1 (
  echo === Tests failed. Not merging to main. ===
  exit /b 1
)

REM テストに成功しているので main ブランチにマージ

REM mainブランチに切り替え
git checkout main
if errorlevel 1 (
  echo === Failed to checkout main branch. ===
  exit /b 1
)

REM 作業ブランチをmainにマージ
echo === Merging %CURRENT_BRANCH% into main... ===
git merge %CURRENT_BRANCH%

REM マージ失敗時はマージの取り消し
if errorlevel 1 (

    echo === Merge to main failed. ===

    echo === Aborting merge... ===
    git merge --abort

    REM TODO: メール通知コマンド（例: powershell でメール送信など）
    call .\tools\notify "Merge failed!" "Test passed, but %CURRENT_BRANCH% cannot merge to main.\nMerging aborted. Please do it manually."

    exit /b 1
)


echo === Finished successfully. ===
exit /b 0
