@echo off

REM Release process.
REM
REM check test,
REM merge to release,
REM estimate version name
REM and add tag.
REM
REM Usage
REM =====
REM sample:
REM   .\tools\release
REM
REM

REM move to project root
cd %~dp0\..

REM テスト実行中フラグがなければ何もしない
if not exist ".test-running" (
    echo === Test is not running. Start `start-test.cmd` first. ===
    exit /b 0
)

REM main ブランチであることを確認
for /f "tokens=*" %%b in ('git rev-parse --abbrev-ref HEAD') do set current_branch=%%b
if NOT "%current_branch%"=="main" (
    echo === Current branch is "%current_branch%". Please switch to "main" branch before release.===
    exit /b 1
)

REM .test-running より後にできた
REM テスト成功レポートが存在するか
uv run python .\tools\_impl\check_test_report.py ".test-running"

REM 存在しなければ終了
if errorlevel 1 (
    echo === No valid reports. ==
    exit /b 1
)

REM release ブランチに切り替え
git checkout release
if errorlevel 1 (
    echo === Failed to checkout release branch. ===
    exit /b 1
)

REM main にマージ
git merge main
if errorlevel 1 (
    echo Merge failed.
    exit /b 1
)

REM 最新タグ名を取得し、マイナーバージョンに +1 する
for /f "tokens=*" %%t in ('git describe --tags --abbrev=0') do set latest_tag=%%t
set "ver=%latest_tag:~1%"

REM バージョン番号を分割
for /f "tokens=1,2,3 delims=." %%a in ("%ver%") do (
    set major=%%a
    set minor=%%b
    set patch=%%c
)

REM マイナーバージョンに +1 し、パッチは0にリセット
set /a minor=minor+1
set patch=0

REM 新しいタグを作成
set new_tag=v%major%.%minor%.%patch%
echo New tag will be %new_tag%

REM タグ付け
git tag %new_tag%
if errorlevel 1 (
    echo Failed to create tag %new_tag%.
    exit /b 1
)

REM 切り戻し
git checkout %current_branch%
if errorlevel 1 (
    echo Failed to checkout back to %current_branch%.
    exit /b 1
)

REM .test-running を削除
del /q ".test-running"
