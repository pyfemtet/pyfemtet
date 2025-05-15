@echo off

REM Create feature branch and switch.
REM
REM Usage
REM =====
REM sample:
REM   .\tools\start-dev "sample"
REM result:
REM   Create "feature/sample" branch.

REM Check if branch name parameter is provided
if "%~1"=="" (
  echo "ERROR: No branch name provided."
  echo Usage: %~nx0 branch-name
  exit /b 1
)

set BRANCH_NAME=%~1
set FEATURE_BRANCH=feature/%BRANCH_NAME%

REM Check if branch already exists
git show-ref --verify --quiet refs/heads/%FEATURE_BRANCH%
if %ERRORLEVEL% EQU 0 (
  echo Branch "%FEATURE_BRANCH%" already exists. Checking out...
  git checkout %FEATURE_BRANCH%
  exit /b 0
)

REM Create and switch to new feature branch
echo Creating and switching to branch "%FEATURE_BRANCH%"...
git checkout -b %FEATURE_BRANCH%
