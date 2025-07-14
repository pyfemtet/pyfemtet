@echo off
cd %~dp0
if exist ".\pyfemtet-installer.ps1" (
    powershell -executionpolicy bypass -file .\pyfemtet-installer.ps1
) else (
    echo `pyfemtet-installer.ps1` is not found in the same folder.
)
pause
