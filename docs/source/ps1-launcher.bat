cd %~dp0
if exists .\pyfemtet-installer.ps1 (
    powershell -ExecutionPolicy ByPass .\pyfemtet-installer.ps1
) else (
    echo "`pyfemtet-installer.ps1` is not found in the same folder."
)
pause
