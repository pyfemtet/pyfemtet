@echo off

cd %~dp0\..\..
uv run --no-sync pybabel compile -d ./pyfemtet/_i18n/locales
pause
