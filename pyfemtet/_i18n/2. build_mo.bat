@echo off

cd %~dp0\..\..
uv run pybabel compile -d ./pyfemtet/_i18n/locales
pause
