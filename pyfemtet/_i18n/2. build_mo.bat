@echo off

cd %~dp0\..\..
poetry run pybabel compile -d ./pyfemtet/_i18n/locales
pause
