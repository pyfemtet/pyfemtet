@echo off

cd %~dp0\..\..
uv run --no-sync pybabel extract -F ./pyfemtet/_i18n/babel.cfg --no-wrap --ignore-dirs=".*" -o ./pyfemtet/_i18n/locales/messages.pot .
uv run --no-sync pybabel update -i ./pyfemtet/_i18n/locales/messages.pot --no-wrap -d ./pyfemtet/_i18n/locales -l ja

echo "Let's translate generated .po files!"
pause
