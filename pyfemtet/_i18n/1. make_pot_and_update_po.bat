@echo off

cd %~dp0\..\..
poetry run pybabel extract -F ./pyfemtet/_i18n/babel.cfg --no-wrap --ignore-dirs=".*" -o ./pyfemtet/_i18n/locales/messages.pot .
poetry run pybabel update -i ./pyfemtet/_i18n/locales/messages.pot --no-wrap -d ./pyfemtet/_i18n/locales -l ja

echo "Let's translate generated .po files!"
pause
