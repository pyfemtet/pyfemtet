@echo off
rem Run once per update source.
rem .pot file defines the target of string to translate.
rem .po file is the implementation of translation.

cd %~dp0\..\..
poetry run pybabel extract -F ./pyfemtet/_message/babel.cfg --no-wrap --ignore-dirs=".*" -o ./pyfemtet/_message/locales/messages.pot .
poetry run pybabel update -i ./pyfemtet/_message/locales/messages.pot --no-wrap -d ./pyfemtet/_message/locales -l ja

echo "生成された .po ファイルを翻訳してください。"
pause
