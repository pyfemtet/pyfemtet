@echo off
rem Run once per update source.
rem .pot file defines the target of string to translate.
rem .po file is the implementation of translation.

cd %~dp0\..\..
poetry run pybabel extract -F ./pyfemtet/message/babel.cfg --no-wrap -o ./pyfemtet/message/locales/messages.pot .
rem poetry run pybabel init -i ./pyfemtet/message/locales/messages.pot --no-wrap -d ./pyfemtet/message/locales -l ja
poetry run pybabel update -i ./pyfemtet/message/locales/messages.pot --no-wrap -d ./pyfemtet/message/locales -l ja

echo "生成された .po ファイルを翻訳してください。"
pause
