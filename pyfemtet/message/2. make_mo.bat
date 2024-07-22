rem Run after edit po.
rem .mo file created. This is compiled translation.

cd %~dp0\..\..
poetry run pybabel compile -d ./pyfemtet/message/locales
pause
