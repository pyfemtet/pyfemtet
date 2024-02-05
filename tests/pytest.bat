cd %~dp0\..
poetry run pytest -s --lf
pause

rem pytest --lf  # 前回失敗したものだけ