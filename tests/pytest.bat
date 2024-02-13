cd %~dp0\..
poetry run pytest -s
pause

rem pytest --lf  # 前回失敗したものだけ