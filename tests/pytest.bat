cd %~dp0\..
poetry run pytest -s -k "not _cad and not _sample"
pause
exit

# all(with stdout)
poetry run pytest -s

# last failed only
poetry run pytest -s --lf

# (un)match function only
poetry run pytest -s -k "not _cad and not _sample"
