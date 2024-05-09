cd %~dp0\..
poetry run pytest ./tests -s --lf
pause
exit

# all(with stdout)
poetry run pytest ./tests -s

# last failed only
poetry run pytest ./tests -s --lf

# (un)match function only
poetry run pytest ./tests -s -k "not _cad and not _sample"
