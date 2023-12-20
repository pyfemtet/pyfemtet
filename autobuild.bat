cd %~dp0
start http://127.0.0.1:8000
poetry run sphinx-autobuild .\.docs .\docs
