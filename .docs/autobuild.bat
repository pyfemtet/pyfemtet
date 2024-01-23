cd %~dp0\..
rd /s /q docs
mkdir docs
rd /s /q .docs\modules
mkdir .docs\modules
start http://127.0.0.1:8000
poetry run sphinx-autobuild .\.docs .\docs
