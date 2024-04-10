cd %~dp0\..
del /q /s .\.docs\modules
del /q /s .\docs
poetry run sphinx-apidoc -f -o .\.docs\modules .\pyfemtet
poetry run .\.docs\make html
