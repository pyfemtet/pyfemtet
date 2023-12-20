cd %~dp0
del /q /s .\.docs\modules
del /q /s .\docs
poetry run .docs\make html