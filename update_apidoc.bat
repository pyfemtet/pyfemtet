cd %~dp0
del /q /s .\docs\modules
del /q /s .\docs\_build
poetry run docs\make html