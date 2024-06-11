Set-Location $PSScriptRoot\..
$yamlFolder = ".\test_results"
$yamlBaseName = Get-Date -Format "yyyyMMdd_HHmmss"
$yamlPath = Join-Path $yamlFolder $yamlBaseName".yaml"

write-host $yamlPath

poetry run pytest ./tests -s -k "sample" --yaml-path=$yamlPath
pause
exit

# all(with stdout)
# poetry run pytest ./tests -s

# last failed only
# poetry run pytest ./tests -s --lf

# (un)match function only
# poetry run pytest ./tests -s -k "not _cad and not _sample"
