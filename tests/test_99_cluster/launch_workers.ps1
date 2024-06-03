Param(
    [String]$hostname,
    [String]$port,
    [String]$nworkers
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..\..
poetry run python -m dask worker ${hostname}:$port --nthreads 1 --nworkers $nworkers --no-dashboard
