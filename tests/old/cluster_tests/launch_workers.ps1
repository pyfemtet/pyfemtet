Param(
    [String]$hostname,
    [String]$port,
    [String]$nworkers
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..\..
uv run python -m dask worker ${hostname}:$port --nthreads 1 --nworkers $nworkers --no-dashboard --death-timeout 600
