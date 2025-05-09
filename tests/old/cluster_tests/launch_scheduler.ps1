Param(
    [String]$port
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..\..
uv run python -m dask scheduler --host 0.0.0.0 --port $port --no-dashboard --no-show
