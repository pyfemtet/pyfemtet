# Error License(s. delemiter is ;)
$licenses_to_avoid = "UNKNOWN; GPL"

# Refresh
write-host "===== init ====="
$result_path = "$PSScriptRoot\result"
if (test-path $result_path) {
    Remove-Item $result_path -Recurse -Force > $null
}
New-Item -Path $result_path -ItemType Directory > $null

# Batch check
write-host "===== batch-checking ====="
$check_result_path = "$PSScriptRoot\result\batch_check_result.txt"
poetry run pip-licenses --format=markdown --fail-on=$licenses_to_avoid > $check_result_path
if ((Get-Item $check_result_path).Length -eq 0) {
    Write-Output "Check failed! Some packages contain $licenses_to_avoid." >> $check_result_path
} else {
    Write-Output "Check succeed! Please re-check the summary files too." >> $check_result_path
}

# Summary to check manually
write-host "===== creating summary ====="
poetry run pip-licenses --format=markdown > $PSScriptRoot\result\_OSS-LICENSES-Summary.md
poetry run pip-licenses --format=csv --with-urls > $PSScriptRoot\result\_OSS-LICENSES-Summary.csv

# OSS-LICENSES.txt
write-host "===== creating output ====="
poetry run pip-licenses --format=plain-vertical --no-license-path --with-license-file > $PSScriptRoot\result\OSS-LICENSES.txt
