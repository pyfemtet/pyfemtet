# プログラムの起動時の引数に応じて locale フォルダ内のフォルダを移動する。
# 前提フォルダ構造
# - apply_locale.ps1
# - locale/
#   - en/
#     - <複数の対象ファイル.png>
#   - ja/
#     - <複数の対象ファイル.png>
# - <複数の対象ファイル.png>
# 処理
# 例えば引数が en であれば、 locale/en/<複数の対象ファイル.png> を <複数の対象ファイル.png> にコピーする。
#
# 使い方:
#   PowerShell でこのスクリプトがあるディレクトリに移動し、以下のように実行してください。
#   例: jaロケールの画像を適用する場合
#     powershell -ExecutionPolicy Bypass -File .\apply_locale.ps1 ja
#   ※ロケール名は locale フォルダ内のサブフォルダ名に合わせてください。

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$locale
)

$scriptDir = $PSScriptRoot
$localeDir = Join-Path $scriptDir "locale"
$targetDir = $scriptDir
$srcDir = Join-Path $localeDir $locale

if (-not (Test-Path $srcDir)) {
    Write-Error "指定されたロケール '$locale' のフォルダが存在しません: $srcDir"
    exit 1
}

# 画像ファイル（.png）をすべてコピー
Get-ChildItem -Path $srcDir -Filter *.png | ForEach-Object {
    $srcFile = $_.FullName
    $destFile = Join-Path $targetDir $_.Name
    Copy-Item -Path $srcFile -Destination $destFile -Force
    Write-Host "Copied: $srcFile -> $destFile"
}
