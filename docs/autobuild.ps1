$ErrorActionPreference = "Stop"  # stop if error

Set-Location $psscriptroot\..  # project root

$SOURCE_DIR = ".\docs\source"
$BUILD_DIR = ".\docs\build"
$HTML_DIR = ".\docs\build\html"
$DOCTREE_DIR = ".\docs\build\doctrees"
$HTML_DIR_JA = ".\docs\build\html_ja"
$DOCTREE_DIR_JA = ".\docs\build\doctrees_ja"
$SAMPLES_ON_DOC_SOURCE = ".\docs\source\examples\_temporary_sample_files"
$SAMPLES_SOURCE = ".\pyfemtet\opt\samples\femprj_sample"
$SAMPLES_SOURCE_JP = ".\pyfemtet\opt\samples\femprj_sample_jp"
$INSTALLER = ".\pyfemtet-installer.ps1"
$INSTALLER_JP = ".\pyfemtet-installer-jp.ps1"
$INSTALLER_ON_DOC_SOURCE = ".\docs\source\pyfemtet-installer.ps1"

if (test-path $BUILD_DIR) {
    remove-item $BUILD_DIR -recurse
    mkdir $BUILD_DIR | Out-Null
}

# copy English sample files to doc_source
if (Test-Path $SAMPLES_ON_DOC_SOURCE) {Remove-Item $SAMPLES_ON_DOC_SOURCE -Recurse -Force}
mkdir $SAMPLES_ON_DOC_SOURCE | Out-Null
Copy-Item -Path "$SAMPLES_SOURCE\*" -Destination $SAMPLES_ON_DOC_SOURCE -Recurse -Force

# copy English installer to doc_source
Copy-Item -Path $INSTALLER -Destination $INSTALLER_ON_DOC_SOURCE -Force

start http://127.0.0.1:8000
uv run --no-sync sphinx-autobuild .\docs\source .\docs\build
