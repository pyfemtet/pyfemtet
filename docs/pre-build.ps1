$ErrorActionPreference = "Stop"  # stop if error

Set-Location $psscriptroot\..  # project root

$SOURCE_DIR = ".\docs\source"
$BUILD_DIR = ".\docs\build"
$HTML_DIR = ".\docs\build\html"
$DOCTREE_DIR = ".\docs\build\doctrees"
$HTML_DIR_JA = ".\docs\build\html_ja"
$DOCTREE_DIR_JA = ".\docs\build\doctrees_ja"
$SAMPLES_ON_DOC_SOURCE = ".\docs\source\examples\_temporary_sample_files"
$SAMPLES_SOURCE = ".\samples\opt\femprj_samples"
$SAMPLES_SOURCE_JP = ".\samples\opt\femprj_samples_jp"
$INSTALLER = ".\docs\installer\pyfemtet-installer.ps1"
$INSTALLER_JP = ".\docs\installer\pyfemtet-installer-jp.ps1"
$INSTALLER_RUNNER = ".\docs\installer\pyfemtet-installer-runner.bat"
$INSTALLER_RUNNER_JP = ".\docs\installer\pyfemtet-installer-runner-jp.bat"
$INSTALLER_ON_DOC_SOURCE = ".\docs\source\pyfemtet-installer.ps1"
$INSTALLER_RUNNER_ON_DOC_SOURCE = ".\docs\source\pyfemtet-installer-runner.bat"

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
Copy-Item -Path $INSTALLER_RUNNER -Destination $INSTALLER_RUNNER_ON_DOC_SOURCE -Force

# update api references
if (Test-Path "docs/source/modules") {Remove-Item "docs/source/modules" -Force -Recurse}
uv run --no-sync sphinx-apidoc --force --no-toc --no-headings --separate -d=1 -o="docs/source/modules" pyfemtet

# setup other sample files
powershell "$SOURCE_DIR\advanced_examples\surrogate_model\copy_files.ps1"
powershell "$SOURCE_DIR\advanced_examples\excel_ui\copy_files.ps1"
powershell "$SOURCE_DIR\advanced_examples\restart\copy_files.ps1"

# build English document
uv run --no-sync python -m sphinx -T -b html -d $DOCTREE_DIR -D language=en $SOURCE_DIR $HTML_DIR

# copy and overwrite with Japanese sample files
Copy-Item -Path "$SAMPLES_SOURCE_JP\*" -Destination $SAMPLES_ON_DOC_SOURCE -Force -Recurse
$files = Get-ChildItem $SAMPLES_ON_DOC_SOURCE -Recurse
foreach ($file in $files) {
    if ($file.BaseName.EndsWith('_jp')) {
        $newName = $file.BaseName -replace '_jp$', ''
        $extension = $file.Extension
        $newPath = Join-Path $file.Directory.FullName ($newName + $extension)
        write-host "Replace $($newName + $extension) to jp file."
        if (Test-Path $newPath) {Remove-Item $newPath -Force}
        Rename-Item -Path $file.FullName -NewName ($newName + $extension) -Force
    }
}

# copy Japanese installer to doc_source
Copy-Item -Path $INSTALLER_JP -Destination $INSTALLER_ON_DOC_SOURCE -Force
Copy-Item -Path $INSTALLER_RUNNER_JP -Destination $INSTALLER_RUNNER_ON_DOC_SOURCE -Force

# update api references
uv run --no-sync sphinx-apidoc --force --no-toc --no-headings --separate -d=1 -o="docs/source/modules" pyfemtet

# setup other sample files
powershell "$SOURCE_DIR\advanced_examples\surrogate_model\copy_files.ps1" 1  # is_JP
powershell "$SOURCE_DIR\advanced_examples\excel_ui\copy_files.ps1" 1  # is_JP
powershell "$SOURCE_DIR\advanced_examples\restart\copy_files.ps1" 1  # is_JP

# build Japanese document
uv run --no-sync python -m sphinx -T -b html -d $DOCTREE_DIR_JA -D language=ja_JP $SOURCE_DIR $HTML_DIR_JA

pause
