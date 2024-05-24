$ErrorActionPreference = "Stop"  # stop if error

Set-Location $psscriptroot\..  # project root

$SOURCE_DIR = ".\docs\source"
$BUILD_DIR = ".\docs\build"
$HTML_DIR = ".\docs\build\html"
$DOCTREE_DIR = ".\docs\build\doctrees"
$HTML_DIR_JA = ".\docs\build\html_ja"
$DOCTREE_DIR_JA = ".\docs\build\doctrees_ja"
$SAMPLES_ON_DOC_SOURCE = ".\docs\source\examples\_temporary_sample_files"
$SAMPLES_SOURCE = ".\pyfemtet\opt\femprj_sample"
$SAMPLES_SOURCE_JP = ".\pyfemtet\opt\femprj_sample_jp"

if (test-path $BUILD_DIR) {
    remove-item $BUILD_DIR -recurse
    mkdir $BUILD_DIR | Out-Null
}

# copy English sample files to doc_source
if (Test-Path $SAMPLES_ON_DOC_SOURCE) {Remove-Item $SAMPLES_ON_DOC_SOURCE -Recurse -Force}
mkdir $SAMPLES_ON_DOC_SOURCE | Out-Null
Copy-Item -Path "$SAMPLES_SOURCE\*" -Destination $SAMPLES_ON_DOC_SOURCE -Recurse -Force

# build English document
poetry run python -m sphinx -T -b html -d $DOCTREE_DIR -D language=en $SOURCE_DIR $HTML_DIR

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

# build Japanese document
poetry run python -m sphinx -T -b html -d $DOCTREE_DIR_JA -D language=ja_JP $SOURCE_DIR $HTML_DIR_JA

pause
