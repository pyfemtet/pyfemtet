$ErrorActionPreference = "Stop"  # stop if error

Set-Location $psscriptroot\..  # project root

$SOURCE_DIR = ".\docs\source"
$BUILD_DIR = ".\docs\build"
$HTML_DIR = ".\docs\build\html"
$DOCTREE_DIR = ".\docs\build\doctrees"
$HTML_DIR_JA = ".\docs\build\html_ja"
$DOCTREE_DIR_JA = ".\docs\build\doctrees_ja"

if (test-path $BUILD_DIR) {
    remove-item $BUILD_DIR -recurse
    mkdir $BUILD_DIR
}

poetry run python -m sphinx -T -b html -d $DOCTREE_DIR -D language=en $SOURCE_DIR $HTML_DIR
poetry run python -m sphinx -T -b html -d $DOCTREE_DIR_JA -D language=ja_JP $SOURCE_DIR $HTML_DIR_JA

pause
