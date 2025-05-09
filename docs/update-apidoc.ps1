$ErrorActionPreference = "Stop"  # stop if error

Set-Location $psscriptroot\..  # project root

$SOURCE_DIR = ".\docs\source"
$HTML_DIR = ".\docs\build\html"
$DOCTREE_DIR = ".\docs\build\doctrees"

# update api references
if (Test-Path "docs/source/modules") {Remove-Item "docs/source/modules" -Force -Recurse}
uv run sphinx-apidoc --force --no-toc --separate -d=1 -o="docs/source/modules" pyfemtet

# build English document
uv run python -m sphinx -T -b html -d $DOCTREE_DIR -D language=en $SOURCE_DIR $HTML_DIR
