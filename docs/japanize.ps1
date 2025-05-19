$ErrorActionPreference = "Stop"  # stop if error

Set-Location $psscriptroot/..  # project root

$SOURCE_DIR = "./docs/source"
$LOCALE_DIR = "./docs/source/locale"
$GETTEXT_DIR = "./docs/gettext"

uv run --no-sync sphinx-apidoc -f -o $SOURCE_DIR\modules .\pyfemtet

if (-not (test-path $LOCALE_DIR)) {mkdir $LOCALE_DIR}

uv run --no-sync sphinx-build -b gettext $SOURCE_DIR $GETTEXT_DIR
uv run --no-sync sphinx-intl update -p $GETTEXT_DIR -l ja_JP -d $LOCALE_DIR --line-width=-1

write-host "$LOCALE_DIR  Translate msgid section of .po file to msgstr section."
pause
