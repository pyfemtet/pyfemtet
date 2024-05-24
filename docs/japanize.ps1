$ErrorActionPreference = "Stop"  # stop if error

Set-Location $psscriptroot/..  # project root

$SOURCE_DIR = "./docs/source"
$LOCALE_DIR = "./docs/source/locale"
$GETTEXT_DIR = "./docs/gettext"

poetry run sphinx-apidoc -f -o $SOURCE_DIR\modules .\pyfemtet

if (-not (test-path $LOCALE_DIR)) {mkdir $LOCALE_DIR}

poetry run sphinx-build -b gettext $SOURCE_DIR $GETTEXT_DIR
poetry run sphinx-intl update -p $GETTEXT_DIR -l ja_JP -d $LOCALE_DIR --line-width=-1

write-host "$LOCALE_DIR  Translate msgid section of .po file to msgstr section."
pause
