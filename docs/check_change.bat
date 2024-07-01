set /p filepath=Enter the file path:
git diff --unified=0 "%filepath%" | findstr "^+msgid" > result.txt