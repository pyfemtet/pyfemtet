$ErrorActionPreference = "Stop"  # stop if error
Set-Location $psscriptroot  # here


$targets = @(".\build", ".\gettext", ".\source")

foreach ($target in $targets) {
    # remove from git
    git rm --cached -r $target

    # add to git
    git add $target
}

pause
