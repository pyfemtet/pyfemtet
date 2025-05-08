# ===== settings =====
Param(
    [bool]$is_JP=$false
)
$ErrorActionPreference = "Stop"


# ===== const =====
$package_root = "$PSScriptRoot\..\..\..\..\"
$sample_location = "$package_root\samples\opt\advanced_samples\excel_ui"


# ===== helper =====
function copy_file([string]$file_name_, [string]$ext_, [boolean]$is_common_ = $false){
    if ($is_JP -and -not($is_common_)) {
        # write-host "JP"
        $file = Get-Item ("$sample_location\$file_name_" + "_jp" + "$ext_")
    } else {
        # write-host "EN"
        $file = Get-Item ("$sample_location\$file_name_" + "$ext_")
    }
    $target_path = "$PSScriptRoot\$file_name_" + "$ext_"
    write-host "copy " $file
    write-host "->as " $target_path
    Copy-Item -Path $file.FullName -Destination $target_path -Force
}


# ===== processing =====
copy_file "femtet-macro" ".xlsm"              $true  # no en file
copy_file "pyfemtet-core" ".py"               $true  # no en file
copy_file "(ref) original_project" ".femprj"  $true  # no en file
