# ===== settings =====
Param(
    [bool]$is_JP=$false
)
$ErrorActionPreference = "Stop"


# ===== const =====
$package_root = "$PSScriptRoot\..\..\..\..\"
$sample_location = "$package_root\samples\opt\experimental_samples\topology_matching"


# ===== helper =====
function copy_file([string]$file_name_, [string]$ext_, [boolean]$is_common_ = $false){
    if ($is_JP -and -not($is_common_)) {
        # write-host "JP"
        $file = Get-Item ("$sample_location\$file_name_" + "_jp" + "$ext_")
    } else {
        # write-host "EN"
        $file = Get-Item ("$sample_location\$file_name_" + "$ext_")
    }
    # copy file to the folder where this script is located
    $target_path = "$PSScriptRoot\$file_name_" + "$ext_"
    write-host "copy " $file
    write-host "->as " $target_path
    Copy-Item -Path $file.FullName -Destination $target_path -Force
}

copy_file "effect_of_topology_matching"       ".pdf"     $true
copy_file "topology_matching"                 ".py"
copy_file "topology_matching"                 ".femprj"     $true
