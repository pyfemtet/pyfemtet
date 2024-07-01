# ===== DPI setting =====
using namespace System.Windows.Forms
using namespace System.Drawing

Add-Type -Assembly System.Windows.Forms

#Enable visual styles
[Application]::EnableVisualStyles()

#Enable DPI awareness
$code = @"
    [System.Runtime.InteropServices.DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
"@
$Win32Helpers = Add-Type -MemberDefinition $code -Name "Win32Helpers" -PassThru
$null = $Win32Helpers::SetProcessDPIAware()


# ===== python command =====
$python_command = "py"


# ===== pre-requirement =====

# --runas (reload)
if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrators")) { Start-Process powershell.exe "-File `"$PSCommandPath`"" -Verb RunAs; exit }

# error setting
$ErrorActionPreference = "Stop"  # "Inquire" for debug.

# message
$res = [System.Windows.Forms.MessageBox]::Show(
    "Femtet (2023.0 or later) and Python (3.11 or later, less than 3.13) are required. Please make sure they are installed before continuing.",
    "pre-request",
    [System.Windows.Forms.MessageBoxButtons]::YesNo
)
if ($res -eq [System.Windows.Forms.DialogResult]::No) {
    throw 'Cancel the process.'
}


# ===== main =====

write-host
write-host "======================"
write-host "installing pyfemtet..."
write-host "======================"
# install pyfemtet-opt-gui
& $python_command -m pip install pyfemtet-opt-gui -U --no-warn-script-location

# check pyfemtet-opt-gui installed
$installed_packages = & $python_command -m pip list
if (-not ($installed_packages | Select-String -Pattern 'pyfemtet-opt-gui')) {
    $title = "Error!"
    $message = "PyFemtet installation failed."
    [System.Windows.Forms.MessageBox]::Show($message, $title)
    throw $message
} else {
    write-host "Install pyfemtet OK"
}


write-host
write-host "========================="
write-host "Enabling Python Macro ..."
write-host "========================="
# get Femtet.exe path using femtetuitls
$femtet_exe_path = & $python_command -c "from femtetutils import util;from logging import disable;disable();print(util.get_femtet_exe_path())"
# estimate FemtetMacro.dll path
$femtet_macro_dll_path = $femtet_exe_path.replace("Femtet.exe", "FemtetMacro.dll")
if (-not (test-path $femtet_macro_dll_path)) {
    $title = "Error!"
    $message = "Femtet macro interface configuration failed. Please refer to Femtet macro help and follow the Enable macros procedure."
    [System.Windows.Forms.MessageBox]::Show($message, $title)
    throw $message
} else {
    write-host "regsvr32 will be OK"
}

# regsvr
regsvr32 $femtet_macro_dll_path  # returns nothing, dialog only


write-host
write-host "========================"
write-host "COM constants setting..."
write-host "========================"
# win32com.client.makepy
& $python_command -m win32com.client.makepy FemtetMacro  # return nothing

# check COM constants setting
$SOLVER_NON_C = & $python_command -c "from win32com.client import Dispatch, constants;Dispatch('FemtetMacro.Femtet');print(constants.SOLVER_NONE_C)"
if ($SOLVER_NON_C -eq "0") {
    write-host "COM constants setting: OK"
} else{
    $title = "warning"
    $message = "PyFmetet installation completed, but COM constant setting failed."
    $message += "Please execute the following command at the command prompt."
    $message += "\n py -m win32com.client.makepy FemtetMacro"
    [System.Windows.Forms.MessageBox]::Show($message, $title)
}

write-host
write-host "==========================="
write-host "Create Desktop shortcuts..."
write-host "==========================="
# make desktop shortcut for pyfemtet-opt-gui
$pyfemtet_package_path = & $python_command -c "import pyfemtet;print(pyfemtet.__file__)"
$pyfemtet_opt_script_builder_path = $pyfemtet_package_path.replace("Lib\site-packages\pyfemtet\__init__.py", "Scripts\pyfemtet-opt.exe")
$pyfemtet_opt_result_viewer_path = $pyfemtet_package_path.replace("Lib\site-packages\pyfemtet\__init__.py", "Scripts\pyfemtet-opt-result-viewer.exe")

$succeed = $true
$succeed = (test-path $pyfemtet_opt_script_builder_path) -and (test-path $pyfemtet_opt_result_viewer_path)

if ($succeed) {
    # create desktop shortcut of pyfemtet-opt.exe in $Scripts_dir folder
    try {
        $Shortcut_file = "$env:USERPROFILE\Desktop\pyfemtet-opt.lnk"
        $WScriptShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WScriptShell.CreateShortcut($Shortcut_file)
        $Shortcut.TargetPath = $pyfemtet_opt_script_builder_path
        $Shortcut.Save()
        write-host "Shortcut of script_builder is created."
    }
    catch {
        $succeed = $false
    }
    # plan to add 0.4.9
    # try {
    #     $Shortcut_file = "$env:USERPROFILE\Desktop\pyfemtet-opt-result-viewer.lnk"
    #     $WScriptShell = New-Object -ComObject WScript.Shell
    #     $Shortcut = $WScriptShell.CreateShortcut($Shortcut_file)
    #     $Shortcut.TargetPath = $pyfemtet_opt_result_viewer_path
    #     $Shortcut.Save()
    #     write-host "Shortcut of result_viewer is created."
    # }
    # catch {
    #     $succeed = $false
    # }
}

if ($succeed) {
    [System.Windows.Forms.MessageBox]::Show("PyFemtet installation and setup is complete.", "Complete!")
} else {
    $title = "warning"
    $message = "PyFmetet installation completed, but creation of desktop shortcut failed. pyfemtet-opt.exe does not exist in the Scripts folder of the Python execution environment."
    [System.Windows.Forms.MessageBox]::Show($message, $title)

}
