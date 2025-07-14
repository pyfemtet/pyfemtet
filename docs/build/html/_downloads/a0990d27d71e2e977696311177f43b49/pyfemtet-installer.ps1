# ===== DPI setting =====
using namespace System.Windows.Forms
using namespace System.Drawing

# error setting
$ErrorActionPreference = "Stop"  # "Inquire" for debug.

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
$python_command = "py"  # Comment out this line if you don't use py launcher
# $python_command = "python"  # And uncomment this line
# ==========================



$python_command_parts = $python_command.Split()
$py = $python_command_parts[0]
$py_base_args = $python_command_parts[1..($python_command_parts.Length)]


# ===== Admin check =====
if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrators")) {
    [System.Windows.Forms.MessageBox]::Show("Please run as administrator", "pre-request")
    throw 'Please run as administrator'
}


# ===== pre-requirement =====
# check Femtet, Excel and Python process existing
$excel_exists = $null -ne (Get-Process Excel -ErrorAction SilentlyContinue)
$femtet_exists = $null -ne (Get-Process Femtet -ErrorAction SilentlyContinue)
$python_exists = $null -ne (Get-Process Python -ErrorAction SilentlyContinue)

if ($excel_exists -or $femtet_exists) {
    # Unable to continue
    $msg = "Femtet or Excel processes exist. Please terminate them before install.`r`n`r`nInstallation canceled."
    $res = [System.Windows.Forms.MessageBox]::Show(
        $msg,
        "error"
    )
    throw 'Cancel the process.'
}
elseif ($python_exists) {
    # May be unable to continue
    $msg = "Python processes exist. "
    $msg += "Please certify that no Python processes are using COM (Typically, Femtet Macro or Office Macro)."
}
else {
    # No problem
    $msg = "Femtet and Python are required. Please make sure they are installed before continuing."
}
$msg += "`r`n`r`nDo you continue installation?"

$res = [System.Windows.Forms.MessageBox]::Show(
    $msg,
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
$python_args = "-m pip install pyfemtet pyfemtet-opt-gui -U --no-warn-script-location"
$py_args = $py_base_args + $python_args.Split()
& $py @py_args

# check pyfemtet-opt-gui installed
$python_args = "-m pip list"
$py_args = $py_base_args + $python_args.Split()
$installed_packages = & $py @py_args
if (-not ($installed_packages | Select-String -Pattern 'pyfemtet-opt-gui')) {
    $title = "Error!"
    $message = "PyFemtet installation failed."
    [System.Windows.Forms.MessageBox]::Show($message, $title)
    throw $message
} else {
    write-host "Install pyfemtet OK"
}


write-host
write-host "================================"
write-host "Checking Femtet Installation ..."
write-host "================================"
$software_keys = Get-ChildItem "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" | Get-ItemProperty
$installed_software = $software_keys | Where-Object { $_.DisplayName }
$femtet_list = $installed_software | Where-Object { $_.DisplayName -like "*Femtet*" }
$femtet_list_sorted = $femtet_list | Sort-Object InstallDate
foreach ($femtet in $femtet_list_sorted) {
    $femtet_location = $femtet.InstallLocation
    $femtet_exe_path = Join-Path (Join-Path $femtet_location "Program") "Femtet.exe"
    if (test-path $femtet_exe_path) {
        break
    }
}
if (-not (test-path $femtet_exe_path)) {
    # existing Femtet.exe not found
    $title = "Error!"
    $message = "Installed Femtet.exe is not found."
    [System.Windows.Forms.MessageBox]::Show($message, $title)
    throw $message
}
write-host "found Femtet: " + $femtet_exe_path


write-host
write-host "========================="
write-host "Enabling Python Macro ..."
write-host "========================="
# estimate FemtetMacro.dll path
$femtet_macro_dll_path_64bit = $femtet_exe_path.replace("Femtet.exe", "FemtetMacro.dll")
$femtet_macro_dll_path_32bit = $femtet_exe_path.replace("Femtet.exe", (join-path "Macro32" "FemtetMacro.dll"))
if ((-not (test-path $femtet_macro_dll_path_64bit)) -or (-not (test-path $femtet_macro_dll_path_32bit))) {
    $title = "Error!"
    $message = "Femtet macro interface configuration failed. Please refer to Femtet macro help and follow the Enable macros procedure."
    [System.Windows.Forms.MessageBox]::Show($message, $title)
    throw $message
} else {
    write-host "Register typelib will be OK."
}
# regsvr
regsvr32 /s $femtet_macro_dll_path_64bit  # returns nothing, disable dialog by /s.
regsvr32 /s $femtet_macro_dll_path_32bit  # returns nothing, disable dialog by /s.


write-host
write-host "========================"
write-host "COM constants setting..."
write-host "========================"

# win32com.client.makepy
$python_args = "-m win32com.client.makepy FemtetMacro"
$py_args = $py_base_args + $python_args.Split()
& $py $py_args   # return nothing

# check COM constants setting
$py_args = $py_base_args + @("-c", """from win32com.client import Dispatch, constants;Dispatch('FemtetMacro.Femtet');print(constants.SOLVER_NONE_C)""")
$SOLVER_NON_C = & $py @py_args
if ($SOLVER_NON_C -eq "0") {
    write-host "COM constants setting: OK"
} else{
    $title = "warning"
    $message = "PyFmetet installation completed, but COM constant setting failed."
    $message += "Please execute the following command at the command prompt."
    $message += "`r`n`r`n py -m win32com.client.makepy FemtetMacro"
    [System.Windows.Forms.MessageBox]::Show($message, $title)
}

write-host
write-host "==========================="
write-host "Create Desktop shortcuts..."
write-host "==========================="
# make desktop shortcut for pyfemtet-opt-gui
$py_args = $py_base_args + @("-c", """import pyfemtet;print(pyfemtet.__file__)""")
$pyfemtet_package_path = & $py @py_args
$pyfemtet_opt_script_builder_path = $pyfemtet_package_path.replace("Lib\site-packages\pyfemtet\__init__.py", "Scripts\pyfemtet-opt.exe")
$pyfemtet_opt_result_viewer_path = $pyfemtet_package_path.replace("Lib\site-packages\pyfemtet\__init__.py", "Scripts\pyfemtet-opt-result-viewer.exe")

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
    try {
        $Shortcut_file = "$env:USERPROFILE\Desktop\pyfemtet-opt-result-viewer.lnk"
        $WScriptShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WScriptShell.CreateShortcut($Shortcut_file)
        $Shortcut.TargetPath = $pyfemtet_opt_result_viewer_path
        $Shortcut.Save()
        write-host "Shortcut of result_viewer is created."
    }
    catch {
        $succeed = $false
    }
}

if ($succeed) {
    [System.Windows.Forms.MessageBox]::Show("PyFemtet installation and setup is complete.", "Complete!")
} else {
    $title = "warning"
    $message = "PyFmetet installation completed, but creation of desktop shortcut failed. pyfemtet-opt.exe does not exist in the Scripts folder of the Python execution environment."
    [System.Windows.Forms.MessageBox]::Show($message, $title)
}
