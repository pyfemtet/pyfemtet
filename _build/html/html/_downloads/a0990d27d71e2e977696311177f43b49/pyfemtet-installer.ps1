﻿# ===== DPI setting =====
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



# ===== pre-requirement =====
if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrators")) {
    # check Femtet, Excel and Python process existing
    $excel_exists = $null -ne (Get-Process Excel -ErrorAction SilentlyContinue)
    $femtet_exists = $null -ne (Get-Process Femtet -ErrorAction SilentlyContinue)
    $python_exists = $null -ne (Get-Process Python -ErrorAction SilentlyContinue)

    if ($excel_exists -or $femtet_exists) {
        $msg = "Femtet 又は Excel のプロセスが存在します。PyFemtet にインストールおよびセットアップ前にそれらのプロセスを停止してください。`r`n`r`nインストールはキャンセルされます。"
        $res = [System.Windows.Forms.MessageBox]::Show(
            $msg,
            "error"
        )
        throw 'Cancel the process.'
    }

    if (-not $python_exists) {
        # OK
        $msg = "Femtet (2023.0 or later) と Python (3.11 or later, less than 3.13) が必要です。続行する前にこれらがインストールされていることを確認してください。"
    } elseif ($excel_exists -or $femtet_exists) {
        # NG
        $msg = "Python プロセスが存在します。"
        $msg += "Python プロセスが COM 機能（典型的には、Femtet マクロや Excel マクロ）を使用していないことを確認してください。"
    }
    $msg += "`r`n`r`nインストールを続行しますか？?"

    $res = [System.Windows.Forms.MessageBox]::Show(
        $msg,
        "pre-request",
        [System.Windows.Forms.MessageBoxButtons]::YesNo
    )
    if ($res -eq [System.Windows.Forms.DialogResult]::No) {
        throw 'Cancel the process.'
    }
}


# --runas (reload after prerequest)
if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrators")) {
    Start-Process powershell.exe "-File `"$PSCommandPath`"" -Verb RunAs
    exit
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
    $message = "PyFemtet のインストールに失敗しました。"
    [System.Windows.Forms.MessageBox]::Show($message, $title)
    throw $message
} else {
    write-host "Install pyfemtet OK"
}


write-host
write-host "========================="
write-host "Checking Femtet Installation ..."
write-host "========================="
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
    $message = "Femtet.exe がインストールされていることを検出できませんでした。処理を中断します。"
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
    $message = "Femtet マクロ有効化設定に失敗しました。Femtet マクロヘルプを参照し「マクロの有効化」手順を実行してください。"
    [System.Windows.Forms.MessageBox]::Show($message, $title)
    throw $message
} else {
    write-host "Register typelib will be OK. Please check regsvr32 dialog to certify."
}
# regsvr
regsvr32 $femtet_macro_dll_path_64bit  # returns nothing, dialog only.
regsvr32 $femtet_macro_dll_path_32bit  # returns nothing, dialog only. No require to run on WOW64 dir


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
    $message = "PyFmetet のインストールは完了しましたが、COM 定数設定に失敗しました。"
    $message += "コマンドプロンプトで下記のコマンドを実行してください。"
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
    [System.Windows.Forms.MessageBox]::Show("PyFemtet インストールとセットアップが完了しました。", "Complete!")
} else {
    $title = "warning"
    $message = "PyFmetet のインストールは完了しましたが、デスクトップにショートカットを作成できませんでした。Python 実行環境の Scripts フォルダ内の pyfemtet-opt.exe が見つかりません。"
    [System.Windows.Forms.MessageBox]::Show($message, $title)

}
