import winreg
from typing import Final


__all__ = [
    '_get_autosave_enabled',
    '_set_autosave_enabled',
]


# レジストリのパスと値の名前
_REGISTRY_PATH: Final[str] = r"SOFTWARE\Murata Software\Femtet2014\Femtet"
_VALUE_NAME: Final[str] = "AutoSave"
_DEFAULT_VALUE: Final[bool] = True


def _get_autosave_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REGISTRY_PATH) as key:
            value, regtype = winreg.QueryValueEx(key, _VALUE_NAME)
            if regtype == winreg.REG_DWORD:
                return bool(value)
            else:
                raise ValueError("Unexpected registry value type.")

    except FileNotFoundError:  # [WinError 2] 指定されたファイルが見つかりません。
        __create_registry_key()
        return _get_autosave_enabled()


def _set_autosave_enabled(enable: bool) -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REGISTRY_PATH, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_DWORD, int(enable))

    except FileNotFoundError:  # [WinError 2] 指定されたファイルが見つかりません。
        __create_registry_key()
        _set_autosave_enabled(enable)


def __test_autosave_setting():

    # 使用例
    current_setting = _get_autosave_enabled()
    print(f"Current AutoSave setting is {'enabled' if current_setting else 'disabled'}.")

    # 設定を変更する例
    new_setting = not current_setting
    _set_autosave_enabled(new_setting)
    print(f"AutoSave setting has been {'enabled' if new_setting else 'disabled'}.")

    # 再度設定を確認する
    after_setting = _get_autosave_enabled()
    print(f"Current AutoSave setting is {'enabled' if after_setting else 'disabled'}.")

    assert new_setting == after_setting, "レジストリ編集に失敗しました。"


def __create_registry_key():
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _REGISTRY_PATH) as key:
        winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_DWORD, int(_DEFAULT_VALUE))


if __name__ == '__main__':
    print(_get_autosave_enabled())
