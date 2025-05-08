"""Excel のエラーダイアログを補足します。"""
from time import sleep
from threading import Thread
import asyncio  # for timeout
import win32gui
import win32con
import win32api
import win32process

from pyfemtet.logger import get_module_logger


__all__ = [
    'watch_excel_macro_error'
]


logger = get_module_logger('util.excel', False)


def _get_pid(hwnd):
    """Window handle から process ID を取得します."""
    if hwnd > 0:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
    else:
        pid = 0
    return pid


class _ExcelDialogProcessor:

    def __init__(self, excel_, timeout, restore_book=True):
        self.excel = excel_
        self.excel_pid = _get_pid(excel_.hWnd)
        self.__excel_window_title = f' - Excel'  # {basename} - Excel
        self.__error_dialog_title = 'Microsoft Visual Basic'
        self.__vbe_window_title = f'Microsoft Visual Basic for Applications - '  # Microsoft Visual Basic for Applications - {basename}
        self.should_stop = False
        self.timeout = timeout
        self.__timed_out = False
        self.__workbook_paths = [wb.FullName for wb in excel_.Workbooks]
        self.__error_raised = False
        self.__excel_state_stash = dict()
        self.__watch_thread = None
        self.restore_book = restore_book

    async def watch(self):

        while True:
            if self.should_stop:
                logger.debug('エラーダイアログの監視を終了')
                return
            logger.debug('エラーダイアログを監視中...')

            win32gui.EnumWindows(self.enum_callback_to_activate, [])
            await asyncio.sleep(0.5)

            found = []
            win32gui.EnumWindows(self.enum_callback_to_close_dialog, found)
            await asyncio.sleep(0.5)
            if any(found):
                await asyncio.sleep(1.)
                break

        logger.debug('ブックを閉じます。')
        win32gui.EnumWindows(self.enum_callback_to_close_book, [])  # 成功していればこの時点でメイン処理では例外がスローされる
        await asyncio.sleep(1)

        logger.debug('確認ダイアログがあれば閉じます。')
        win32gui.EnumWindows(self.enum_callback_to_close_confirm_dialog, [])
        await asyncio.sleep(1)
        self.__error_raised = True

    def enum_callback_to_activate(self, hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        # Excel 本体
        if (self.excel_pid == _get_pid(hwnd)) and (self.__excel_window_title in title):
            # Visible == True の際、エラーが発生した際、
            # 一度 Excel ウィンドウをアクティブ化しないと dialog が出てこない
            # が、これだけではダメかも。
            win32gui.PostMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)

    def enum_callback_to_close_dialog(self, hwnd, found):
        title = win32gui.GetWindowText(hwnd)
        # エラーダイアログ
        if (self.excel_pid == _get_pid(hwnd)) and (self.__error_dialog_title == title):
            # 何故かこのコマンド以外受け付けず、
            # このコマンドで問答無用でデバッグモードに入る
            logger.debug('エラーダイアログを見つけました。')
            win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
            win32api.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
            logger.debug('エラーダイアログを閉じました。')
            found.append(True)

    def enum_callback_to_close_confirm_dialog(self, hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        # 確認ダイアログ
        if (self.excel_pid == _get_pid(hwnd)) and ("Microsoft Excel" in title):
            # DisplayAlerts が False の場合は不要
            win32gui.SendMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_CLOSE, 0)

    def enum_callback_to_close_book(self, hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        # VBE
        if (self.excel_pid == _get_pid(hwnd)) and (self.__vbe_window_title in title):
            # 何故かこれで book 本体が閉じる
            win32gui.SendMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    async def watch_main(self):
        try:
            await asyncio.wait_for(self.watch(), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.debug('タイムアウトしました。')
            self.should_stop = True
            self.__timed_out = True

    def __enter__(self):
        logger.debug('Excel を 不可視にします。')
        self.__excel_state_stash['visible'] = self.excel.Visible
        self.__excel_state_stash['display_alerts'] = self.excel.DisplayAlerts
        self.excel.Visible = False
        self.excel.DisplayAlerts = False

        logger.debug('エラー監視を開始します。')
        self.__watch_thread = Thread(
            target=asyncio.run,
            args=(self.watch_main(),),
        )
        self.__watch_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):

        logger.debug('Excel の状態を回復します。')

        self.should_stop = True
        self.__watch_thread.join()

        self.excel.Visible = self.__excel_state_stash['visible']
        self.excel.DisplayAlerts = self.__excel_state_stash['display_alerts']

        if self.__timed_out:
            logger.debug('Excel プロセスを強制終了します。')
            logger.error('Excel プロセス強制終了は未実装です。')
            raise TimeoutError('マクロの実行がタイムアウトしました。')

        # if exc_type is not None:
        #     if issubclass(exc_type, com_error) and self.__error_raised:
        if self.__error_raised:
            if self.restore_book:
                logger.debug('エラーハンドリングの副作用でブックを閉じているので'
                             'Excel のブックを開きなおします。')
                for wb_path in self.__workbook_paths:
                    self.excel.Workbooks.Open(wb_path)


def watch_excel_macro_error(excel_, timeout, restore_book=True):
    """Excel のエラーダイアログの出現を監視し、検出されればブックを閉じます。"""
    return _ExcelDialogProcessor(excel_, timeout, restore_book)


if __name__ == '__main__':

    import os
    os.chdir(os.path.dirname(__file__))

    path = os.path.abspath('sample.xlsm')
    path2 = os.path.abspath('sample2.xlsm')

    from win32com.client import Dispatch
    # noinspection PyUnresolvedReferences
    from pythoncom import com_error

    logger.debug('Excel を起動しています。')
    excel = Dispatch('Excel.Application')
    excel.Visible = True
    excel.DisplayAlerts = False
    excel.Interactive = True

    logger.debug('Workbook を開いています。')
    excel.Workbooks.Open(path)

    logger.debug('別の Workbook を開いています。')
    excel.Workbooks.Open(path2)

    logger.debug('Workbook に変更を加えます。')
    excel.Workbooks(1).ActiveSheet.Range('A1').Value = 1.

    logger.debug('開いている Workbooks 数：')
    logger.debug(excel.Workbooks.Count)

    try:
        with watch_excel_macro_error(excel, timeout=10):
            sleep(3)
            try:
                excel.Run('raise_error')
            except com_error as e:
                logger.debug('この段階ではまだ Excel 回復機能が働きません。')
                logger.debug('開いている Workbooks 数：')
                logger.debug(excel.Workbooks.Count)
                raise e

            # excel.Run('no_error')

    except com_error as e:
        logger.debug('メイン処理でエラーを補足しました。：')
        logger.debug(e)

    logger.debug('開いている Workbooks 数：')
    logger.debug(excel.Workbooks.Count)

    logger.debug('保存していない場合、Workbook の変更は失われます。')
