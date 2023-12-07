from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QApplication
from PySide6.QtCore import QTimer, Qt
from typing import Callable, Any
from multiprocessing import Value # 型ヒントのためのみ
import sys
import webbrowser


def open_browser():
    webbrowser.open('http://localhost:8080')


class SimplestDialog(QDialog):
    def __init__(
            self,
            FEMOpt,
            interrput_flag:Value,
            get_close_flag:Callable[[], bool] or None = None,
            fun_to_update:[Callable[[], Any]] or None = None,
            ):
        self.FEMOpt = FEMOpt
        
        # UI の設定
        super().__init__()
        self.setWindowTitle("最適化")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        self.layout = QVBoxLayout()

        self.label = QLabel("経過時間：0秒")
        self.layout.addWidget(self.label)

        self.button = QPushButton("最適化を中断")
        self.button.clicked.connect(self.on_button_clicked)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

        # 1 秒ごとに更新する関数
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_label)
        if fun_to_update is not None:
            for fun in fun_to_update:
                self.timer.timeout.connect(fun)
        self.timer.start(1000)
        
        # 終了判定
        self.get_close_flag = get_close_flag

        # 経過時間
        self.seconds = 0

    def update_label(self):
        self.seconds += 1

        # 中断フラグが立ってなければ経過時間を表示
        if self.FEMOpt.shared_interruption_flag.value==0:
            self.label.setText(f"経過時間：{self.seconds}秒")

        # そうでなければ終了中ですと表示
        else:
            text = '''終了中です。
  現在の計算がすべて終了し、関連する非表示の
  プロセスがすべて終了するとこのダイアログが閉じます。'''
            self.label.setText(text)

        # 終了判定
        if self.get_close_flag is not None:
            if self.get_close_flag():
                self.close()


    def on_button_clicked(self):
        self.button.setEnabled(False)
        self.FEMOpt.shared_interruption_flag.value = 1

    def closeEvent(self, event):
        super().closeEvent(event)

if __name__ == "__main__":
    # app = QApplication(sys.argv)
    app = QApplication.instance()
    if app == None:
        app = QApplication([])
    dialog = SimplestDialog()
    dialog.show()
    app.exec()
    # del dialog
    # del app
    # import gc
    # gc.collect()
