import os
here, me = os.path.split(__file__)
import sys
sys.path.append(here)
mother_path = os.path.abspath(os.path.join(here, '..'))
sys.path.append(mother_path)

# plot ライブラリ
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 日付
import matplotlib.dates as mdates

# 日本語
import japanize_matplotlib

# dpi
import ctypes
scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)

# フォント設定
from matplotlib import rcParams
rcParams["font.size"] = 10.5
rcParams["figure.dpi"] = scaleFactor
# rcParams['font.family'] = 'MS Gothic'

# スタイル
plt.style.use("ggplot")

# # 3D
# from mpl_toolkits.mplot3d import Axes3D

# # clipboard
# import io
# from PyQt5.QtGui import QImage
# from PyQt5.QtWidgets import QApplication






# ax.set_picker('scroll_event')

#     for ax in figure.axes:
#         children = ax.get_children()
#         for child in children:
#             if type(child)==PathCollection:
#                 child.set_picker('pick_event')

#     def on_pick_PathCollection(event):
#         if type(event.artist)==PathCollection:
#             # xlabel, xの値、 ylabel, yの値の取得
#             xlabel = event.artist.axes.get_xlabel()
#             ylabel = event.artist.axes.get_ylabel()
#             xdata, ydata = event.artist.get_offsets()[event.ind[0]]
#             # history を検索
#             idx = (history[xlabel]==xdata) * (history[ylabel]==ydata)
#             print()
#             for c in history.columns:
#                 print(c, history[idx][c].values[0])
#             print()

#     figure.canvas.mpl_connect('pick_event', on_pick_PathCollection)







class SimpleProcessMonitor:
    def __init__(self, FEMOpt):
        # FEMOpt を貰う
        self.FEMOpt = FEMOpt
        self.N = len(self.FEMOpt.objectives)
        # サブプロットを定義
        self.fig, axes = plt.subplots(self.N, 1, sharex=True)
        self.fig.suptitle('シンプル目的関数モニター')
        # ax を作って ilne と一緒に保持する
        self.lines = []
        for i in range(self.N):
            ax = self.fig.axes[i]
            line, = ax.plot([], [], 'o-')
            subLine = ax.axhline(y=0, color='red', linestyle='--', lw=1)
            self.lines.append([line, subLine])
            # スクロールした時に direction を移動させる関数を登録するためにイベントピッカーを登録する
            ax.set_picker('draw_event')
        # 登録したイベントピッカーを接続する
        self.fig.canvas.mpl_connect('draw_event', self.adjustDirectionPosition)

        
    def update(self):
        objectives = self.FEMOpt.objectives
        xdata = self.FEMOpt.history['time']
        for ax, (line, subLine), objective in zip(self.fig.axes, self.lines, objectives):
            # 描画
            ydata = self.FEMOpt.history[objective.name].values
            line.set_data(xdata, ydata)
            subLine.set_ydata(ydata[-1]) # 再設定の時前の値にひきずられるから
            # lim 再設定
            ax.set_ylabel(objective.name)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            ax.relim()
            ax.autoscale_view(True, True, True)
            # ターゲット線の描画(get_limするから後で)
            minimum, maximum = ax.get_ylim()
            if objective.direction=='maximize':
                y = maximum - (maximum - minimum) * 0.01
            elif objective.direction=='minimize':
                y = minimum + (maximum - minimum) * 0.01
            else: # 指定値と見做す
                y = objective.direction
            subLine.set_ydata(float(y))
        plt.pause(0.001)
        
    def adjustDirectionPosition(self, _):
        objectives = self.FEMOpt.objectives
        for ax, (line, subLine), objective in zip(self.fig.axes, self.lines, objectives):
            # ターゲット線の描画(get_limするから後で)
            minimum, maximum = ax.get_ylim()
            if objective.direction=='maximize':
                y = maximum - (maximum - minimum) * 0.01
            elif objective.direction=='minimize':
                y = minimum + (maximum - minimum) * 0.01
            else: # 指定値と見做す
                y = objective.direction
            subLine.set_ydata(float(y))
