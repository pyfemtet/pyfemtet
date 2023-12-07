import numpy as np
import pandas as pd

# plot ライブラリ
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 日付
import matplotlib.dates as mdates

# dpi
import ctypes

scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)

# フォント設定
from matplotlib import rcParams

rcParams["font.size"] = 9
rcParams["figure.dpi"] = scaleFactor
# rcParams['font.family'] = 'MS Gothic'

# スタイル
plt.style.use("ggplot")

# 3D
from mpl_toolkits.mplot3d import Axes3D

# hint
from abc import ABC, abstractmethod
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D  # plot
from matplotlib.collections import PathCollection  # scatter
from matplotlib.text import Text  # title, suptitle, etc
# 日本語
from japanize_matplotlib import japanize

japanize()

# FEMOpt class
from ..core import FemtetOptimizationCore

default_supfigsize = (7, 3)  # size(px) = figsize*dpi
default_figsize = (5, 4)


class ConcreteClassOfUpdatableFigure:
    '''
    UpdatableFigureの具象クラス。
    インスタンスではなくクラスであることに注意してください。
    '''
    pass


class UpdatableSuperFigure:

    def __init__(
            self,
            FEMOpt: FemtetOptimizationCore,
            UpdatableFigure1: 'ConcreteClassOfUpdatableFigure',
            UpdatableFigure2: 'ConcreteClassOfUpdatableFigure',
    ):

        self.FEMOpt = FEMOpt
        self.monitors = []

        # figure の設定
        self.fig = plt.figure(figsize=default_supfigsize)
        self.fig.canvas.mpl_connect('button_press_event', self.force_redraw)
        self.fig.suptitle('画面をクリックして表示を更新；要素の選択はできません')

        # subfigure の配置
        gs = self.fig.add_gridspec(1, 2)
        # ひだり
        sub_fig1 = self.fig.add_subfigure(gs[0, 0])
        monitor1 = UpdatableFigure1(self.FEMOpt, sub_fig1)
        self.monitors.append(monitor1)
        # みぎ
        sub_fig2 = self.fig.add_subfigure(gs[0, 1])
        monitor2 = UpdatableFigure2(self.FEMOpt, sub_fig2)
        self.monitors.append(monitor2)

        # 閉じたときの処理はどれかに対して行えばいい、あるいはクラスメソッドにするか
        self.fig.canvas.mpl_connect('close_event', monitor1.on_close)

        # ※ matplotlib 3.8.2 では subfigure の中の artist に対して pick_event が効かない
        # 修正はすでにされており 23.11.18 にテストが終了していたので、
        # 近々 main ブランチにマージされるのではないか。
        # for m in self.monitors:
        #     self.fig.canvas.mpl_connect('pick_event', m.on_pick)

    def update(self):
        for m in self.monitors:
            m.update()
        if self.fig.canvas.figure.stale:
            self.fig.canvas.draw_idle()
        # self.fig.canvas.start_event_loop(1)

    def force_redraw(self, *args, **kwargs):
        self.fig.canvas.draw()  # 再描画されないことがある？
        plt.pause(1)


class UpdatableFigure(ABC):

    def __init__(
            self,
            FEMOpt: FemtetOptimizationCore,
            fig: SubFigure = None
    ):

        self.FEMOpt = FEMOpt
        self.labels = {}
        self.picked_idx = -1

        if fig is None:
            self.fig = plt.figure(figsize=default_figsize)
            self.fig.canvas.mpl_connect('pick_event', self.on_pick)
            self.fig.canvas.mpl_connect('close_event', self.on_close)
        else:
            # as subfigure
            self.fig = fig

    def _update_range(self, xrange, xdata):
        if len(xdata) > 0:
            if xrange[0] is None:
                xrange = (min(xdata), max(xdata))
            else:
                xrange = (min(xrange[0], min(xdata)), max(xrange[1], max(xdata)))
        return xrange

    def _finilize_range(self, xlim, xrange, xdata, mergin=(0.1, 0.1)):
        '''
        mergin つき range と lim を比較し、範囲の広いほうを採用する
        '''
        if xrange[0] is None:
            return xlim

        width = xrange[1] - xrange[0]
        minimum = xrange[0] - width * mergin[0]
        maximum = xrange[1] + width * mergin[1]

        # データが 2 点だけならば桁を加味して強制的に xrange を採用する
        if len(xdata) == 2:
            return (minimum, maximum)

        minimum = min(minimum, xlim[0])
        maximum = max(maximum, xlim[1])

        return (minimum, maximum)

    def _search_xylabels_in_shared_axes(self, ax):
        '''ax に label がなければ share された ax から label を探す'''
        xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()

        axes = ax.get_shared_x_axes().get_siblings(ax)
        for shared_ax in axes:
            if xlabel == '':
                xlabel = shared_ax.get_xlabel()

        axes = ax.get_shared_y_axes().get_siblings(ax)
        for shared_ax in axes:
            if ylabel == '':
                ylabel = shared_ax.get_ylabel()

        return xlabel, ylabel

    def _get_labeled_artists(self, ax, label):
        artists = ax.get_children()
        filterd_artists = []
        for artist in artists:
            if hasattr(artist, 'get_label') == True:
                if type(artist) == PathCollection:
                    if artist.get_label() == label:
                        filterd_artists.append(artist)
                elif type(artist) == Line2D:
                    if artist.get_label() == label:
                        filterd_artists.append(artist)
        return filterd_artists

    def update(self):
        for ax in self.fig.axes:
            xlabel, ylabel = self._search_xylabels_in_shared_axes(ax)

            # data から計算する lim の初期化
            xrange, yrange = (None, None), (None, None)
            for key, label in self.labels.items():
                # picked がなければ selected は飛ばす
                if (key == 'selected') and (self.picked_idx == -1):
                    continue
                # label された artist 全てを plot
                labeld_artists = self._get_labeled_artists(ax, label)
                for artist in labeld_artists:
                    # all, selected は予約
                    if key == 'all':
                        idx = slice(None)
                    elif key == 'selected':
                        idx = self.picked_idx
                    else:
                        idx = self.FEMOpt.history[key] == True
                    xdata, ydata = self.FEMOpt.history[[xlabel, ylabel]].dropna().values[idx].T
                    # もし長さが 1 だと ndarray にならないので変換
                    if isinstance(xdata, float) or isinstance(xdata, int):  # np.float などでも可
                        xdata = np.array([xdata])
                    if isinstance(ydata, float) or isinstance(ydata, int):  # np.float などでも可
                        ydata = np.array([ydata])
                    # plot の更新
                    if isinstance(artist, Line2D):
                        artist.set_data(xdata, ydata)
                        xrange = self._update_range(xrange, xdata)
                        yrange = self._update_range(yrange, ydata)
                    elif isinstance(artist, PathCollection):
                        artist.set_offsets(np.array([xdata, ydata]).T)
                        xrange = self._update_range(xrange, xdata)
                        yrange = self._update_range(yrange, ydata)
            # 全てのラベルのプロットが終われば set_lim
            xrange = self._finilize_range(ax.get_xlim(), xrange, xdata)
            yrange = self._finilize_range(ax.get_ylim(), yrange, ydata)
            ax.set(
                xlim=xrange,
                ylim=yrange,
            )
        if type(self.fig) == Figure:
            self.text_suptitle.set_y(0.96)
            # self.fig.subplots_adjust(
            #     left=0.2,
            #     bottom=0.2,
            #     right=0.95,
            #     top=0.95,
            #     wspace=0.05,
            #     hspace=0.05
            #     )
            self.fig.tight_layout()
            if self.fig.canvas.figure.stale:
                self.fig.canvas.draw_idle()
            # self.fig.canvas.start_event_loop(1) # pause すると画面がそのたびアクティブになって邪魔
        elif type(self.fig) == SubFigure:
            self.text_suptitle.set_y(0.85)
            self.fig.subplots_adjust(
                left=0.2,
                bottom=0.2,
                right=0.95,
                top=0.8,
                wspace=0.05,
                hspace=0.05
            )

    @abstractmethod
    def create(self):
        pass

    def _get_xydata(self, event) -> [np.ndarray]:
        fig: Figure = event.canvas.figure
        ax: Axes = event.artist.axes
        artist = event.artist
        idx: np.ndarray = event.ind  # 選択したデータのインデックス
        if isinstance(artist, Line2D):
            xdata, ydata = [data[idx] for data in artist.get_data()]
        if isinstance(artist, PathCollection):
            xdata, ydata = artist.get_offsets()[idx].T
        # print(xdata, ydata)
        return xdata, ydata

    def on_close(self, *args, **kwargs):
        self.FEMOpt.shared_interruption_flag.value = 1



