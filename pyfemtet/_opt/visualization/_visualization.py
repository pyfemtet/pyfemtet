from __future__ import annotations
from typing import Any, Callable
from abc import ABC, abstractmethod
from functools import wraps

from time import sleep
import datetime

import numpy as np

from ._visualizationNameSpace import *


class SimpleProcessMonitor:
    def __init__(self, FEMOpt):
        # FEMOpt を貰う
        self.FEMOpt = FEMOpt
        self.N = len(self.FEMOpt.objectives)
        # サブプロットを定義
        self.fig, axes = plt.subplots(self.N, 1, sharex=True, figsize=(3, 2))
        self.fig.suptitle('シンプル目的関数モニター')
        # ax を作って ilne と一緒に保持する
        self.lines = []
        for i in range(self.N):
            ax = self.fig.axes[i]
            line, = ax.plot([], [], 'o-', zorder=2) # メインの履歴
            scat = ax.scatter([], [], s=50, marker='x', color='black', lw=3, label='FEM error', zorder=3) # エラーが生じているとき
            subLine = ax.axhline(y=0, color='red', linestyle='--', lw=1, zorder=1)
            self.lines.append([line, subLine, scat])
            # スクロールした時に direction を移動させる関数を登録するためにイベントピッカーを登録する
            ax.set_picker('draw_event')
        # 登録したイベントピッカーを接続する
        self.fig.canvas.mpl_connect('draw_event', self.adjustDirectionPosition)
        
        def lmd(*args, **kwargs):
            self.FEMOpt.shared_interruption_flag.value = 1
        self.fig.canvas.mpl_connect('close_event', lmd)
        
        self.fig.tight_layout()

    def update(self):
        objectives = self.FEMOpt.objectives
        xdata = self.FEMOpt.history['time']
        if len(xdata)==0:
            plt.pause(0.5)
            return
        for ax, (line, subLine, scat), objective in zip(self.fig.axes, self.lines, objectives):
            # データの描画
            ydata = self.FEMOpt.history[objective.name].values
            line.set_data(xdata, ydata)
            subLine.set_ydata(ydata[-1]) # ylim 再設定の時に邪魔になるから
            
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
        if self.fig.canvas.figure.stale:
            self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(1)
        
    def adjustDirectionPosition(self, _):
        objectives = self.FEMOpt.objectives
        for ax, lines, objective in zip(self.fig.axes, self.lines, objectives):
            # ターゲット線の描画(get_limするから後で)
            subLine = lines[1]
            minimum, maximum = ax.get_ylim()
            if objective.direction=='maximize':
                y = maximum - (maximum - minimum) * 0.01
            elif objective.direction=='minimize':
                y = minimum + (maximum - minimum) * 0.01
            else: # 指定値と見做す
                y = objective.direction
            subLine.set_ydata(float(y))


class HypervolumeMonitor(UpdatableFigure):

    @wraps(UpdatableFigure.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # label の設定
        # ！all, selected 以外は bool を持つ history の column とする
        # ！xlabel, ylabel も history の column とする
        self.labels['all'] = 'ハイパーボリューム'
        self.labels['selected'] = '選択中'
        # title の設定
        self.text_suptitle = self.fig.suptitle('ハイパーボリュームプロット')
        # 作図
        self.create()
        self.update()
        plt.pause(0.1)

    def _plot(self, ax):
        #### 空の scatter の実行
        # 基本。灰色。
        ax.plot([], [], label=self.labels['all'],
                   marker='o',
                   # markersize=5,
                   # markerfacecolor='gray',
                   # markeredgecolor='gray',
                   color='gray',
                   lw=1,
                   zorder=1, picker=True
                   )
        # 選択。色なし、黄枠線。何よりも手前にあるべき。
        ax.scatter([], [], label=self.labels['selected'],
                   facecolors='none',
                   edgecolors='yellow',
                   s=60, lw=2, zorder=4, picker=True
                   )
    
    def create(self):
        #### プロットを作る
        # 配置を作る
        gs = self.fig.add_gridspec(1, 1)
        ax = self.fig.add_subplot(gs[0, 0])
        self.axes = np.array([[ax]])
        # 作図
        self._plot(self.axes[0,0])
        # ラベルなどの追加
        self.axes[0,0].set(xlabel='n_trial', ylabel='hypervolume')
        #### legend の設定
        axleg = self.axes[0,0]
        loc = 'upper right'
        axleg.legend(
            *self.axes[0,0].get_legend_handles_labels(),
            loc=loc,
            )

    def on_pick(self, event):
        # ax の取得
        ax:Axes = event.artist.axes
        # pick されたデータの取得
        xdata, ydata = super()._get_xydata(event)
        # データからインデックスを取得
        idx = int(xdata[0])
        # picked_idx の更新
        self.picked_idx = idx
        # figure の更新
        self.update()
        plt.pause(0.1)    


class MultiobjectivePairPlot(UpdatableFigure):
    
    @wraps(UpdatableFigure.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels['non_domi'] = '非劣解'
        self.labels['fit'] = '拘束範囲内'
        self.labels['all'] = 'その他'
        self.labels['selected'] = '選択中'
        self.text_suptitle = self.fig.suptitle('多目的ペアプロット')
        self.create()
        self.update()
        plt.pause(0.1)

    def _plot(self, ax):
        #### 空の scatter の実行
        # 非劣解。色なし、赤色。拘束違反でありうる。
        ax.scatter([], [], label=self.labels['non_domi'],
                   facecolors='none',
                   edgecolors='orangered',
                   s=20, lw=1, zorder=3, picker=True
                   )
        # 拘束範囲内。青色、枠線無し
        ax.scatter([], [], label=self.labels['fit'],
                   facecolors='royalblue',
                   edgecolors='none',
                   s=20, lw=0, zorder=2, picker=True
                   )
        # 基本。灰色、枠線無し。
        ax.scatter([], [], label=self.labels['all'],
                   facecolors='gray',
                   edgecolors='none',
                   s=20, lw=0, zorder=1, picker=True
                   )
        # 選択。色なし、黄枠線。何よりも手前にあるべき。
        ax.scatter([], [], label=self.labels['selected'],
                   facecolors='none',
                   edgecolors='yellow',
                   s=60, lw=2, zorder=4, picker=True
                   )
    
    def create(self):
        #### 下半分のペアプロットを実行
        # 配置を作る
        objective_names = self.FEMOpt.get_history_columns('objective')
        n = len(objective_names)
        gs = self.fig.add_gridspec(n-1, n-1)
        self.axes = []
        # 下半分にプロットを行っていく
        for r in range(n-1):
            self.axes.append([])
            for c in range(n-1):
                ax = self.fig.add_subplot(gs[r, c])
                self.axes[-1].append(ax)
                # 下半分ならプロット
                if r>=c:
                    # 作図
                    self._plot(ax)
                    # 一番下なら xlabel を表示、そうでなければ xticklabels を非表示
                    if r==n-2:
                        ax.set_xlabel(objective_names[c])
                    else:
                        [text.set_visible(False) for text in ax.xaxis.get_ticklabels()]
                    # 一番左なら ylabel を表示、そうでなければ yticklabels を非表示
                    if c==0:
                        ax.set_ylabel(objective_names[r+1])
                    else:
                        [text.set_visible(False) for text in ax.yaxis.get_ticklabels()]
                # そうでなければ非表示
                else:
                    ax.set_visible(False)
        self.axes = np.array(self.axes)

        #### 軸の同期設定
        # 同じ column で sharex
        for c in range(n-1):
            ax1 = self.axes[0,c]
            for r in range(1,n-1):
                ax2 = self.axes[r,c]
                ax2.sharex(ax1)
        # 同じ row で sharey
        for r in range(n-1):
            ax1 = self.axes[r,0]
            for c in range(0, n-1):
                ax2 = self.axes[r,c]
                ax2.sharey(ax1)

        #### legend の設定
        # 一番右上を legend 用にする
        axleg = self.axes[0,-1]
        axleg.set_visible(True)
        if id(axleg)==id(self.axes[0,0]):
            loc = 'upper right'
        else:
            loc = 'center'
            axleg.axis('off')
        axleg.legend(
            *self.axes[0,0].get_legend_handles_labels(),
            loc=loc,
            )
        
    def get_idx(self, x, y, xlabel, ylabel):
        x, y = x[0], y[0]
        # xdata, ydata ともに一致する idx を返す
        xdata = self.FEMOpt.history[xlabel]
        ydata = self.FEMOpt.history[ylabel]
        match_idx_x, = np.where(xdata==x)
        match_idx_y, = np.where(ydata==y)
        match_idx = np.intersect1d(match_idx_x, match_idx_y)
        if len(match_idx)==0:
            return -1
        else:
            return match_idx[0]

    def on_pick(self, event):
        # ax の取得
        ax:Axes = event.artist.axes
        # pick されたデータの取得
        xdata, ydata = super()._get_xydata(event)
        xlabel, ylabel = self._search_xylabels_in_shared_axes(ax)
        # データからインデックスを取得
        idx = self.get_idx(xdata, ydata, xlabel, ylabel)
        self.picked_idx = idx

        self.update()
        plt.pause(0.1)



