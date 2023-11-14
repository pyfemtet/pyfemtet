# import os
# here, me = os.path.split(__file__)
# mother = os.path.dirname(here)

# cwd = os.getcwd()
# os.chdir(mother)
from ._visualizationNameSpace import *
# os.chdir(cwd)


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
        
        # TODO:中断（experimental）
        def lmd(*args, **kwargs):
            self.FEMOpt.interruption = True
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
            
            # エラーの描画
            idx = self.FEMOpt.history['error_message']!=''
            xdata = list(self.FEMOpt.history['time'][idx])
            ydata = list(self.FEMOpt.history[objective.name][idx])
            scat.set_offsets(np.array([xdata, ydata]).T)

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
        plt.pause(0.01)
        
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



class MultiobjectivePairPlot:
    labelAll = 'the others'
    labelNonDomi = 'non-inferior solution'
    labelNonFit = 'out of constraints(if exists)'
    labelSelected = 'selection'
    
    def __init__(self, FEMOpt):
        self.FEMOpt = FEMOpt
        self.g = None
        self.picked = None # クリックされた history の idx
        self.create() # self.g がこの中で入る
        
        
    def create(self):
        self.generateGrid() # self.g がこの中で入る
        self.update()

    def getLabeledScatter(self, ax, label):
        artists = ax.get_children()
        for artist in artists:
            if type(artist)==PathCollection:
                if hasattr(artist, 'get_label')==True:
                    if artist.get_label()==label:
                        return artist

        
    def update(self):
        '''FEMOpt の内容を g の scatter に反映する'''
        # データの用意
        history = self.FEMOpt.history

        if len(history)==0:
            plt.pause(0.5)
            return

        objectiveNames = self.FEMOpt.get_history_columns('objective')
        constraintNames = self.FEMOpt.get_history_columns('constraint')
        
    
        # ax の用意（objectives の並び順になっているはず）
        for rName, axes in zip(objectiveNames[1:], self.g.axes):
            for cName, ax in zip(objectiveNames[:-1], axes):
                if ax is None:
                    continue
                # scatter（set_offsets）
                # all
                idx = history['error_message']==''
                pdf = history[[cName, rName]][idx]
                scat = self.getLabeledScatter(ax, label=self.labelAll)
                if scat is not None:
                    scat.set_offsets(pdf.values)
                #non-dominant
                idx = history['non_domi']==True
                pdf = history[[cName, rName]][idx]
                scat = self.getLabeledScatter(ax, label=self.labelNonDomi)
                if scat is not None:
                    scat.set_offsets(pdf.values)
                # non-fit
                idx = history['fit']==False
                pdf = history[[cName, rName]][idx]
                scat = self.getLabeledScatter(ax, label=self.labelNonFit)
                if scat is not None:
                    scat.set_offsets(pdf.values)

                # 選択                
                if self.picked is not None:
                    pdf = history[[cName, rName]].loc[self.picked]
                    scat = self.getLabeledScatter(ax, label=self.labelSelected)
                    if scat is not None:
                        scat.set_offsets(pdf.values.T)
                    
                # 視野調整
                idx = history['error_message']==''
                pdf = history[[cName, rName]][idx]
                if len(pdf)==1:
                    xlim = pdf.values.T[0].min(), pdf.values.T[0].max()
                    xrange = 1
                    ax.set_xlim(xlim[0] - xrange*0.1, xlim[1] + xrange*0.1)
                    ylim = pdf.values.T[1].min(), pdf.values.T[1].max()
                    yrange = 1
                    ax.set_ylim(ylim[0] - yrange*0.1, ylim[1] + yrange*0.1)

                elif len(pdf)==2:
                    xlim = pdf.values.T[0].min(), pdf.values.T[0].max()
                    ylim = pdf.values.T[1].min(), pdf.values.T[1].max()
                    xrange = xlim[1] - xlim[0]
                    yrange = ylim[1] - ylim[0]
                    ax.set_xlim(xlim[0] - xrange*0.1, xlim[1] + xrange*0.1)
                    ax.set_ylim(ylim[0] - yrange*0.1, ylim[1] + yrange*0.1)

                elif len(pdf)>2:
                    xlim = pdf.values.T[0].min(), pdf.values.T[0].max()
                    ylim = pdf.values.T[1].min(), pdf.values.T[1].max()
                    xrange = xlim[1] - xlim[0]
                    yrange = ylim[1] - ylim[0]
                    xlim = xlim[0] - xrange*0.1, xlim[1] + xrange*0.1
                    ylim = ylim[0] - yrange*0.1, ylim[1] + yrange*0.1
                    prevXlim = ax.get_xlim()
                    prevYlim = ax.get_ylim()
                    xmin = min(xlim[0], prevXlim[0])
                    xmax = max(xlim[1], prevXlim[1])
                    ymin = min(ylim[0], prevYlim[0])
                    ymax = max(ylim[1], prevYlim[1])
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)

        nRun = len(self.FEMOpt.history)
        if nRun==0:
            suffix = ''
        elif nRun==1:
            suffix = '1 run'
        elif nRun>1:
            suffix = f'{nRun} runs'
        self.g.figure.suptitle(f'solution space: {suffix}')
        # self.g.figure.tight_layout(rect=[0,0,1,0.96])
        plt.pause(0.01)
    
    def generateGrid(self):
        # データのセットアップ
        history = self.FEMOpt.history
        objective_names = self.FEMOpt.get_history_columns('objective')
        
        # 空のデータを作成
        empty = pd.DataFrame([], columns=history.columns)
        
        # 空のデータで pairplot の実行
        g = sns.pairplot(empty, vars=objective_names, diag_kind='None', corner=True)
        g.fig.set_figheight(4)
        g.fig.set_figwidth(6)
        plt.tight_layout()

        # pairplot の対角要素は意味を持たないので
        # GridSpec を用意して ax の再配置を行う
        n = len(objective_names)
        gs = gridspec.GridSpec(n-1, n-1)
        for r in range(1,n):
            for c in range(0,n-1):
                if r>c:
                    g.axes[r, c].set_position(gs[r-1, c].get_position(g.figure))
    
        # 対角の ax や上半分の ax を削除する
        for r in range(n):
            for c in range(n):
                if r<=c:
                    # pg.axes[r,c].set_visible(False)
                    ax = g.axes[r,c]
                    for _ax in g.figure.axes:
                        if id(ax)==id(_ax):
                            g.figure.delaxes(_ax)
        g.axes = np.delete(g.axes, 0, axis=0)
        g.axes = np.delete(g.axes, -1, axis=1)
    
        # 残っているデータに対し、空の scatter を実行する
        for ax in g.figure.axes:
            # 非劣解。色なし、赤枠線
            ax.scatter([], [], label=self.labelNonDomi, facecolors='none', edgecolors='orangered', s=40, lw=1, zorder=3, picker='pick_event')
            # 拘束を満たさない。灰色、枠線無し
            ax.scatter([], [], label=self.labelNonFit, color='gray', edgecolors='none', s=20, zorder=2, picker='pick_event')
            # 基本。青プロット、枠線無し。legendで最後にするためここにあるが、他より下にないといけない。
            ax.scatter([], [], label=self.labelAll, color='royalblue', edgecolors='gray', s=20, lw=1, zorder=1, picker='pick_event')
            # 選択。黄色プロット、青枠線。何よりも手前にあるべき。
            ax.scatter([], [], label=self.labelSelected, color='yellow', edgecolors='royalblue', s=60, lw=2, zorder=4, picker='pick_event')


        g.fig.canvas.mpl_connect('pick_event', self.onPick)

        # TODO:中断（experimental）
        def lmd(*args, **kwargs):
            self.FEMOpt.interruption = True
        g.figure.canvas.mpl_connect('close_event', lmd)


        
        # legend を作る（位置調整のため legend 用の ax を作る）
        if n==2:
            gs = gridspec.GridSpec(3, 4)
            ax_legend = plt.subplot(gs[1,-1])
        else:
            ax_legend = plt.subplot(gs[:int((n-1)/2), int(n/2):])
        ax_legend.axis('off') # 軸の非表示
        # for 文の中で最後に実行された ax から label を取得（全部変わらないはずだから）
        handles, labels = ax.get_legend_handles_labels()
        legend = ax_legend.legend(handles, labels, loc='center') # , ncol=2
        # 書式を決める
        frame = legend.get_frame()
        frame.set_facecolor('white') # 凡例の背景色を白に設定します
        frame.set_alpha(1) # 凡例の背景の透明度を0に設定します
        
        # タイトル
        g.figure.suptitle('solution space')
        # g.figure.tight_layout(rect=[0,0,1,0.96])
        
        # 登録
        self.g = g
        
    def onPick(self, event):
        # 準備
        objectiveNames = self.FEMOpt.get_history_columns('objective')
        n = len(objectiveNames)-1
        ax = event.artist.axes
        # どのデータか？
        r, c = -1, -1
        for _r in range(n):
            for _c in range(n):
                if id(self.g.axes[_r,_c])==id(ax):
                    r = _r
                    c = _c
                    break
        rName = objectiveNames[r+1] # 最初のひとつが削除されているため +1 する
        cName = objectiveNames[c] # 最後のひとつが削除されているため何もしなくていい        
        xydata = event.artist.get_offsets()[event.ind[0]]
        
        # history の 何行目か？->picked の更新
        df = self.FEMOpt.history        
        xydataList = df[[cName, rName]].values
        idx = np.where(xydata==xydataList)[0][0] # 行の 1 つめと 2 つめは一致しているはずだから 0 でいい
        self.picked = idx
        
        # 各 ax でデータを更新
        self.updateSelection()
        
        
        plt.pause(0.01)

    def updateSelection(self):
        # r, c を順次見ていく
        # 準備
        objectiveNames = self.FEMOpt.get_history_columns('objective')
        df = self.FEMOpt.history
        n = len(objectiveNames)-1
        for r in range(n):
            for c in range(n):
                # ax が None でなければ
                ax = self.g.axes[r, c]
                if ax is not None:
                    # cName, rName を取得
                    rName = objectiveNames[r+1] # 最初のひとつが削除されているため +1 する
                    cName = objectiveNames[c] # 最後のひとつが削除されているため何もしなくていい        
                    # pdf を作成
                    pdf = df[[cName, rName]].loc[self.picked]
                    # scatter を更新
                    scat = self.getLabeledScatter(ax, label=self.labelSelected)
                    if scat is not None:
                        scat.set_offsets(pdf.values.T)
        
        



