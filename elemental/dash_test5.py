from threading import Thread

import webbrowser

import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import plotly.express as px

import datetime
from time import sleep

import numpy as np
import pandas as pd


def update_scatter_matrix(FEMOpt):
    # data
    data = FEMOpt.history
    parameter_names = FEMOpt.get_history_columns('parameter')
    objective_names = FEMOpt.get_history_columns('objective')
    # is_fit = data['fit']
    # is_nondomi = data['non_domi']

    # pairplot
    # text_list = []
    # for i, row in data[parameter_names].iterrows():
    #     text = ''
    #     for prm in row.index:
    #         value = row[prm]
    #         text = text + f'{prm} : {value:.3f}<br>'
    #     text_list.append(text)
    trace=go.Splom(
        dimensions=[dict(label=c, values=data[c]) for c in objective_names],
        diagonal_visible=False,
        showupperhalf=False,
        # text=text_list,
        )
    
    # figure layout
    layout = go.Layout(
        title_text="多目的ペアプロット",
        )
            
    # figure
    fig = go.Figure(
        data=trace,
        layout=layout,
        )
    
    return fig    


def update_hypervolume(FEMOpt):
    data = FEMOpt.history
    
    # plot
    trace = go.Scatter(
        x=data['n_trial'],
        y=data['hypervolume'],
        mode='lines+markers',
        # text=[dt.strftime('終了時刻：%Y/%m/%d %H:%M:%S') for dt in data['time']]
        )

    # figure layout
    layout = go.Layout(
        title_text="ハイパーボリューム",
        )
            
    # figure
    fig = go.Figure(
        data=trace,
        layout=layout,
        )
    
    return fig


class DashProcessMonitor:

    def start(self):
        self.app.run(debug=True, host='localhost', port=8080)
        webbrowser.open("http://localhost:8080")

    
    def __init__(self, FEMOpt):
        # 引数の処理
        self.FEMOpt = FEMOpt
        
        # application の準備
        self.app = dash.Dash(__name__)

        # layout の設定
        self.app.layout = html.Div(
            html.Div(
                [
                    dcc.Graph(id='hypervolume-graph'),
                    dcc.Graph(id='scatter-matrix-graph'),
                    dcc.Interval(
                        id='interval-component',
                        interval=1*1000, # in milliseconds
                        n_intervals=0
                        )
                    ]
                )
            )

        # callback（分割したほうが並列計算できるらしい）

        # hypervolume
        @self.app.callback(
            Output('hypervolume-graph', 'figure'),
            Input('interval-component', 'n_intervals'))
        def update_hv(_):
            return update_hypervolume(self.FEMOpt)

        # scatter matrix
        @self.app.callback(
            Output('scatter-matrix-graph', 'figure'),
            Input('interval-component', 'n_intervals'))
        def update_sm(_):
            return update_scatter_matrix(self.FEMOpt)

        # フラグが立っていれば更新しない
        @self.app.callback(
            Output('interval-component', 'max_intervals'),
            Input('interval-component', 'n_intervals'))
        def stop_interval(_):
            if self.FEMOpt.should_finish():
                return 0  # Stop the interval
            else:
                return -1  # Allow the interval to run indefinitely

def f(FEMOpt):
    for i in range(10):
        FEMOpt.calc()
    FEMOpt.flag = True

    
if __name__=='__main__':
    class DummyFEMOpt:
        def __init__(self):
            self.flag = False
            self.calc()
            
        def calc(self):
            self.history = pd.DataFrame(
                dict(
                    n_trials=np.arange(10),
                    obj1=np.random.rand(10),
                    obj2=np.random.rand(10),
                    obj3=np.random.rand(10),
                    prm1=np.random.rand(10),
                    prm2=np.random.rand(10),
                    non_domi=np.random.rand(10)>0.5,
                    fit=np.random.rand(10)>0.5,
                    hypervolume=np.random.rand(10),
                    n_trial=np.arange(10),
                    time=[datetime.datetime.now()]*10,
                    )
                )
            sleep(1)
            
        def get_history_columns(self, kind):
            if kind=='parameter':
                return ['prm1', 'prm2']
            if kind=='objective':
                return ['obj1', 'obj2', 'obj3']
        
        def should_finish(self):
            return self.flag

    FEMOpt = DummyFEMOpt()
    
    t = Thread(target=f, args=(FEMOpt,))
    t.start()

    dpm = DashProcessMonitor(FEMOpt)
    dpm.start()
    
    t.join()
    