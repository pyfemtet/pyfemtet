import webbrowser

import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import plotly.express as px

import dash_bootstrap_components as dbc

import datetime

import numpy as np
import pandas as pd

COLORS = dict(
    Primary='#0d6efd', # 青
    Secondary='#6c757d', # 灰色
    Success='#198754', # 緑
    Danger='#dc3545', # 赤
    Warning='#ffc107', # 黄
    Info='#0dcaf0', # 水色
    Light='#f8f9fa', # 白っぽい灰色
    Dark='#212529', # 黒っぽい灰色
    )


def update_scatter_matrix(FEMOpt):
    # data
    data = FEMOpt.history
    parameter_names = FEMOpt.get_history_columns('parameter')
    objective_names = FEMOpt.get_history_columns('objective')
    # is_fit = data['fit']
    # is_nondomi = data['non_domi']

    # マウスオーバーで変数セットを表示する設定
    text_list = []
    for i, row in data[parameter_names].iterrows():
        text = ''
        for prm in row.index:
            value = row[prm]
            text = text + f'{prm} : {value:.3f}<br>'
        text_list.append(text)

    # # nondomi と fit に基づいて書式を設定
    # linecolor_nondomi = COLORS['Warning']
    # linecolor_others = COLORS['Light']
    # color_normal = COLORS['Primary']
    # color_nonfit = COLORS['Secondary']
    
    # marker_colors = [color_normal if fit else color_nonfit for fit in data['fit']]
    # marker_line_colors = [linecolor_nondomi if optimal else linecolor_others for optimal in data['non_domi']]
    # marker_line_width = [2 if optimal else 1 for optimal in data['non_domi']]

    # # 設定を反映
    # trace=go.Splom(
    #     dimensions=[dict(label=c, values=data[c]) for c in objective_names],
    #     diagonal_visible=False,
    #     showupperhalf=False,
    #     text=text_list,
    #     showlegend=True,
    #     marker=dict(
    #         color=marker_colors,
    #         line=dict(
    #             color=marker_line_colors,
    #             width=marker_line_width
    #             )
    #         )
    #     )
    
    # # legends=[
    # #     go.Scatter(
    # #         x=[None], y=[None], mode='markers', showlegend=True, legendgroup='非劣解',
    # #         name='非劣解',
    # #         marker=dict(
    # #             color='rgba(0,0,0,0)',
    # #             line=dict(color=linecolor_nondomi, width=2)
    # #             ),
    # #         ),
        
    # #     go.Scatter(
    # #         x=[None], y=[None], mode='markers', showlegend=True, legendgroup='拘束外の解',
    # #         name='拘束外の解',
    # #         marker=dict(color=color_nonfit),
    # #         ),
    
    # #     go.Scatter(
    # #          x=[None], y=[None], mode='markers', showlegend=True, legendgroup='その他の解',
    # #          name='その他の解',
    # #          marker=dict(color=color_normal),
    # #      )
    # # ]
    
    # # figure layout
    # layout = go.Layout(title_text="多目的ペアプロット")
    
    # # figure
    # # fig = go.Figure(data=[trace] + legends, layout=layout)
    # fig = go.Figure(data=[trace], layout=layout)



    data['拘束'] = data['fit'].astype(str)
    data.loc[data['fit']==True, '拘束'] = 'OK'
    data.loc[data['fit']==False, '拘束'] = 'NG'

    data['好ましい'] = data['non_domi'].astype(str)
    data.loc[data['non_domi']==True, '好ましい'] = 'OK'
    data.loc[data['non_domi']==False, '好ましい'] = 'NG'
    
    fig = px.scatter_matrix(
        data,
        dimensions=objective_names,
        color="拘束",
        symbol='好ましい',
        )

    return fig    


def update_hypervolume(FEMOpt):
    data = FEMOpt.history
    
    # plot
    trace = go.Scatter(
        x=data['n_trial'],
        y=data['hypervolume'],
        mode='lines+markers',
        text=[dt.strftime('終了時刻：%Y/%m/%d %H:%M:%S') for dt in data['time']]
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
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


        #### component と layout の設定
        graph_layout = dbc.Container(
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id='hypervolume-graph'), md=6),
                    dbc.Col(dcc.Graph(id='scatter-matrix-graph'), md=6),
                    ]
                )
            )
        toggle_button = dbc.Button('Toggle auto-update', id='toggle-button', n_clicks=0)
        layout = html.Div(
            [
                html.H1("最適化の進捗状況"),
                html.Label(id='status-label'),
                dcc.Interval(
                    id='interval-component',
                    interval=2*1000,  # in milliseconds
                    max_intervals=0  # max number of intervals
                    ),
                graph_layout,
                toggle_button,
                ]
            )

        self.app.layout = layout


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

        # フラグが立っていれば自動更新しない
        @self.app.callback(
            [
                Output('interval-component', 'max_intervals'),
                Output('status-label','children'),
                Output('toggle-button', 'disabled'),
                ],
            [
                Input('interval-component', 'n_intervals'),
                ]
            )
        def stop_interval(_):
            if self.FEMOpt.should_finish():
                max_intervals = 0
                status_text = '最適化が終了しました。'
                toggle_button_disable = True
            else:
                max_intervals = -1
                status_text = '最適化が実行中です。'
                toggle_button_disable = False
            return max_intervals, status_text, toggle_button_disable

        # 自動更新の on / off
        @self.app.callback(
            [
                Output('interval-component', 'disabled'),
                Output('toggle-button', 'children'),
                ],
            [Input('toggle-button', 'n_clicks')])
        def toggle_interval(n):
            if n % 2 == 0:
                return False, 'Turn off auto update'
            else:
                return True, 'Turn on auto update'
    
    
    
    
if __name__=='__main__':
    class DummyFEMOpt:
        def __init__(self):
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
        def get_history_columns(self, kind):
            if kind=='parameter':
                return ['prm1', 'prm2']
            if kind=='objective':
                return ['obj1', 'obj2', 'obj3']
        
        def should_finish(self):
            return False
    
    FEMOpt = DummyFEMOpt()
    dpm = DashProcessMonitor(FEMOpt)
    dpm.start()