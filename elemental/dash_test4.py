# 目的：外から与えられた値に対する scatter matrix の live-update
# 目標：再利用可能なコードスタイル

from time import sleep

import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly
import plotly.graph_objects as go

import numpy as np
import pandas as pd

from threading import Thread


def update_scatter_matrix(calculator):
    # data
    data = calculator.df

    # pairplot
    trace=go.Splom(
        dimensions=[dict(label=c, values=data[c]) for c in data.columns],
        diagonal_visible=False,
        showupperhalf=False,
        )
    
    # figure layout
    layout = go.Layout(
        title_text="random",
        )
            
    # figure
    fig = go.Figure(
        data=trace,
        layout=layout,
        )
    
    return fig


def update_another_graph(calculator):
    ...
    return update_scatter_matrix(calculator)


class DashApp:

    def start(self):
        self.app.run(debug=True, host='localhost', port=8080)
    
    def __init__(self, calculator):
        
        self.calculator = calculator
        
        # application の設定
        self.app = Dash(__name__)

        # layout の設定
        self.app.layout = html.Div(
            html.Div(
                [
                    dcc.Graph(id='live-update-graph'),
                    dcc.Graph(id='live-update-graph2'),
                    dcc.Interval(
                        id='interval-component',
                        interval=1*1000, # in milliseconds
                        n_intervals=0
                        )
                    ]
                )
            )
                
        # callback の設定
        @callback(
            Output('live-update-graph', 'figure'),
            Output('live-update-graph2', 'figure'),
            Input('interval-component', 'n_intervals')
            )
        def update(_):
            ret = (
                update_scatter_matrix(self.calculator),
                update_another_graph(self.calculator)                
                )
            return ret


class Calculator:
    def calc(self):
        for i in range(10):
            _data = np.random.rand(5,5)
            self.df = pd.DataFrame(_data, columns='A B C D E'.split())
            sleep(1)


if __name__ == '__main__':
    c = Calculator()
    
    t = Thread(target=c.calc)
    t.start()
    
    pm = DashApp(c)
    pm.start()
    
    t.join()
    
    
