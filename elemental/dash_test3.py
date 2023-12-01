# 目的：外から与えられた値に対する scatter matrix の live-update
# 目標：外から与えられた値に対する scatter matrix の live-update

from time import sleep

import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly
import plotly.graph_objects as go

import numpy as np
import pandas as pd

from threading import Thread


class DashApp:
    
    def __init__(self, calculator):
        
        self.calculator = calculator
        
        # application の設定
        self.app = Dash(__name__)
        self.app.layout = html.Div(
            html.Div(
                [
                    dcc.Graph(id='live-update-graph'),
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
            Input('interval-component', 'n_intervals')
            )
        def update(_):
            return self.update_graph_live()


    def update_graph_live(self):
        
        # data
        data = self.calculator.df
    
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
    
    def start(self):
        self.app.run(debug=True, host='localhost', port=8080)

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
    
