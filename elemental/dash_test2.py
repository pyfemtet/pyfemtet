# 目的：外から与えられた値に対する scatter matrix の live-update
# 目標：クラス化した app での scatter matrix の live-update


import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly
import plotly.graph_objects as go

import numpy as np
import pandas as pd


class DashApp:
    
    def __init__(self):
        # application の設定
        self.app = Dash(__name__)
        self.app.layout = html.Div(
            html.Div(
                [
                    dcc.Graph(id='live-update-graph'),
                    dcc.Interval(
                        id='interval-component',
                        interval=3*1000, # in milliseconds
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
        _data = np.random.rand(5,5)
        data = pd.DataFrame(_data, columns='A B C D E'.split())
    
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

if __name__ == '__main__':
    pm = DashApp()
    pm.start()
