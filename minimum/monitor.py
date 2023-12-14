import webbrowser
import logging
from dash import Dash, html, dcc
from dash.dependencies import Output, Input
import plotly.graph_objs as go


def update_scatter_matrix(opt):
    # data
    data = opt.history.data

    trace = go.Splom(
        dimensions=[dict(label=c, values=data[c]) for c in opt.history.data.columns],
        diagonal_visible=False,
        showupperhalf=False,
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


class Monitor(object):

    def __init__(self, opt):

        self.opt = opt

        log_path = self.opt.history.path.replace('.csv', '.uilog')
        l = logging.getLogger()
        l.addHandler(logging.FileHandler(log_path))

        self.app = Dash(__name__)

        # layout の設定
        self.app.layout = html.Div(
            html.Div(
                [
                    html.Div('', id='dummy'),
                    dcc.Graph(id='scatter-matrix-graph'),
                    dcc.Interval(
                        id='interval-component',
                        interval=1*1000,  # in milliseconds
                        n_intervals=0
                    ),
                    html.Button('中断', id='interrupt-button')
                ]
            )
        )

        # 中断の設定
        @self.app.callback(
            Output('dummy', 'value'),
            Input('interrupt-button', 'n_clicks'))
        def interrupt(_):
            if _ is not None:
                self.opt.pdata.set_state('interrupted')
            return ''

        # scatter matrix
        @self.app.callback(
            Output('scatter-matrix-graph', 'figure'),
            Input('interval-component', 'n_intervals'))
        def update_sm(_):
            return update_scatter_matrix(self.opt)

    def start_server(self):
        webbrowser.open('http://localhost:8080')
        self.app.run(debug=False, host='localhost', port=8080)

