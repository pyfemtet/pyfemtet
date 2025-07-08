from __future__ import annotations

from typing import Literal, TypeAlias

import pandas as pd
import optuna

# dash components
from pyfemtet.opt.visualization.history_viewer._wrapped_components import dcc, dbc, html

# dash callback
from dash import Output, Input, callback_context, no_update
from dash.exceptions import PreventUpdate

from pyfemtet.logger import get_module_logger
from pyfemtet._i18n import _

from pyfemtet.opt.history import History, MAIN_FILTER
from pyfemtet.opt.visualization.history_viewer._base_application import AbstractPage
from pyfemtet.opt.visualization.plotter.parallel_plot_creator import parallel_plot
from pyfemtet.opt.visualization.plotter.contour_creator import contour_creator


ItemKind: TypeAlias = Literal['prm', 'all_output', 'obj', 'cns', 'other_output']


class SelectablePlot(AbstractPage):
    location: dcc.Location
    graph: dcc.Graph
    input_items: dcc.Checklist | dcc.RadioItems | html.Div
    output_items: dcc.Checklist | dcc.RadioItems | html.Div
    InputItemsClass = dcc.Checklist
    OutputItemsClass = dcc.Checklist
    alerts: html.Div
    input_item_kind: set[ItemKind] = {'prm'}
    output_item_kind: set[ItemKind] = {'all_output'}
    description_markdown: str = ''

    def __init__(self, title='base-page', rel_url='/', application=None,
                 location=None):
        self.location = location
        super().__init__(title, rel_url, application)

    @property
    def plot_title(self) -> str:
        raise NotImplementedError

    def setup_layout(self):

        self.layout = dbc.Container([
            # ----- hidden -----
            dbc.Row([self.location]),

            # ----- visible -----
            dbc.Row([html.H2(self.plot_title)]),
            dbc.Row([dbc.Col(dcc.Markdown(self.description_markdown))]),
            dbc.Row(
                [
                    dbc.Col(dbc.Spinner(self.graph)),
                    dbc.Col(
                        [
                            dbc.Row(html.H3(_('Choices:', '選択肢:'))),
                            dbc.Row(html.Hr()),
                            dbc.Row(self.input_items),
                            dbc.Row(self.output_items),
                        ],
                        md=2
                    ),
                ],
            ),
            dbc.Row([self.alerts]),
            dbc.Row(html.Hr()),
        ])

    def setup_component(self):

        if self.location is None:
            self.location = dcc.Location(id='selectable-plot-location', refresh=True)

        # graph
        self.graph = dcc.Graph(style={'height': '85vh'})

        # checklist
        self.input_items = self.InputItemsClass(options=[]) if self.InputItemsClass is not None else html.Div()
        self.output_items = self.OutputItemsClass(options=[]) if self.OutputItemsClass is not None else html.Div()

        # alert
        self.alerts = html.Div()

    def _check_precondition(self, logger) -> tuple[History, pd.DataFrame, pd.DataFrame]:

        if callback_context.triggered_id is None:
            logger.debug('PreventUpdate. No trigger.')
            raise PreventUpdate

        if self.application is None:
            logger.debug('PreventUpdate. No application.')
            raise PreventUpdate

        if self.application.history is None:
            logger.debug('PreventUpdate. No history.')
            raise PreventUpdate

        history = self.application.history

        df = self.application.get_df()
        main_df = self.application.get_df(MAIN_FILTER)
        if len(df) == 0:
            logger.debug('PreventUpdate. No df.')
            raise PreventUpdate

        return history, df, main_df

    @staticmethod
    def _return_checklist_options_and_value(history, types: set[Literal]) -> tuple[list[dict], list[str]]:

        keys = []

        if 'prm' in types:
            keys.extend(history.prm_names)

        if 'all_output' in types:
            keys.extend(history.all_output_names)

        else:
            if 'obj' in types:
                keys.extend(history.obj_names)
            if 'cns' in types:
                keys.extend(history.cns_names)
            if 'other_output' in types:
                keys.extend(history.other_output_names)

        return [dict(label=key, value=key) for key in keys], keys

    def _return_input_checklist_options_and_value(self, history):
        return self._return_checklist_options_and_value(history, self.input_item_kind)

    def _return_output_checklist_options_and_value(self, history):
        return self._return_checklist_options_and_value(history, self.output_item_kind)

    def setup_update_plot_input_checklist_callback(self):

        @self.application.app.callback(
            Output(self.input_items, 'options'),
            Output(self.input_items, 'value'),
            Input(self.location, 'pathname'),  # on page load
        )
        def update_plot_input_checklist(_):

            logger_name = f'opt.{type(self).__name__}.update_plot_input_checklist()'

            logger = get_module_logger(
                logger_name,
                debug=False,
            )

            logger.debug('callback fired!')

            # ----- preconditions -----

            history, _, _ = self._check_precondition(logger)

            # ----- main -----
            options, value = self._return_input_checklist_options_and_value(history)

            if isinstance(self.input_items, dcc.RadioItems):
                value = value[0]

            return options, value

    def setup_update_plot_output_checklist_callback(self):

        @self.application.app.callback(
            Output(self.output_items, 'options'),
            Output(self.output_items, 'value'),
            Input(self.location, 'pathname'),  # on page load
        )
        def update_plot_output_checklist(_):

            logger_name = f'opt.{type(self).__name__}.update_plot_output_checklist()'

            logger = get_module_logger(
                logger_name,
                debug=False,
            )

            logger.debug('callback fired!')

            # ----- preconditions -----

            history, _, _ = self._check_precondition(logger)

            # ----- main -----
            options, value = self._return_output_checklist_options_and_value(history)

            logger.debug(value)
            if isinstance(self.output_items, dcc.RadioItems):
                value = value[0]
            logger.debug(value)

            return options, value

    def setup_update_plot_graph_callback(self):

        @self.application.app.callback(
            # graph output
            Output(self.graph, 'figure'),
            Output(self.alerts, 'children'),
            # checklist input
            inputs=dict(
                selected_input_values=Input(self.input_items, 'value'),
                selected_output_values=Input(self.output_items, 'value'),
            ),
        )
        def update_plot_graph(
                selected_input_values: list[str] | str,
                selected_output_values: list[str] | str,
        ):

            logger_name = f'opt.{type(self).__name__}.update_plot_graph()'

            logger = get_module_logger(
                logger_name,
                debug=False,
            )

            logger.debug('callback fired!')

            # ----- preconditions -----

            history, df, main_df = self._check_precondition(logger)

            # null selected values
            if selected_input_values is None:
                logger.debug('No input items.')
                return no_update, [dbc.Alert('No input items.', color='danger')]

            if selected_output_values is None:
                logger.debug('No output items.')
                return no_update, [dbc.Alert('No output items.', color='danger')]

            # type correction
            if isinstance(selected_input_values, str):
                selected_input_values = [selected_input_values]
            if isinstance(selected_output_values, str):
                selected_output_values = [selected_output_values]

            # nothing selected
            # selected_values = selected_input_values + selected_output_values
            # if len(selected_values) == 0:
            #     logger.debug('No items are selected.')
            #     return no_update, [dbc.Alert('No items are selected.', color='danger')]
            if len(selected_input_values) == 0:
                logger.debug('No input items are selected.')
                return no_update, [dbc.Alert('No input items are selected.', color='danger')]
            if len(selected_output_values) == 0:
                logger.debug('No output items are selected.')
                return no_update, [dbc.Alert('No output items are selected.', color='danger')]

            # ----- main -----
            used_df = self.make_used_df(history, df, main_df, selected_input_values, selected_output_values)
            assert len(used_df) > 0
            assert len(used_df.columns) > 0

            fig_or_err = self.create_plot(used_df)

            if isinstance(fig_or_err, str):
                return no_update, [dbc.Alert(fig_or_err, color='danger')]

            return fig_or_err, []

    def setup_callback(self):
        self.setup_update_plot_input_checklist_callback()
        self.setup_update_plot_output_checklist_callback()
        self.setup_update_plot_graph_callback()

    # noinspection PyUnusedLocal
    @staticmethod
    def make_used_df(history, df, main_df, selected_input_values, selected_output_values):
        # NotImplementedError でもいいが、汎用的なので

        columns = [
            col for col in history.prm_names + history.all_output_names
            if col in selected_input_values + selected_output_values
        ]

        use_df = main_df[columns]

        return use_df

    @staticmethod
    def create_plot(used_df):
        raise NotImplementedError


class ParallelPlot(SelectablePlot):

    plot_title = _('parallel coordinate plot', '平行座標プロット')
    description_markdown: str = _(
        en_message='Visualize the relationships between input and output values in multiple dimensions. '
                   'You can intuitively grasp trends and the magnitude of influence between variables for specific output values.\n\n'
                   '**Tips: You can rearrange the axes and select ranges.**',
        jp_message='各入力値と出力値の関係を多次元で可視化。'
                   '特定の出力値に対する変数間の傾向や影響の大きさを'
                   '直観的に把握できます。\n\n'
                   '**Tips: 軸は順番を入れ替えることができ、範囲選択することができます。**'
    )

    @staticmethod
    def create_plot(used_df):
        return parallel_plot(used_df)


class ContourPlot(SelectablePlot):

    plot_title = _('contour plot', 'コンタープロット')
    OutputItemsClass = dcc.RadioItems
    description_markdown: str = _(
        en_message='Visualize the correlation between input variables and changes in output using contour plots. '
                   'You can identify combinations of variables that have a strong influence.\n\n'
                   '**Tips: You can hide the scatter plot.**',
        jp_message='入力変数間の相関と、出力の変化をコンターで可視化。'
                   '影響の強い変数の組合せを確認できます。\n\n'
                   '**Tips: 点プロットは非表示にできます。**'
    )

    @staticmethod
    def create_plot(used_df):
        return contour_creator(used_df)


class SelectableOptunaPlot(SelectablePlot):

    def setup_update_plot_graph_callback(self):

        @self.application.app.callback(
            # graph output
            Output(self.graph, 'figure'),
            Output(self.alerts, 'children'),
            # checklist input
            inputs=dict(
                selected_input_values=Input(self.input_items, 'value'),
                selected_output_values=Input(self.output_items, 'value'),
            ),
        )
        def update_plot_graph(
                selected_input_values: list[str] | str,
                selected_output_values: list[str] | str,
        ):

            logger_name = f'opt.{type(self).__name__}.update_plot_graph()'

            logger = get_module_logger(
                logger_name,
                debug=False,
            )

            logger.debug('callback fired!')

            # ----- preconditions -----

            history, df, main_df = self._check_precondition(logger)

            # null selected values
            if selected_input_values is None:
                logger.debug('No input items.')
                return no_update, [dbc.Alert('No input items.', color='danger')]

            if selected_output_values is None:
                logger.debug('No output items.')
                return no_update, [dbc.Alert('No output items.', color='danger')]

            # type correction
            if isinstance(selected_input_values, str):
                selected_input_values = [selected_input_values]
            if isinstance(selected_output_values, str):
                selected_output_values = [selected_output_values]

            # nothing selected
            # selected_values = selected_input_values + selected_output_values
            # if len(selected_values) == 0:
            #     logger.debug('No items are selected.')
            #     return no_update, [dbc.Alert('No items are selected.', color='danger')]
            if len(selected_input_values) == 0:
                logger.debug('No input items are selected.')
                return no_update, [dbc.Alert('No input items are selected.', color='danger')]
            if len(selected_output_values) == 0:
                logger.debug('No output items are selected.')
                return no_update, [dbc.Alert('No output items are selected.', color='danger')]

            # ----- main -----
            fig = self.create_optuna_plot(
                history._create_optuna_study_for_visualization(),
                selected_input_values,
                selected_output_values,
                [history.all_output_names.index(v) for v in selected_output_values],
            )

            return fig, []

    @staticmethod
    def create_optuna_plot(
            study,
            prm_names: list[str],
            obj_name: list[str],
            obj_indices: list[int],
    ):

        raise NotImplementedError


class SelectableOptunaPlotAllInput(SelectablePlot):

    InputItemsClass = None
    OutputItemsClass = dcc.RadioItems

    def setup_update_plot_graph_callback(self):

        @self.application.app.callback(
            # graph output
            Output(self.graph, 'figure'),
            Output(self.alerts, 'children'),
            # checklist input
            inputs=dict(
                selected_output_value=Input(self.output_items, 'value'),
            ),
        )
        def update_plot_graph(
                selected_output_value: str,
        ):

            logger_name = f'opt.{type(self).__name__}.update_plot_graph()'

            logger = get_module_logger(
                logger_name,
                debug=False,
            )

            logger.debug('callback fired!')

            # ----- preconditions -----

            history, df, main_df = self._check_precondition(logger)

            # null selected values
            if selected_output_value is None:
                logger.debug('No output items.')
                return no_update, [dbc.Alert('No output items.', color='danger')]

            # ----- main -----
            fig = self.create_optuna_plot(
                history._create_optuna_study_for_visualization(),
                selected_output_value,
                history.all_output_names.index(selected_output_value)
            )

            return fig, []

    def setup_callback(self):
        self.setup_update_plot_output_checklist_callback()
        self.setup_update_plot_graph_callback()

    @staticmethod
    def create_optuna_plot(
            study, obj_name, obj_index,
    ):

        raise NotImplementedError


class ImportancePlot(SelectableOptunaPlotAllInput):

    plot_title = _('importance plot', '重要度プロット')
    description_markdown: str = _(
        en_message='Evaluate the importance of each input variable for the output using fANOVA. '
                   'You can quantitatively understand which inputs are important.',
        jp_message='出力に対する各入力変数の重要度を fANOVA で評価。'
                   '重要な入力を定量的に把握できます。'
    )

    @staticmethod
    def create_optuna_plot(
            study, obj_name, obj_index,
    ):

        # create plot using optuna
        fig = optuna.visualization.plot_param_importances(
            study,
            target=lambda trial: trial.values[obj_index],
            target_name=obj_name
        )
        fig.update_layout(
            title=f'Normalized importance of {obj_name}'
        )

        return fig


class HistoryPlot(SelectableOptunaPlotAllInput):

    plot_title = _('optimization history plot', '最適化履歴プロット')
    description_markdown: str = _(
        en_message='Display the history of outputs generated during optimization. '
                   'You can check the progress of improvements and the variability of the search.',
        jp_message='最適化中に生成された出力の履歴を表示。'
                   '改善の進行や探索のばらつきを確認できます。'
    )

    @staticmethod
    def create_optuna_plot(
            study, obj_name, obj_index,
    ):

        # create plot using optuna
        fig = optuna.visualization.plot_optimization_history(
            study,
            target=lambda trial: trial.values[obj_index],
            target_name=obj_name
        )
        fig.update_layout(
            title=f'Optimization history of {obj_name}'
        )

        return fig


class SlicePlot(SelectableOptunaPlot):

    plot_title = _('slice plot', 'スライスプロット')
    OutputItemsClass = dcc.RadioItems
    description_markdown: str = _(
        en_message='Displays the output response to a specific input. '
                   'You can intuitively see the univariate effect, ignoring other variables.',
        jp_message='特定の入力に対する出力の応答を表示。'
                   '他変数を無視した単変量の影響を'
                   '直観的に確認できます。'
    )

    @staticmethod
    def create_optuna_plot(
            study,
            prm_names: list[str],
            obj_names: list[str],
            obj_indices: list[int],
    ):

        assert len(obj_names) == len(obj_indices) == 1

        fig = optuna.visualization.plot_slice(
            study,
            params=prm_names,
            target=lambda trial: trial.values[obj_indices[0]],
            target_name=obj_names[0],
        )

        return fig
