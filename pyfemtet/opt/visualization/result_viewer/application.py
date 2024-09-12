from pyfemtet.opt.visualization.base import PyFemtetApplicationBase
from pyfemtet.opt.visualization.result_viewer.pages import HomePage, PredictionModelPage
from pyfemtet.opt.visualization.process_monitor.pages import OptunaVisualizerPage

from pyfemtet.message import Msg


class ResultViewerApplication(PyFemtetApplicationBase):

    def __init__(self):
        super().__init__(
            title='PyFemtet ResultView',
            subtitle='visualize optimization result',
            history=None,
        )

    def setup_callback(self):
        super().setup_callback()


def debug():
    import os
    os.chdir(os.path.dirname(__file__))

    g_application = ResultViewerApplication()

    g_home_page = HomePage(Msg.PAGE_TITLE_RESULT)
    g_rsm_page = PredictionModelPage(Msg.PAGE_TITLE_PREDICTION_MODEL, '/prediction-model', g_application)
    g_optuna = OptunaVisualizerPage(Msg.PAGE_TITLE_OPTUNA_VISUALIZATION, '/optuna', g_application)

    g_application.add_page(g_home_page, 0)
    g_application.add_page(g_rsm_page, 1)
    g_application.add_page(g_optuna, 2)
    g_application.setup_callback()

    g_application.run(debug=True)


def main():
    g_application = ResultViewerApplication()

    g_home_page = HomePage(Msg.PAGE_TITLE_RESULT)
    g_rsm_page = PredictionModelPage(Msg.PAGE_TITLE_PREDICTION_MODEL, '/prediction-model', g_application)
    g_optuna = OptunaVisualizerPage(Msg.PAGE_TITLE_OPTUNA_VISUALIZATION, '/optuna', g_application)

    g_application.add_page(g_home_page, 0)
    g_application.add_page(g_rsm_page, 1)
    g_application.add_page(g_optuna, 2)
    g_application.setup_callback()

    g_application.run()


if __name__ == '__main__':
    main()
