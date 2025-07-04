from pyfemtet.opt.visualization.history_viewer._base_application import *
from pyfemtet.opt.visualization.history_viewer._common_pages import *
from pyfemtet.opt.visualization.history_viewer.result_viewer._pages import *
from pyfemtet.opt.visualization.history_viewer._detail_page import DetailPage

from pyfemtet._i18n import Msg


class ResultViewerApplication(PyFemtetApplicationBase):

    def __init__(self):
        super().__init__(
            title='PyFemtet ResultView',
            subtitle='visualize optimization result',
            history=None,
        )

    def setup_callback(self):
        super().setup_callback()


def _debug():
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


def result_viewer_main():
    g_application = ResultViewerApplication()

    g_home_page = HomePage(Msg.PAGE_TITLE_RESULT)
    g_rsm_page = PredictionModelPage(Msg.PAGE_TITLE_PREDICTION_MODEL, '/prediction-model', g_application)
    g_detail = DetailPage(Msg.PAGE_TITLE_OPTUNA_VISUALIZATION, '/detail', g_application)

    g_application.add_page(g_home_page, 0)
    g_application.add_page(g_rsm_page, 1)
    g_application.add_page(g_detail, 2)
    g_application.setup_callback()

    g_application.run()


if __name__ == '__main__':
    result_viewer_main()
