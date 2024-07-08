from pyfemtet.opt.visualization.base import PyFemtetApplicationBase
from pyfemtet.opt.visualization.result_viewer.pages import HomePage


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

    g_home_page = HomePage('result')

    g_application.add_page(g_home_page, 0)
    g_application.setup_callback()

    g_application.run(debug=True)


def main():
    g_application = ResultViewerApplication()

    g_home_page = HomePage('result')

    g_application.add_page(g_home_page, 0)
    g_application.setup_callback()

    g_application.run()


if __name__ == '__main__':
    debug()
