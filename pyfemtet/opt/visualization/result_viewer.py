import argparse


def show_static_monitor(csv_path):
    from pyfemtet.opt._femopt_core import History
    from pyfemtet.opt.visualization._monitor import ResultViewerApp
    _h = History(history_path=csv_path)
    _monitor = ResultViewerApp(history=_h)
    _monitor.run()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('csv_path', help='pyfemtet を実行した結果の csv ファイルのパスを指定してください。', type=str)

    args = parser.parse_args()

    if args.csv_path:
        show_static_monitor(args.csv_path)

