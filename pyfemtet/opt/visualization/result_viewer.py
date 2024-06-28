import argparse
from pyfemtet.opt.visualization._graphs import show_static_monitor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('csv_path', help='pyfemtet を実行した結果の csv ファイルのパスを指定してください。', type=str)

    args = parser.parse_args()

    if args.csv_path:
        show_static_monitor(args.csv_path)
