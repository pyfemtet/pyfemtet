from optuna._hypervolume import wfg
from optuna._hypervolume import hssp
import numpy as np
from time import time
import plotly.graph_objs as go
import datetime
import optuna
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info, _get_hypervolume_history_plot
from pyfemtet.opt.history._hypervolume import calc_hypervolume
import os

here = os.path.dirname(__file__)


def test_optuna_hypervolume():
    np.random.seed(42)

    N = 500
    m = 3

    # y: N x m array
    y = np.array([
        np.random.randn(N) * np.random.rand() - np.random.rand() * np.random.rand()
        for _ in range(m)
    ]).T

    # f: N array
    v = np.random.rand(N)
    v[0] = 1.
    hard_f = np.array(v > 0.2)
    soft_f = np.array(v > 0.4)
    f = hard_f & soft_f
    y[~hard_f] = np.nan

    # 確認
    print(f"{y=}")
    print(f"{hard_f=}")
    print(f"{soft_f=}")
    print(f"{f=}")

    # 現在の手法で hv 計算
    start = time()
    hv_now = calc_hypervolume(y, f, 'optuna-nadir')
    end = time()

    title = f'optuna での {m} 目的 {N} データの処理時間: {int(end - start)} 秒'
    print(title)

    path = os.path.join(here, 'ref_hv.npy')

    # # save
    # np.save(path, hv_now)

    fig = go.Figure()
    fig.update_layout(title=title)
    fig.add_trace(go.Scatter(x=np.arange(len(hv_now)), y=hv_now, mode='markers+lines'))
    fig.show()

    hv_ref = np.load(path)
    assert np.allclose(hv_ref, hv_now, rtol=1.e-2)


if __name__ == '__main__':
    test_optuna_hypervolume()
