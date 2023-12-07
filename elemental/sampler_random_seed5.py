"""
目的：マルチプロセスにおいて seed を利かすための実験
目標：マルチプロセスにおいて load_study 直後に sampler を渡したときの挙動を確認
"""

import os
from multiprocessing import Process
import optuna

here, me = os.path.split(__file__)
os.chdir(here)


def objective(trial):
    x = trial.suggest_float('x', 0, 1)
    y = trial.suggest_float('y', 0, 1)
    return x**2 + y**2


def f(sampler):
    _study = optuna.load_study(
        study_name=f'{me.replace(".py", "")}',
        storage=f'sqlite:///{me.replace(".py", ".db")}',
        sampler=sampler,
    )
    _study.optimize(objective, n_trials=3)
    # return _study


if __name__ == '__main__':

    # sampler setting
    sampler = optuna.samplers.TPESampler(seed=42)

    # delete previous study
    try:
        optuna.delete_study(
            study_name=f'{me.replace(".py", "")}',
            storage=f'sqlite:///{me.replace(".py", ".db")}',
        )
    except:
        pass

    # study setting
    optuna.create_study(
        sampler=sampler,
        study_name=f'{me.replace(".py", "")}',
        storage=f'sqlite:///{me.replace(".py", ".db")}',
        # load_if_exists=True,
    )


    # process setting
    procs = []
    for i in range(3):
        p = Process(target=f, args=(sampler,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # result
    study = optuna.load_study(
        study_name=f'{me.replace(".py", "")}',
        storage=f'sqlite:///{me.replace(".py", ".db")}'
    )
    for t in study.trials:
        print(t.params)


# 1 回目
# {'x': 0.3745401188473625, 'y': 0.9507143064099162}
# {'x': 0.7319939418114051, 'y': 0.5986584841970366}
# {'x': 0.15601864044243652, 'y': 0.15599452033620265}
# {'x': 0.3745401188473625, 'y': 0.9507143064099162}
# {'x': 0.7319939418114051, 'y': 0.5986584841970366}
# {'x': 0.15601864044243652, 'y': 0.15599452033620265}
# {'x': 0.3745401188473625, 'y': 0.9507143064099162}
# {'x': 0.7319939418114051, 'y': 0.5986584841970366}
# {'x': 0.15601864044243652, 'y': 0.15599452033620265}


# 結果
# 同じ seed が渡されたら 違う Process も同じ試行をする

# 考察
# 予想通りの挙動。
# args を保存して seed = seed + proc_id とするのがいいだろう

