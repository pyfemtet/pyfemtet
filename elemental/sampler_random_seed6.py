"""
目的：マルチプロセスにおいて seed を利かすための実験
目標：マルチプロセスにおいて load_study 直後に seed = seed + subprocess_id とする sampler を渡す
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


def f(sampler_class, sampler_args, seed, process_index):
    _sampler = sampler_class(*sampler_args, seed=seed+process_index)
    _study = optuna.load_study(
        study_name=f'{me.replace(".py", "")}',
        storage=f'sqlite:///{me.replace(".py", ".db")}',
        sampler=_sampler,
    )
    _study.optimize(objective, n_trials=3)
    # return _study


if __name__ == '__main__':

    # sampler setting
    RANDOM_SEED = 42
    sampler_args = tuple()
    sampler_class = optuna.samplers.TPESampler

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
        # sampler=sampler, # もしかしたら後で設定するならこの段階ではこれ不要？
        study_name=f'{me.replace(".py", "")}',
        storage=f'sqlite:///{me.replace(".py", ".db")}',
        # load_if_exists=True,
    )


    # process setting
    procs = []
    for i in range(3):
        p = Process(target=f, args=(sampler_class, sampler_args, RANDOM_SEED, i,))
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
# {'x': 0.11505456638977896, 'y': 0.6090665392794814}
# {'x': 0.8348421486656494, 'y': 0.10479610436986975}
# {'x': 0.7319939418114051, 'y': 0.5986584841970366}
# {'x': 0.13339096418598828, 'y': 0.24058961996534878}
# {'x': 0.7446404816795178, 'y': 0.3605008362562857}
# {'x': 0.35931083780807194, 'y': 0.6092383806181517}
# {'x': 0.3271390558111398, 'y': 0.8591374909485977}
# {'x': 0.15601864044243652, 'y': 0.15599452033620265}

# 2 回目
# {'x': 0.3745401188473625, 'y': 0.9507143064099162}
# {'x': 0.11505456638977896, 'y': 0.6090665392794814}
# {'x': 0.8348421486656494, 'y': 0.10479610436986975}
# {'x': 0.7319939418114051, 'y': 0.5986584841970366}
# {'x': 0.13339096418598828, 'y': 0.24058961996534878}
# {'x': 0.7446404816795178, 'y': 0.3605008362562857}
# {'x': 0.3271390558111398, 'y': 0.8591374909485977}
# {'x': 0.35931083780807194, 'y': 0.6092383806181517}
# {'x': 0.15601864044243652, 'y': 0.15599452033620265}

# 結果
# 順番は違うが別の試行を行っている

# 考察
# 予想通りかつ目的通りの挙動。これを実装する。

