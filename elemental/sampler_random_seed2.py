"""
目的：マルチプロセスにおいて seed を利かすための実験
目標：マルチプロセスで seed が効くかの確認
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


def f():
    _study = optuna.load_study(
        study_name=f'{me.replace(".py", "")}',
        storage=f'sqlite:///{me.replace(".py", ".db")}'
    )
    _study.optimize(objective, n_trials=3)


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
        p = Process(target=f)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    study = optuna.load_study(
        study_name=f'{me.replace(".py", "")}',
        storage=f'sqlite:///{me.replace(".py", ".db")}'
    )

    for t in study.trials:
        print(t.params)


# 1 回目
# {'x': 0.5823969095472119, 'y': 0.3762802018747}
# {'x': 0.2828600927073819, 'y': 0.4741323781687615}
# {'x': 0.8472595605771589, 'y': 0.11556020116549592}
# {'x': 0.406924505006491, 'y': 0.06680396974377079}
# {'x': 0.15521874721873774, 'y': 0.53933566606965}
# {'x': 0.5073403172669668, 'y': 0.5986863700558612}
# {'x': 0.6344041675413732, 'y': 0.3205098023941819}
# {'x': 0.8149031700436502, 'y': 0.7772193045802201}
# {'x': 0.5611223570239943, 'y': 0.6261648220395272}

# 2 回目
# {'x': 0.8867104834133772, 'y': 0.7518041747045389}
# {'x': 0.30452573775059166, 'y': 0.3279020330357406}
# {'x': 0.22588596682432327, 'y': 0.2216721820006622}
# {'x': 0.14834280123669175, 'y': 0.8970728413532157}
# {'x': 0.1802566848764232, 'y': 0.5674760109170519}
# {'x': 0.29005533204332934, 'y': 0.06794384407770282}
# {'x': 0.29992315737175124, 'y': 0.9094312137068865}
# {'x': 0.7234905783459394, 'y': 0.16198809511113743}
# {'x': 0.2134462531648602, 'y': 0.9602373248482184}

# 結果
# seed は効かない

# 考察
# seed はおそらく load_study した段階で reseed_rng() が呼ばれるために効かなくなる

