"""
目的：マルチプロセスにおいて seed を利かすための実験
目標：load_study で seed が効くかの確認
"""

import os
# from multiprocessing import Process
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
    return _study


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

    # load and then optimization
    study = f()

    # result
    for t in study.trials:
        print(t.params)


# 1 回目
# {'x': 0.5538925912833553, 'y': 0.8971144707986053}
# {'x': 0.625757123038565, 'y': 0.9201323806970965}
# {'x': 0.797215571593796, 'y': 0.10090499190934443}

# 2 回目
# {'x': 0.11484858968716272, 'y': 0.9607568405523439}
# {'x': 0.942774590879533, 'y': 0.028013017988325783}
# {'x': 0.6834696189236454, 'y': 0.6219035541669511}

# 結果
# seed は効かない

# 考察
# seed はやはり load_study した段階で reseed_rng() が呼ばれるために効かなくなる

