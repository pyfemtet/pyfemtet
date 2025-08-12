import os
import numpy as np
from optuna.samplers import RandomSampler
from pyfemtet.opt.history import History
from pyfemtet.opt.optimizer._trial_queue import *
from pyfemtet.opt import OptunaOptimizer
from pyfemtet.opt.interface import NoFEM


here = os.path.dirname(__file__)


def cns(_, opt):
    return len(opt.history.get_df())


def obj(_):
    return 1.


# restart T/F
# include T/F
# constraint T/F


def test_no_add_trial_include():
    csv_path = os.path.join(here, 'test_no_add_trial_include.csv')
    db_path = csv_path.removesuffix('csv') + 'db'

    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    df = _impl_add_trial(
        n_trials=3,
        add_trial=False,
        csv_path=csv_path,
        include_queued_in_n_trials=True,
    )
    assert len(df) == 3


def test_add_trial_continue_include():
    csv_path = os.path.join(here, 'test_add_trial_continue_include.csv')
    db_path = csv_path.removesuffix('csv') + 'db'

    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    df = _impl_add_trial(
        n_trials=20,
        csv_path=csv_path,
        include_queued_in_n_trials=True,
    )
    print(f'途中で止めた場合: {len(df)}')
    assert len(df) == 20

    df = _impl_add_trial(
        n_trials=10,
        csv_path=csv_path,
        include_queued_in_n_trials=True,
    )
    print(f'途中で止めた場合: {len(df)}')
    assert len(df) == 30


def test_add_trial_continue():

    csv_path = os.path.join(here, 'test_add_trial_continue.csv')
    db_path = csv_path.removesuffix('csv') + 'db'

    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    df = _impl_add_trial(n_trials=-10, csv_path=csv_path)
    print('途中で止めた場合')
    print(f'{len(df)=}')
    assert len(df) == 17

    df = _impl_add_trial(n_trials=-2, csv_path=csv_path)
    print('続きから実行した場合')
    print(f'{len(df)=}')
    assert len(df) == 25

    df = _impl_add_trial(n_trials=3, csv_path=csv_path)
    print('さらに続きから実行した場合')
    print(f'{len(df)=}')
    assert len(df) == 29  # initial step と重複したステップがあるため


def test_add_trial_simple():
    csv_path = os.path.join(here, 'test_add_trial_simple.csv')
    db_path = csv_path.removesuffix('csv') + 'db'

    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    df = _impl_add_trial(csv_path=csv_path)
    assert len(df) == 27


def test_no_add_trial():
    csv_path = os.path.join(here, 'test_no_add_trial.csv')
    db_path = csv_path.removesuffix('csv') + 'db'

    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    df = _impl_add_trial(
        n_trials=3,
        add_trial=False,
        csv_path=csv_path,
    )
    assert len(df) == 3


def test_no_add_trial_add_constraint_continue():
    csv_path = os.path.join(here, 'test_no_add_trial_add_constraint_continue.csv')
    db_path = csv_path.removesuffix('csv') + 'db'

    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    df = _impl_add_trial(
        n_trials=5,
        add_trial=False,
        csv_path=csv_path,
        add_constraint=True,
    )
    print(f'{len(df)}')
    assert len(df) == 8

    df = _impl_add_trial(
        n_trials=5,
        add_trial=False,
        csv_path=csv_path,
        add_constraint=True,
    )
    print(f'{len(df)}')
    assert len(df) == 13


def test_continue_add_trial_add_constraint():
    csv_path = os.path.join(here, 'test_continue_add_trial_add_constraint.csv')
    db_path = csv_path.removesuffix('csv') + 'db'

    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    df = _impl_add_trial(
        n_trials=20,
        csv_path=csv_path,
        include_queued_in_n_trials=True,
        add_constraint=True,
    )
    print(f'途中で止めた場合: {len(df)}')
    assert len(df) == 23

    df = _impl_add_trial(
        n_trials=10,
        csv_path=csv_path,
        include_queued_in_n_trials=True,
        add_constraint=True,
    )
    print(f'途中で止めた場合: {len(df)}')
    assert len(df) == 33


def _impl_add_trial(
        n_trials=0,
        include_queued_in_n_trials=False,
        add_trial=True,
        csv_path=None,
        add_constraint=False,
):

    if csv_path is None:
        csv_path = os.path.join(here, '_impl_add_trial.csv')

    opt = OptunaOptimizer(
        sampler_kwargs=dict(seed=42)
    )
    opt.fem = NoFEM()

    opt.history.path = os.path.join(csv_path)

    opt.add_parameter('x1', 0, -1, 1)
    opt.add_parameter('x2', 0, -1, 1)
    opt.add_parameter('x3', 0, -1, 1)

    opt.add_objective('y', obj)

    if add_constraint:
        opt.add_constraint('c', cns, lower_bound=3, args=(opt,))

    if add_trial:
        for x1 in np.linspace(-1, 1, 3):
            for x2 in np.linspace(-1, 1, 3):
                for x3 in np.linspace(-1, 1, 3):
                    parameters = dict(
                        x1=x1,
                        x2=x2,
                        x3=x3,
                    )
                    opt.add_trial(parameters)

    opt.n_trials = n_trials
    opt.include_queued_in_n_trials = include_queued_in_n_trials
    opt.run()

    return opt.history.get_df()


def test_trial_queue():
    csv_path = os.path.join(here, 'test_trial_queue.reccsv')
    history = History()
    history.load_csv(csv_path, True)
    out = get_tried_list_from_history(
        history
    )
    for t in out:
        print(t)
    assert out == [
        {'x': 10.0, 'y': 6.0, 'z': 'A'},
        {'x': 20.0, 'y': 5.0, 'z': 'B'},
        {'x': 30.0, 'y': 4.0, 'z': 'C'},
        {'x': 40.0, 'y': 3.0, 'z': 'A'},
        {'x': 50.0, 'y': 2.0, 'z': 'B'},
        {'x': 60.0, 'y': 1.0, 'z': 'C'},
    ]


if __name__ == '__main__':

    test_no_add_trial()
    # test_add_trial_continue()
