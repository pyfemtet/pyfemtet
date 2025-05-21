from pyfemtet.opt.optimizer import OptunaOptimizer
from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.exceptions import SolveError
from pyfemtet._i18n import ENCODING
from pyfemtet.opt.history._history import _RECORD_MESSAGE_DELIMITER

import pandas as pd


def obj(fem: NoFEM, opt: OptunaOptimizer):
    df = opt.history.get_df()

    if len(df) == 1:
        raise SolveError

    return 0.


def cns(fem: NoFEM, opt: OptunaOptimizer):
    df = opt.history.get_df()

    if len(df) == 2:
        return -1

    return 1


def test_save_worker_name():

    fem = NoFEM()

    opt = OptunaOptimizer()
    opt.fem = fem
    opt.n_trials = 2
    opt.add_parameter('x', 0, 0, 1)
    args = (opt,)
    opt.add_objective('y', obj, args=args)
    opt.add_constraint('c', cns, lower_bound=0, args=args)
    opt._worker_index = 0
    opt._worker_name = 'test main worker'
    opt.run()

    # save csv as ' | ' separated str
    df = pd.read_csv(opt.history.path, encoding=ENCODING, header=2)
    print(df['messages'].values.tolist())
    try:
        assert df['messages'].values.tolist() == [
            'test main worker',
            f'test main worker{_RECORD_MESSAGE_DELIMITER}Hidden constraint violation during objective function evaluation: SolveError',
            f'test main worker{_RECORD_MESSAGE_DELIMITER}Hard constraint violation: c_lower_bound',
            'test main worker',
        ]
    except AssertionError:
        assert df['messages'].values.tolist() == [
            'test main worker',
            f'test main worker{_RECORD_MESSAGE_DELIMITER}目的関数評価中の隠れた拘束違反：SolveError',
            f'test main worker{_RECORD_MESSAGE_DELIMITER}厳格拘束違反: c_lower_bound',
            'test main worker',
        ]



if __name__ == '__main__':
    test_save_worker_name()
