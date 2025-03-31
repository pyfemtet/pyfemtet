import os
from pyfemtet.opt.history import History
from pyfemtet.opt.optimizer import AbstractOptimizer
from pyfemtet.opt.interface import NoFEM
from tests import get


class NoFEMWithAdditionalData(NoFEM):

    def _get_additional_data(self):
        return dict(a=1, b=None)


def test_standalone_history():
    history = History()
    history.load_csv(os.path.join(os.path.dirname(__file__), 'test_history.reccsv'), with_finalize=True)

    print(f'{history.prm_names=}')
    print(f'{history.obj_names=}')
    print(f'{history.cns_names=}')
    assert history.prm_names == ['x', 'y', 'z']
    assert history.obj_names == ['output']
    assert history.cns_names == ['constraint']

    df = history.get_df()
    df.to_csv(os.path.join(os.path.dirname(__file__), 'loaded_internal_df.csv'))

    history.path = os.path.join(os.path.dirname(__file__), 'saved_history.csv')
    history.save()


def test_new_additional_data():

    fem = NoFEMWithAdditionalData()

    opt = AbstractOptimizer()

    opt.fem = fem

    opt._finalize_history()

    c = list(opt.history.get_df().columns)
    m = opt.history._records.column_manager.meta_columns

    a_data = m[c.index(opt.history._records.column_manager._get_additional_data_column())]
    print(c)
    print(m)
    print(a_data, type(a_data))
    print(opt.history.additional_data)
    assert opt.history.additional_data == fem._get_additional_data()


def test_restart_additional_data():
    history = History()
    history.load_csv(
        get(__file__, 'test_history_additional_data.reccsv'),
        with_finalize=True
    )
    print(history.additional_data)
    assert history.additional_data == dict(a="c:\\program files", b=None)


if __name__ == '__main__':
    # test_standalone_history()
    # test_new_additional_data()
    test_restart_additional_data()
