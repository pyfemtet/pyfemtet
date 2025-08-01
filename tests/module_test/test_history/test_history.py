import os
import csv
import pyfemtet
from pyfemtet.opt.history import *
from pyfemtet.opt.history._history import DuplicatedColumnNameError
from pyfemtet.opt.optimizer import AbstractOptimizer
from pyfemtet.opt.interface import NoFEM
from pyfemtet._i18n import ENCODING
import pytest
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
    print(f'{history.other_output_names=}')
    assert history.prm_names == ['x', 'y', 'z']
    assert history.obj_names == ['output']
    assert history.cns_names == ['constraint']
    assert history.other_output_names == ['other_output_name']

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
    co = {'version': pyfemtet.__version__}
    co.update(fem._get_additional_data())
    assert opt.history.additional_data == co


def test_restart_additional_data():
    history = History()
    history.load_csv(
        get(__file__, 'test_history_additional_data.reccsv'),
        with_finalize=True
    )
    print(history.additional_data)
    assert history.additional_data == dict(a="c:\\program files", b=None)


def test_history_column_order():

    history = History()
    history.column_order_mode = 'important_first'

    history.load_csv(os.path.join(os.path.dirname(__file__), 'test_history.reccsv'), with_finalize=True)
    print(history.get_df().columns)

    history.path = get(__file__, 'order_if.csv')
    history.save()

    with open(history.path, 'r', encoding=ENCODING) as f:
        reader = csv.reader(f)
        first_line = reader.__next__()
        _second_line = reader.__next__()
        third_line = reader.__next__()

    print(first_line)
    assert first_line == ['{}', 'prm.num.value', 'prm.num.value', 'prm.cat.value', 'obj', 'cns', 'other_output.value', '', '', 'prm.num.lower_bound', 'prm.num.upper_bound', 'prm.num.step', 'prm.num.lower_bound', 'prm.cat.choices', 'obj.direction', 'cns.upper_bound', '', '', '', '']
    print(third_line)
    assert third_line == ['trial', 'x', 'y', 'z', 'output', 'constraint', 'other_output_name', 'feasibility', 'optimality', 'x_lower_bound', 'x_upper_bound', 'x_step', 'y_lower_bound', 'z_choices', 'output_direction', 'constraint_upper_bound', 'state', 'datetime_start', 'datetime_end', 'messages']

    # ===== per_cat =====
    history = History()
    # history.column_order_mode = 'important_first'

    history.load_csv(os.path.join(os.path.dirname(__file__), 'test_history.reccsv'), with_finalize=True)
    print(history.get_df().columns)

    history.path = get(__file__, 'order_if.csv')
    history.save()

    with open(history.path, 'r', encoding=ENCODING) as f:
        reader = csv.reader(f)
        first_line = reader.__next__()
        _second_line = reader.__next__()
        third_line = reader.__next__()

    print(f'{first_line=}')
    assert first_line == ['{}', 'prm.num.value', 'prm.num.lower_bound', 'prm.num.upper_bound', 'prm.num.step', 'prm.num.value', 'prm.num.lower_bound', 'prm.cat.value', 'prm.cat.choices', 'obj', 'obj.direction', 'cns', 'cns.upper_bound', 'other_output.value', '', '', '', '', '', '']
    print(f'{third_line=}')
    assert third_line == ['trial', 'x', 'x_lower_bound', 'x_upper_bound', 'x_step', 'y', 'y_lower_bound', 'z', 'z_choices', 'output', 'output_direction', 'constraint', 'constraint_upper_bound', 'other_output_name', 'state', 'datetime_start', 'datetime_end', 'messages', 'feasibility', 'optimality']


def test_history_duplicated_name():
    opt = AbstractOptimizer()
    opt.fem = NoFEM()
    opt.add_parameter('a', 0, 0, 1)
    opt.add_objective('a', lambda: 1)
    try:
        opt._finalize()
    except DuplicatedColumnNameError:
        print('DuplicatedColumnNameError Successfully raised.')
    else:
        print('DuplicatedColumnNameError should be raised but not be raised!')
        assert False


@pytest.mark.manual
def test_history_duplicated_name_other_output():
    opt = AbstractOptimizer()
    opt.fem = NoFEM()
    opt.add_other_output('a', lambda: 1)
    opt.add_other_output('a', lambda: 1)
    print('Please check console output to certify logger.warning is successfully output.')
    if input('Press "NG" to set test failed.').lower() == 'ng':
        assert False


if __name__ == '__main__':
    test_standalone_history()
    test_new_additional_data()
    test_restart_additional_data()
    test_history_column_order()
    test_history_duplicated_name()
    test_history_duplicated_name_other_output()
