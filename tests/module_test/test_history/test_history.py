import os
from pyfemtet.opt.history import History


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


if __name__ == '__main__':
    test_standalone_history()
