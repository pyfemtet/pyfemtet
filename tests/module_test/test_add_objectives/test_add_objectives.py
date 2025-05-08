from pyfemtet.opt.optimizer import AbstractOptimizer
from pyfemtet.opt.interface import AbstractFEMInterface


def base_function_1(v):
    print('base_function_1 fired')
    return [v + i for i in range(2)]


def base_function_2(v):
    print('base_function_2 fired')
    return [v + i for i in range(3)]


# noinspection PyUnresolvedReferences
def test_set_obj_names_by_add_objectives():
    opt = AbstractOptimizer()
    opt.fem = AbstractFEMInterface()

    opt.add_objectives(
        names='base_name',
        fun=base_function_1,
        n_return=2,

    )

    opt.add_objectives(
        names=['name1', 'name2', 'name3'],
        fun=base_function_2,
        n_return=3,
        directions=[0, 'maximize', None]
    )

    opt._finalize_history()

    print(opt.history.obj_names)
    assert opt.history.obj_names == ['base_name_0', 'base_name_1', 'name1', 'name2', 'name3']
    print([obj.direction for obj in opt.objectives.values()])
    assert [obj.direction for obj in opt.objectives.values()] == ['minimize', 'minimize', 0, 'maximize', None]

    # 実際に目的関数を評価
    print(res := tuple(opt.objectives.values())[0].fun(0))
    assert res == 0

    # 二個目の実行は実際には評価せず記憶した値を呼び出すのみ
    print(res := tuple(opt.objectives.values())[1].fun(4))
    assert res == 1

    # すべての要素を呼び出した後は再び実際に目的関数を評価する
    print(res := tuple(opt.objectives.values())[1].fun(4))
    assert res == 5

    # 実際に目的関数を評価
    print(res := tuple(opt.objectives.values())[2].fun(0))
    assert res == 0

    # 記憶値を呼び出し
    print(res := tuple(opt.objectives.values())[3].fun(None))
    assert res == 1

    # すべての要素を呼び出さないうちは記憶値を呼び出す
    print(res := tuple(opt.objectives.values())[2].fun(None))
    assert res == 0


if __name__ == '__main__':
    test_set_obj_names_by_add_objectives()
