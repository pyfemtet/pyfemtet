from __future__ import annotations
import functools
from typing import Any, Callable, Optional, TypeVar, ParamSpec, Concatenate, Generic

P = ParamSpec("P")
R = TypeVar("R")


class _WithReopen(Generic[P, R]):
    """
    メソッド実行前にそのメソッドを定義したクラスの reopen を呼ぶデコレータ。
    """

    def __init__(self, func: Callable[Concatenate[Any, P], R]) -> None:
        # デコレータが読み込まれたとき、
        # つまりデコレート対象メソッドを持つクラスの定義中、
        # デコレート対象の関数オブジェクトの構築後に
        # デコレータでラップされた関数オブジェクトを作成するために
        # このクラスのインスタンスが作成される、つまりこれが呼ばれる。

        # デコレート対象のメソッド
        self.func = func
        self.__doc__ = func.__doc__

        # この時点ではデコレート対象メソッドの所属クラスは未構築なので
        # self.owner は None としておき、後で設定する。
        self.owner: Optional[type] = None

    def __set_name__(self, owner: type, name: str) -> None:
        # このインスタンスはメソッドをデコレートするので
        # owner.name に対応する関数オブジェクトとして owner クラスにバインドされる。
        # つまりクラス構築後にこれが呼ばれ、
        # 第一引数にデコレート対象メソッドを定義したクラスが渡される。

        # 所属クラスを保存しておく
        self.owner = owner

        # デコレートしたオブジェクトの名前は
        # クラスにバインドされた属性名にあわせる
        self.__name__ = name

    def __get__(self, instance: Any, owner: type) -> Callable[P, R]:
        # 実際にメソッドが呼ばれたときにこれが呼ばれる。

        # クラス属性経由: Child1.do_something(self) で呼ばれた場合
        if instance is None:
            # 第一引数がインスタンスのはずなので
            # それを取り出して自身をもう一度呼び出すラッパーを返す
            @functools.wraps(self.func)
            def class_call(inst: Any, *args: P.args, **kwargs: P.kwargs) -> R:
                return self.__get__(inst, owner)(*args, **kwargs)

            return class_call

        @functools.wraps(self.func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if self.owner is not None:
                rop = getattr(self.owner, "reopen", None)
                if rop is not None:
                    # owner 固定でバインドして呼ぶ
                    rop.__get__(instance, self.owner)()
            return self.func(instance, *args, **kwargs)

        return wrapper


with_reopen = _WithReopen
