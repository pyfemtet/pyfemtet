関数による上下限設定（dynamic_bounds）
====================

dynamic_bounds とは
--------------------

通常の拘束条件の代替として使える機能で、
**「ある変数の上下限が他の変数の関数で表せる」** 場合に利用できます。

pyfemtet では、通常の拘束条件は

- 変数を提案
- 拘束違反か判定
- 違反していれば再提案

という流れを取り、拘束が厳しい場合は 2→3 が何度も繰り返されます。
dynamic_bounds を使うと **不可能な提案がそもそも出ない** ため、この無駄が発生しません。

.. note::

    dynamic_bounds は拘束関数を高速化する機能ではなく、
    「上下限で書ける拘束」に限り、拘束そのものを置き換える方法です。

.. note::

    initial_value は常に上下限の範囲に入っている必要があります。


使える条件
-----------------------------
- 拘束が「ある変数の下限・上限が他の変数の関数で書ける」形にできること
- それ以外の拘束は通常どおり add_constraint を使います
（dynamic_bounds と通常拘束の併用は可能）


dynamic_bounds が利用可能な例
-----------------------------

問題
^^^^^^^^

- 変数 a: 下限 0, 上限 10
- 変数 b: 下限 0, 上限 10
- 拘束: a + b < 10

この問題は、以下のように書き換えることができます。

- 変数 a: 下限 0, 上限 10
- 変数 b: 下限 0, 上限 10 - a


dynamic_bounds の定義
^^^^^^^^

まず、上下限を計算する関数を定義します。

.. code-block:: python

    def dynamic_bounds_of_b(opt) -> tuple[float, float]:
        """b の動的な上下限を返す。"""
        params = opt.get_variables()
        return 0, 10. - params['a']

.. note::

    動的な上下限を計算する関数は、引数として opt: ``AbstractOptimizer`` を取り、
    戻り値として下限・上限を表すふたつの数値を返さなければなりません。


変数登録
^^^^^^^^

次に、変数追加時にその関数を登録します。


.. code-block:: python
    femopt.add_parameter(name='a', initial_value=0, lower_bound=0, upper_bound=10)
    femopt.add_parameter(
        name='b', initial_value=0,
        properties={
            'dynamic_bounds': dynamic_bounds_of_b
        }
    )

これにより、b の上下限は常に (0, 10 - a) となり、
a + b < 10 を満たさない組合せは最初から提案されなくなります。
