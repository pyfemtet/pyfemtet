Welcome to PyFemtet's documentation!
====================================

pyfemtet は有限要素法解析ソフト `Femtet <https://www.muratasoftware.com/>`_ の
高機能な Python インターフェースのパッケージです。

pyfemtet は BSD-3 ライセンスの下で公開されているオープンソースライブラリです。
pyfemtet ライブラリは Femtet 本体のコア機能およびライセンスを変更することなく
Femtet の商用利用可能な機能拡充を無料で提供します。

現在の pyfemtet の唯一の主要なサブパッケージ **pyfemtet.opt** は
設計のパラメータ最適化を行うシンプルな API を提供します。



pyfemtet.opt の主要機能
----------------------------
- **単目的および多目的** の最適化に対応
- プロセスモニタによる **リアルタイム進行状況** の表示
- 複数の Femtet インスタンスによる **並列計算**
- Excel 等で分析が容易な csv 形式での結果管理


シンプルな API
----------------------------

下記は多目的最適化の実施例です。

.. code-block:: python

   from pyfemtet.opt import OptimizerOptuna

   def max_displacement(Femtet):
       dx, dy, dz = Femtet.Gogh.Galileo.GetMaxDisplacement()
       return dy

   def volume(Femtet):
       w = Femtet.GetVariableValue('w')
       d = Femtet.GetVariableValue('d')
       h = Femtet.GetVariableValue('h')
       return w * d * h

   if __name__ == '__main__':
       femopt = OptimizerOptuna()
       femopt.add_parameter('w', 10, 2, 20)
       femopt.add_parameter('d', 10, 2, 20)
       femopt.add_objective(max_displacement, name='最大変位', direction=0)
       femopt.add_objective(volume, name='体積', direction='minimize')
       femopt.set_random_seed(42)
       femopt.main(n_trials=20)

.. note::

   :download:`サンプル プロジェクト<opt_sample.femprj>` をダウンロードし、
    Femtet で開いた状態で上記スクリプトを実行してください。

使用方法の詳細は :doc:`usage` セクションを確認してください。


実施例
-----
※サムネイルと FemtetPrjSample みたいな例を作って入れる





Contents
--------

.. toctree::

   Home <self>
   usage.md
   api


Copyright (C) 2023 Murata Manufacturing Co., Ltd. All Rights Reserved.
Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.