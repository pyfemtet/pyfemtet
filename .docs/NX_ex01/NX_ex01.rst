外部 CAD (NX) 連携
==============================

pyfemtet では外部 CAD (NX) で作成したモデルを
Femtet にインポートして解析を行う場合でも
CAD と Femtet を連携して最適化を行うことができます。

Femtet の応力解析ソルバーを用いて
外部 CAD (NX) でパラメトリックモデリングを行った H 型鋼について
体積を最小化しつつ
変位を最小にする
例題を解説します。


サンプルファイル
------------------------------
.. note::

   :download:`サンプルモデル<../../pyfemtet/FemtetPJTSample/NX_ex01/NX_ex01.prt>`
   と
   :download:`サンプルプロジェクト<../../pyfemtet/FemtetPJTSample/NX_ex01/NX_ex01.femprj>`
   を同じフォルダに配置し、
   プロジェクトを Femtet で開き、
   :download:`サンプルコード<../../pyfemtet/FemtetPJTSample/NX_ex01/NX_ex01.py>`
   を実行してください。


FEM 問題としての詳細
------------------------------

.. figure:: NX_ex01_analysis.png
   
   モデルの外観 (解析条件)

- fix ... 完全固定
- load ... -Z 方向への荷重（1N）
- mirror ... YZ 平面に対して対称


設計変数
------------------------------

.. figure:: NX_ex01_model_dsgn.png
   
   モデルの外観 (設計変数)

====== ======
変数名 説明
====== ======
A      ウェブ板厚
B      フランジ板厚
C      フランジ曲げ
====== ======


目的関数
------------------------------

- Z 方向最大変位（0 にする）
- 体積（最小にする）


サンプルコード
------------------------------

.. literalinclude:: ../../pyfemtet/FemtetPJTSample/NX_ex01/NX_ex01.py
   :language: python
   :linenos:
   :caption: NX_ex01.py


サンプルコードの実行結果
------------------------------

.. figure:: NX_ex01_result.png
   :width: 300

   NX_ex01.py の実行結果。
   横軸が 変位 、
   縦軸が 体積 です。

20 回目の試行の結果、
変位と体積のパレート集合が得られます。


.. note::

   Femtet, pyfemtet および依存する最適化エンジンのバージョンにより、結果は多少異なる場合があります。
