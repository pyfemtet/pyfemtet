ソレノイドコイルの自己インダクタンス
========================================

Femtet の磁場解析ソルバーを用い、有限長ソレノイドコイルの
自己インダクタンスを特定の値にする例題を解説します。


サンプルファイル
------------------------------
.. note::

   :download:`サンプルプロジェクト<../../../pyfemtet/FemtetPJTSample/gau_ex08_parametric.femprj>`
   を Femtet で開いたまま、
   :download:`サンプルコード<../../../pyfemtet/FemtetPJTSample/gau_ex08_parametric.py>`
   をダブルクリックして実行してください。

.. note::

   FEM 問題としての詳細については、FemtetHelp / 例題集 / 磁場解析 / 例題8 を参照してください。


設計変数
------------------------------

.. figure:: gau_ex08_model.png
   
   モデルの外観

====== ======
変数名 説明
====== ======
h      1巻きあたりのピッチ
r      コイルの半径
n      コイルの巻き数
====== ======


目的関数
------------------------------

自己インダクタンス


サンプルコード
------------------------------

.. literalinclude:: ../../../pyfemtet/FemtetPJTSample/gau_ex08_parametric.py
   :language: python
   :linenos:
   :caption: gau_ex08_parametric.py


サンプルコードの実行結果
------------------------------

.. figure:: gau_ex08_result.png
   :width: 300

   gau_ex08_parametric.py の実行結果。
   横軸が試行回数、縦軸が自己インダクタンスです。

20 回目の試行の結果、自己インダクタンスは 0.103 μF となります。

.. note::

   Femtet, pyfemtet および依存する最適化エンジンのバージョンにより、結果は多少異なる場合があります。
