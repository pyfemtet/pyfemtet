外部 CAD (Solidworks) 連携
==============================

pyfemtet では外部 CAD (Solidworks) で作成したモデルを
Femtet にインポートして解析を行う場合でも
CAD と Femtet を連携して最適化を行うことができます。

Femtet の応力解析ソルバーを用いて
外部 CAD (Solidworks) でパラメトリックモデリングを行った H 型鋼について
体積を最小化しつつ
変位を最小にする
例題を解説します。

.. note::

    サンプルコード及び実行結果以外の項目は :doc:`../NX_ex01/NX_ex01` と同じです。



サンプルファイル
------------------------------
.. note::

   :download:`サンプルモデル<../../../pyfemtet/FemtetPJTSample/Sldworks_ex01/Sldworks_ex01.SLDPRT>`
   と
   :download:`サンプルプロジェクト<../../../pyfemtet/FemtetPJTSample/Sldworks_ex01/Sldworks_ex01.femprj>`
   を同じフォルダに配置し、
   プロジェクトを Femtet で開いたまま、
   :download:`サンプルコード<../../../pyfemtet/FemtetPJTSample/Sldworks_ex01/Sldworks_ex01.py>`
   をダブルクリックして実行してください。



サンプルコード
------------------------------

.. literalinclude:: ../../../pyfemtet/FemtetPJTSample/Sldworks_ex01/Sldworks_ex01.py
   :language: python
   :linenos:
   :caption: Sldworks_ex01.py


サンプルコードの実行結果
------------------------------

.. figure:: Sldworks_ex01_result.png
   :width: 300

   Sldworks_ex01.py の実行結果。
   横軸が 変位 、
   縦軸が 体積 です。

20 回目の試行の結果、
変位と体積のパレート集合が得られます。


.. note::

   Femtet, pyfemtet および依存する最適化エンジンのバージョンにより、結果は多少異なる場合があります。
