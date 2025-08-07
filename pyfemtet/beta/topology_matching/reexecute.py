import os
from contextlib import closing

try:
    from win32com.client import Dispatch, constants
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Failed to import `win32com.client`. "
        "Please check the `pywin32` installation. "
        "Note that this feature is only for Windows."
    ) from e

try:
    from brepmatching.pyfemtet_scripts import Predictor
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Failed to import `brepmatching`. "
        "Please check the `brepmatching` installation by "
        "`pip install -U pyfemtet[matching]` "
        "or `pip install -U brepmatching`. "
    ) from e

from pyfemtet.logger import get_module_logger

logger = get_module_logger("brep", debug=False)


# モデルエリア全体を取得するための定数点
POINT1 = Dispatch("FemtetMacro.GaudiPoint")
POINT2 = Dispatch("FemtetMacro.GaudiPoint")
(POINT1.X, POINT1.Y, POINT1.Z) = (-500, -500, -500)
(POINT2.X, POINT2.Y, POINT2.Z) = (500, 500, 500)


def reexecute_model_with_topology_matching(
    Femtet,
    rebuild_fun: callable,
    parasolid_version=None,
    threshold=0.7,
    _image_path=None,
    args=None,
    kwargs=None,
):
    """Rebuild Femtet's model with topology matching.

    This feature tries to rebuild model
    with keeping geometric compatibility
    of boundary condition and mesh size assignment
    even if the internal ids of the topologies are changed.

    Note that the body number must be 1 in the model.

    Args:
        Femtet: The CDispatch instance of the Femtet
        rebuild_fun: The callable to rebuild the model in Femtet.
        parasolid_version: The win32.client.constants object
        threshold: Matching threshold.
        _image_path: Optional. Temporary file path.
        args: The args of ``rebuild_fun``. This is passed as *args.
        kwargs: The kwargs of ``rebuild_fun``. This is passed as **kwargs.

    Returns:
        None

    Raises:
        (RuntimeError, com_error): In case that the Femtet raises an error.
        AssertionError: In case that the body number is larger than 1.

    """

    args = args or ()
    kwargs = kwargs or {}

    # parasolid_version
    if parasolid_version is None:
        parasolid_version = constants.PARASOLID_VER_30_1_C

    # ===== 現在の Femtet に存在するモデルを取得する =====
    logger.debug("現在の Femtet のモデルをエクスポートしています。")

    # ボディキーを取得する
    succeed, all_bodies = Femtet.Gaudi.FindBodyAllByBox_py(
        Point1 := POINT1,
        Point2 := POINT2,
    )
    if not succeed:
        Femtet.ShowLastError()

    # ボディ数は 1 でないと非対応
    assert len(all_bodies) == 1, "ボディ数 1 を超える場合は matching は非対応です。"

    # パスを作る
    current_xt_path = os.path.abspath(
        "_tmp_current.x_t"
    )  # どこでもいいが絶対パス形式であること

    # 念のため存在していれば削除する
    if os.path.exists(current_xt_path):
        os.remove(current_xt_path)

    # エクスポート
    succeed = Femtet.Gaudi.Export_py(
        current_xt_path,  # FileName
        all_bodies,  # expBodies
        parasolid_version,  # ExpVer
        True,  # bForce
    )
    if not succeed:
        Femtet.ShowLastError()

    # 一応実行できたか確認する
    if not os.path.exists(current_xt_path):
        raise RuntimeError(
            f"Femtet のモデルを{current_xt_path}に出力することに失敗しました。"
        )

    # ===== モデルを置き換える前に境界条件の情報を取得する =====
    logger.debug(
        "現在の Femtet の境界条件・メッシュサイズとトポロジーの対応を取得しています。"
    )

    # すべてのトポロジーを取得する
    succeed, vertices, edges, faces = Femtet.Gaudi.FindTopologyAllByBox_py(
        POINT1, POINT2
    )
    if not succeed:
        Femtet.ShowLastError()

    # トポロジー ID に対して境界条件・メッシュサイズを調べる
    topo_id_vs_boundaries_org = {}
    topo_id_vs_mesh_org = {}
    for topo in vertices:
        boundary_names = [topo.Boundary(i) for i in range(topo.BoundaryNum)]
        id_ = topo.ID
        assert id_ not in topo_id_vs_boundaries_org.keys(), "重複したトポロジーID"
        if len(boundary_names) > 0:
            topo_id_vs_boundaries_org[id_] = boundary_names
        if topo.MeshSize != -1:
            topo_id_vs_mesh_org[id_] = topo.MeshSize
    for topo in edges:
        boundary_names = [topo.Boundary(i) for i in range(topo.BoundaryNum)]
        id_ = topo.ID
        assert id_ not in topo_id_vs_boundaries_org.keys(), "重複したトポロジーID"
        if len(boundary_names) > 0:
            topo_id_vs_boundaries_org[id_] = boundary_names
        if topo.MeshSize != -1:
            topo_id_vs_mesh_org[id_] = topo.MeshSize
    for topo in faces:
        boundary_names = [topo.Boundary(i) for i in range(topo.BoundaryNum)]
        id_ = topo.ID
        assert id_ not in topo_id_vs_boundaries_org.keys(), "重複したトポロジーID"
        if len(boundary_names) > 0:
            topo_id_vs_boundaries_org[id_] = boundary_names
        if topo.MeshSize != -1:
            topo_id_vs_mesh_org[id_] = topo.MeshSize

    # ===== モデルを置き換える =====
    logger.debug("新しいモデルに置き換えています。")
    rebuild_fun(*args, **kwargs)

    # ===== 置き換えたモデルをエクスポートする =====
    logger.debug("新しいモデルをもとに再構築されたモデルをエクスポートしています。")

    # ボディキーを取得する
    succeed, all_bodies = Femtet.Gaudi.FindBodyAllByBox_py(
        Point1 := POINT1,
        Point2 := POINT2,
    )
    if not succeed:
        Femtet.ShowLastError()

    # ボディ数は 1 でないと非対応
    assert len(all_bodies) == 1, (
        "再構築するとボディ数が 1 を超えました。この場合は matching は非対応です。"
    )

    # パスを作る
    var_xt_path = os.path.abspath("_tmp_variance.x_t")

    # 念のため存在していれば削除する
    if os.path.exists(var_xt_path):
        os.remove(var_xt_path)

    # エクスポート
    succeed = Femtet.Gaudi.Export_py(
        var_xt_path,  # FileName
        all_bodies,  # expBodies
        parasolid_version,  # ExpVer
        True,  # bForce
    )
    if not succeed:
        Femtet.ShowLastError()

    # 一応実行できたか確認する
    if not os.path.exists(var_xt_path):
        raise RuntimeError(
            "Femtet の Import コマンドでインポートしたモデルを"
            "出力することに失敗しました。"
        )

    # ===== マッチングを作る =====
    logger.debug("以前のモデルと新しいモデルのトポロジーの対応を計算しています。")

    # マッチングを作る
    predictor = Predictor(Femtet)
    with closing(predictor):
        exp_id_map = predictor.predict(
            current_xt_path,
            var_xt_path,
            threshold=threshold,
            _image_path=_image_path,
        )

    logger.debug(f"{current_xt_path=}")
    logger.debug(f"{var_xt_path=}")

    # export id と topology id の変換を取得する
    # noinspection PyUnresolvedReferences
    from coincidence_matching import get_topo_id_vs_export_id

    topo_id_vs_exp_id_org = get_topo_id_vs_export_id(current_xt_path, Femtet.hWnd)
    topo_id_vs_exp_id_var = get_topo_id_vs_export_id(var_xt_path, Femtet.hWnd)

    # matching を topology id から topology id へのマップにする
    def get_topo_id_from_exp_id(exp_id, topo_id_vs_exp_id):
        for _topo_id, _exp_id in topo_id_vs_exp_id.items():
            if exp_id == _exp_id:
                return _topo_id
        raise RuntimeError(f"{exp_id} is not found in specified map")

    topo_id_map = {}
    for exp_id_org, exp_id_var in exp_id_map.items():
        topo_id_org = get_topo_id_from_exp_id(exp_id_org, topo_id_vs_exp_id_org)
        topo_id_var = get_topo_id_from_exp_id(exp_id_var, topo_id_vs_exp_id_var)
        topo_id_map[topo_id_org] = topo_id_var

    # 新しいトポロジー ID に対して割り振られる境界条件・メッシュサイズを列挙する
    topo_id_vs_boundaries_var = {}
    topo_id_vs_mesh_var = {}
    for topo_id_org, boundary_names in topo_id_vs_boundaries_org.items():
        assert topo_id_org in topo_id_map.keys(), (
            f"境界条件が与えられたトポロジー {topo_id_org} のマッチング相手が見つかりませんでした。"
        )
        topo_id_var = topo_id_map[topo_id_org]
        topo_id_vs_boundaries_var[topo_id_var] = boundary_names
    for topo_id_org, mesh in topo_id_vs_mesh_org.items():
        topo_id_var = topo_id_map[topo_id_org]
        topo_id_vs_mesh_var[topo_id_var] = mesh

    # ===== モデルだけを置き換えた現在のプロジェクトから境界条件をすべて remove する =====
    logger.debug(
        "プロジェクトから古いトポロジー番号に割り当てられた境界条件・メッシュサイズを削除しています。"
    )

    succeed, vertices, edges, faces = Femtet.Gaudi.FindTopologyAllByBox_py(
        POINT1, POINT2
    )
    if not succeed:
        Femtet.ShowLastError()

    for topo in vertices:
        boundary_names = [topo.Boundary(i) for i in range(topo.BoundaryNum)]
        for boundary_name in boundary_names:
            succeed = topo.RemoveBoundary(boundary_name)
            if not succeed:
                Femtet.ShowLastError()
            if topo.MeshSize != -1.0:
                topo.MeshSize = -1.0
    for topo in edges:
        boundary_names = [topo.Boundary(i) for i in range(topo.BoundaryNum)]
        for boundary_name in boundary_names:
            succeed = topo.RemoveBoundary(boundary_name)
            if not succeed:
                Femtet.ShowLastError()
            if topo.MeshSize != -1.0:
                topo.MeshSize = -1.0
    for topo in faces:
        boundary_names = [topo.Boundary(i) for i in range(topo.BoundaryNum)]
        for boundary_name in boundary_names:
            succeed = topo.RemoveBoundary(boundary_name)
            if not succeed:
                Femtet.ShowLastError()
            if topo.MeshSize != -1.0:
                topo.MeshSize = -1.0
    Femtet.Gaudi.ReExecute()
    Femtet.Redraw()

    # ===== 境界条件を付けなおす =====
    logger.debug(
        "古いトポロジー番号に対応する新しいトポロジー番号に境界条件を再割り当てしています。"
    )

    succeed, vertices, edges, faces = Femtet.Gaudi.FindTopologyAllByBox_py(
        POINT1, POINT2
    )
    if not succeed:
        Femtet.ShowLastError()

    for topo in vertices:
        if topo.ID in topo_id_vs_boundaries_var.keys():
            boundaries = topo_id_vs_boundaries_var[topo.ID]
            for boundary in boundaries:
                succeed = topo.SetBoundary(boundary)
                if not succeed:
                    Femtet.ShowLastError()
        if topo.ID in topo_id_vs_mesh_var.keys():
            mesh_size = topo_id_vs_mesh_var[topo.ID]
            topo.MeshSize = mesh_size
    for topo in edges:
        if topo.ID in topo_id_vs_boundaries_var.keys():
            boundaries = topo_id_vs_boundaries_var[topo.ID]
            for boundary in boundaries:
                succeed = topo.SetBoundary(boundary)
                if not succeed:
                    Femtet.ShowLastError()
        if topo.ID in topo_id_vs_mesh_var.keys():
            mesh_size = topo_id_vs_mesh_var[topo.ID]
            topo.MeshSize = mesh_size
    for topo in faces:
        if topo.ID in topo_id_vs_boundaries_var.keys():
            boundaries = topo_id_vs_boundaries_var[topo.ID]
            for boundary in boundaries:
                succeed = topo.SetBoundary(boundary)
                if not succeed:
                    Femtet.ShowLastError()
        if topo.ID in topo_id_vs_mesh_var.keys():
            mesh_size = topo_id_vs_mesh_var[topo.ID]
            topo.MeshSize = mesh_size

    Femtet.Gaudi.ReExecute()
    Femtet.Redraw()

    # ===== 一時ファイルを削除する =====
    logger.debug("一時ファイルを削除しています。")
    if os.path.exists(current_xt_path):
        os.remove(current_xt_path)
    if os.path.exists(var_xt_path):
        os.remove(var_xt_path)
