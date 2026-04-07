"""check_problem_compatibility のエラーメッセージ生成を検証する軽量テスト"""

import pytest
import pandas as pd

from pyfemtet.opt.history._history import Records, ColumnManager
from pyfemtet.opt.problem.variable_manager import NumericParameter


def _make_records(prm_names, obj_names, cns_names):
    """指定した名前で ColumnManager を初期化した Records を返す。"""
    records = Records()
    params = {}
    for name in prm_names:
        p = NumericParameter()
        p.name = name
        p.value = 0.0
        p.lower_bound = 0.0
        p.upper_bound = 1.0
        p.step = None
        params[name] = p

    records.column_manager.initialize(
        parameters=params,
        y_names=obj_names,
        c_names=cns_names,
        other_output_names=[],
        additional_data={},
    )
    return records


def _set_loaded(records, prm_names, obj_names, cns_names):
    """records.loaded_df / loaded_meta_columns を疑似的にセットする。"""
    # 一時的に同じ構造の Records を作ってカラム情報を借りる
    tmp = _make_records(prm_names, obj_names, cns_names)
    columns = list(tmp.column_manager.column_dtypes.keys())
    meta_columns = list(tmp.column_manager.meta_columns)
    records.loaded_df = pd.DataFrame(columns=columns)
    records.loaded_meta_columns = meta_columns


class TestCheckProblemCompatibility:
    """check_problem_compatibility が正しくエラーを出す / 出さないことを確認"""

    def test_compatible_no_error(self):
        """完全一致すればエラーにならない"""
        records = _make_records(["x", "y"], ["obj1"], ["cns1"])
        _set_loaded(records, ["x", "y"], ["obj1"], ["cns1"])
        records.check_problem_compatibility()  # should not raise

    def test_no_loaded_data(self):
        """loaded_df が None なら何もしない"""
        records = _make_records(["x"], ["obj1"], [])
        records.check_problem_compatibility()  # should not raise

    def test_obj_removed_is_ok(self):
        """目的関数が減っている分には OK"""
        records = _make_records(["x"], ["obj1"], [])
        _set_loaded(records, ["x"], ["obj1", "obj2"], [])
        records.check_problem_compatibility()  # should not raise

    def test_prm_added(self):
        """パラメータが追加されていたらエラー"""
        records = _make_records(["x", "y", "z"], ["obj1"], [])
        _set_loaded(records, ["x", "y"], ["obj1"], [])
        with pytest.raises(RuntimeError, match="z") as exc_info:
            records.check_problem_compatibility()
        print(f"\n--- test_prm_added ---\n{exc_info.value}")

    def test_prm_removed(self):
        """パラメータが削除されていたらエラー"""
        records = _make_records(["x"], ["obj1"], [])
        _set_loaded(records, ["x", "y"], ["obj1"], [])
        with pytest.raises(RuntimeError, match="y") as exc_info:
            records.check_problem_compatibility()
        print(f"\n--- test_prm_removed ---\n{exc_info.value}")

    def test_obj_added(self):
        """目的関数が追加されていたらエラー"""
        records = _make_records(["x"], ["obj1", "obj2"], [])
        _set_loaded(records, ["x"], ["obj1"], [])
        with pytest.raises(RuntimeError, match="obj2") as exc_info:
            records.check_problem_compatibility()
        print(f"\n--- test_obj_added ---\n{exc_info.value}")

    def test_cns_added(self):
        """拘束条件が追加されていたらエラー"""
        records = _make_records(["x"], ["obj1"], ["cns1", "cns2"])
        _set_loaded(records, ["x"], ["obj1"], ["cns1"])
        with pytest.raises(RuntimeError, match="cns2") as exc_info:
            records.check_problem_compatibility()
        print(f"\n--- test_cns_added ---\n{exc_info.value}")

    def test_cns_removed(self):
        """拘束条件が削除されていたらエラー"""
        records = _make_records(["x"], ["obj1"], ["cns1"])
        _set_loaded(records, ["x"], ["obj1"], ["cns1", "cns2"])
        with pytest.raises(RuntimeError, match="cns2") as exc_info:
            records.check_problem_compatibility()
        print(f"\n--- test_cns_removed ---\n{exc_info.value}")

    def test_multiple_incompatibilities_combined(self):
        """複数カテゴリの不整合がまとめて1つのエラーに含まれる"""
        records = _make_records(["x", "z"], ["obj1", "obj_new"], ["cns_new"])
        _set_loaded(records, ["x", "y"], ["obj1"], ["cns_old"])
        with pytest.raises(RuntimeError) as exc_info:
            records.check_problem_compatibility()
        msg = str(exc_info.value)
        print(f"\n--- test_multiple_incompatibilities_combined ---\n{exc_info.value}")
        # パラメータの差分
        assert "z" in msg
        assert "y" in msg
        # 目的関数の差分
        assert "obj_new" in msg
        # 拘束条件の差分
        assert "cns_new" in msg
        assert "cns_old" in msg

    def test_message_contains_remediation(self):
        """エラーメッセージに対処方法 (history_path, .db 削除) が含まれる"""
        records = _make_records(["x", "y"], ["obj1"], [])
        _set_loaded(records, ["x"], ["obj1"], [])
        with pytest.raises(RuntimeError) as exc_info:
            records.check_problem_compatibility()
        msg = str(exc_info.value)
        print(f"\n--- test_message_contains_remediation ---\n{exc_info.value}")
        assert "history_path" in msg
        assert ".db" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
