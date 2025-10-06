from __future__ import annotations

from typing import TYPE_CHECKING

import os
import gc
import tempfile
from time import sleep
from pathlib import Path

import numpy as np

from win32com.client import DispatchEx, Dispatch
from win32com.client.dynamic import CDispatch
# noinspection PyUnresolvedReferences
from pythoncom import CoInitialize, CoUninitialize
# noinspection PyUnresolvedReferences
from pywintypes import com_error
from femtetutils import util

from pyfemtet._util.dask_util import *
from pyfemtet._util.process_util import *
from pyfemtet._util.excel_parse_util import *
from pyfemtet._util.excel_macro_util import *
from pyfemtet._util.femtet_autosave import *
from pyfemtet._i18n import _

from pyfemtet.opt.exceptions import *
from pyfemtet.opt.problem.problem import *
from pyfemtet.opt.interface._base_interface import COMInterface

from pyfemtet.logger import get_module_logger
from pyfemtet._util.helper import float_

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer
    from pyfemtet.opt.problem.problem import Objective, Constraint

logger = get_module_logger('opt.interface')


class WorkbookNotOpenedError(Exception):
    pass


class WorkSheetNotFoundError(Exception):
    pass


class ExcelInterface(COMInterface):
    """Excel を計算コアとして利用するためのクラス。

        通常の有限要素法を Excel に
        置き換えて使用することが可能です。

        すでに Excel マクロと Femtet を
        連携させた自動解析システムを
        構築している場合、このクラスは
        それをラップします。これにより、
        PyFemtet を用いた最適化を
        行う際に便利な機能を提供します。

        Args:
            input_xlsm_path (str or Path):
                設計変数の定義を含む Excel ファイルのパスを指定
                します。

            input_sheet_name (str):
                設計変数の定義を含むシートの名前を指定します。

            output_xlsm_path (str or Path, optional):
                目的関数の定義を含む Excel ファイルのパスを指定
                します。指定しない場合は ``input_xlsm_path`` と
                同じと見做します。

            output_sheet_name (str, optional):
                目的関数の定義を含む含むシートの名前を指定します。
                指定しない場合は ``input_sheet_name`` と同じと見
                做します。

            procedure_xlsm_path (str, optional):
                最適化ループ中に呼ぶ Excel マクロ関数を
                含む xlsm のパスです。
                指定しない場合は ``input_xlsm_path`` と
                同じと見做します。

            procedure_name (str, optional):
                Excel マクロ関数名を指定します。指定しない場合は
                ``FemtetMacro.FemtetMain`` と見做します。

            procedure_args (list or tuple, optional):
                Excel マクロ関数に渡す引数をリストまたはタプルで
                指定します。

            connect_method (str, optional):
                Excel との接続方法を指定します。 'auto' または
                'new' が利用可能です。デフォルトは 'auto' です。

            procedure_timeout (float or None, optional):
                Excel マクロ関数のタイムアウト時間を秒単位で指定
                します。 None の場合はタイムアウトなしとなります。

            setup_xlsm_path (str or Path, optional):
                セットアップ時に呼ぶ関数を含む xlsm のパスです。
                指定しない場合は ``input_xlsm_path`` と
                同じと見做します。

            setup_procedure_name (str, optional):
                セットアップ時に呼ぶマクロ関数名です。
                指定しない場合、セットアップ時に何もしません。

            setup_procedure_args (list or tuple, optional):
                セットアップ時に呼ぶマクロ関数の引数です。

            teardown_xlsm_path (str or Path, optional):
                終了時に呼ぶ関数を含む xlsm のパスです。
                指定しない場合は ``input_xlsm_path`` と
                同じと見做します。

            teardown_procedure_name (str, optional):
                終了時に呼ぶマクロ関数名です。
                指定しない場合、終了時に何もしません。

            teardown_procedure_args (list or tuple, optional):
                終了時に呼ぶマクロ関数の引数です。

            visible (bool):
                excel を可視化するかどうかです。
                ただし、 True を指定した場合でもマクロの実行中は
                不可視になります。
                デフォルトは False です。

            display_alerts (bool):
                excel ダイアログを表示するかどうかです。
                デバッグ目的の場合以外は True にしないでください。
                デフォルトは False です。

            terminate_excel_when_quit (bool):
                終了時に Excel を終了するかどうかです。
                指定しない場合、 connect_method が 'new' の場合
                True とふるまい 'auto' の場合 False と振舞います。

            interactive (bool):
                excel を対話モードにするかどうかです。
                False にすると、 visible == True であっても
                自動化プロセス中にユーザーが誤って
                Excel 本体を操作できないようにします。
                デフォルトは True です。

        Attributes:
            input_xlsm_path (Path):
                設計変数の定義を含む Excel ファイルのパス。

            input_sheet_name (str):
                設計変数の定義を含むシートの名前。

            output_xlsm_path (Path):
                目的関数の定義を含む Excel ファイルのパス。

            output_sheet_name (str):
                目的関数の定義を含む含むシートの名前。

            procedure_name (str):
                実行する Excel マクロ関数名。

            procedure_args (list or tuple):
                Excel マクロ関数に渡す引数のリストまたはタプル。

            excel_connect_method (str):
                接続方法。'new' または 'auto'。

            procedure_timeout (float or None):
                Excel マクロ関数の実行タイムアウト。
                Noneの場合は無制限。

            terminate_excel_when_quit (bool):
                プログラム終了時に Excel を終了するかどうか。
                connect_method が 'new' の場合 True,
                'auto' の場合 False。

            excel (CDispatch):
                Excel の COM オブジェクト。

        """

    input_xlsm_path: str  # 操作対象の xlsm パス
    input_sheet_name: str  # 変数セルを定義しているシート名
    output_xlsm_path: str  # 操作対象の xlsm パス (指定しない場合、input と同一)
    output_sheet_name: str  # 計算結果セルを定義しているシート名 (指定しない場合、input と同一)
    constraint_xlsm_path: str  # 操作対象の xlsm パス (指定しない場合、input と同一)
    constraint_sheet_name: str  # 拘束関数セルを定義しているシート名 (指定しない場合、input と同一)

    related_file_paths: list[str]  # 並列時に個別に並列プロセスの space にアップロードする必要のあるパス

    procedure_name: str  # マクロ関数名（or モジュール名.関数名）
    procedure_args: list  # マクロ関数の引数

    excel: CDispatch  # Excel Application
    wb_input: CDispatch  # システムを構成する Workbook
    sh_input: CDispatch  # 変数の定義された WorkSheet
    wb_output: CDispatch  # システムを構成する Workbook
    sh_output: CDispatch  # 計算結果の定義された WorkSheet (sh_input と同じでもよい)
    wb_constraint: CDispatch  # システムを構成する Workbook
    sh_constraint: CDispatch  # 計算結果の定義された WorkSheet (sh_input と同じでもよい)
    wb_procedure: CDispatch  # システムを構成する Workbook
    wb_setup: CDispatch  # システムを構成する Workbook
    wb_teardown: CDispatch  # システムを構成する Workbook

    visible: bool  # excel を可視化するかどうか
    display_alerts: bool  # ダイアログを表示するかどうか
    terminate_excel_when_quit: bool  # 終了時に Excel を終了するかどうか
    interactive: bool  # excel を対話モードにするかどうか

    _load_problem_from_fem = True
    _excel_pid: int
    _excel_hwnd: int
    _with_femtet_autosave_setting: bool = True  # Femtet の自動保存機能の自動設定を行うかどうか。Femtet がインストールされていない場合はオフにする。クラス変数なので、インスタンス化前に設定する。
    _femtet_autosave_buffer: bool  # Femtet の自動保存機能の一時退避場所。最適化中はオフにする。
    com_members = {'excel': 'Excel.Application'}

    setup_xlsm_path: str
    setup_procedure_name: str
    setup_procedure_args: list or tuple
    teardown_xlsm_path: str
    teardown_procedure_name: str
    teardown_procedure_args: list or tuple

    use_named_range: bool  # input を定義したシートにおいて input の値を名前付き範囲で指定するかどうか。
    force_override_when_load: bool  # すでに指定されている parameter 等を excel の load 時に上書きするかどうか。

    _tmp_dir: tempfile.TemporaryDirectory

    def __init__(
            self,
            input_xlsm_path: str or Path,
            input_sheet_name: str,
            output_xlsm_path: str or Path = None,
            output_sheet_name: str = None,
            constraint_xlsm_path: str or Path = None,
            constraint_sheet_name: str = None,
            procedure_xlsm_path: str or Path = None,
            procedure_name: str = None,
            procedure_args: list or tuple = None,
            connect_method: str = 'new',  # or 'auto'
            procedure_timeout: float or None = None,
            setup_xlsm_path: str or Path = None,
            setup_procedure_name: str = None,
            setup_procedure_args: list or tuple = None,
            teardown_xlsm_path: str or Path = None,
            teardown_procedure_name: str = None,
            teardown_procedure_args: list or tuple = None,
            related_file_paths: list[str or Path] = None,
            visible: bool = False,
            display_alerts: bool = False,
            terminate_excel_when_quit: bool = None,
            interactive: bool = True,
            use_named_range: bool = True,
            force_override_when_load: bool = False,
    ):

        def proc_path(path_):
            if path_ is None:
                return self._original_input_xlsm_path
            else:
                ret_ = os.path.abspath(path_)
                assert os.path.isfile(ret_), f'{ret_} が見つかりません。'
                return os.path.abspath(path_)

        # 初期化
        self._original_input_xlsm_path = os.path.abspath(str(input_xlsm_path))
        assert os.path.isfile(self._original_input_xlsm_path), f'{self._original_input_xlsm_path} が見つかりません。'
        self._original_output_xlsm_path = proc_path(output_xlsm_path)
        self._original_constraint_xlsm_path = proc_path(constraint_xlsm_path)
        self._original_procedure_xlsm_path = proc_path(procedure_xlsm_path)
        self._original_setup_xlsm_path = proc_path(setup_xlsm_path)
        self._original_teardown_xlsm_path = proc_path(teardown_xlsm_path)
        self._original_related_file_paths = \
            [proc_path(p) for p in related_file_paths] \
            if related_file_paths is not None else []

        self.input_xlsm_path = self._original_input_xlsm_path
        self.output_xlsm_path = self._original_output_xlsm_path
        self.constraint_xlsm_path = self._original_constraint_xlsm_path
        self.procedure_xlsm_path = self._original_procedure_xlsm_path
        self.setup_xlsm_path = self._original_setup_xlsm_path
        self.teardown_xlsm_path = self._original_teardown_xlsm_path
        self.related_file_paths = self._original_related_file_paths

        self.input_sheet_name = input_sheet_name
        self.output_sheet_name = output_sheet_name if output_sheet_name is not None else input_sheet_name
        self.constraint_sheet_name = constraint_sheet_name or self.input_sheet_name
        self.procedure_name = procedure_name
        self.procedure_args = procedure_args or []
        assert connect_method in ['new', 'auto']
        self.excel_connect_method = connect_method
        self.procedure_timeout = procedure_timeout
        if terminate_excel_when_quit is None:
            self.terminate_excel_when_quit = self.excel_connect_method == 'new'
        else:
            self.terminate_excel_when_quit = terminate_excel_when_quit

        self.setup_procedure_name = setup_procedure_name
        self.setup_procedure_args = setup_procedure_args or []

        self.teardown_procedure_name = teardown_procedure_name
        self.teardown_procedure_args = teardown_procedure_args or []

        self.visible = visible
        self.interactive = interactive
        self.display_alerts = display_alerts

        self.use_named_range = use_named_range
        self.force_override_when_load = force_override_when_load

    @property
    def object_pass_to_fun(self):
        """The object pass to the first argument of user-defined objective functions.

        Returns:
            excel (CDispatch): COM object of Microsoft Excel.
        """
        return self.excel

    # ===== setup =====
    def _setup_before_parallel(self, scheduler_address=None) -> None:
        # メインプロセスで、並列プロセスを開始する前に行う前処理

        related_files = [self._original_input_xlsm_path]

        if not _is_same_path(self._original_input_xlsm_path, self._original_output_xlsm_path):
            related_files.append(self._original_output_xlsm_path)

        if not _is_same_path(self._original_input_xlsm_path, self._original_constraint_xlsm_path):
            related_files.append(self._original_constraint_xlsm_path)

        if not _is_same_path(self._original_input_xlsm_path, self._original_procedure_xlsm_path):
            related_files.append(self._original_procedure_xlsm_path)

        if not _is_same_path(self._original_input_xlsm_path, self._original_setup_xlsm_path):
            related_files.append(self._original_setup_xlsm_path)

        if not _is_same_path(self._original_input_xlsm_path, self._original_teardown_xlsm_path):
            related_files.append(self._original_teardown_xlsm_path)

        related_files.extend(self._original_related_file_paths)

        # dask worker 向け
        self._distribute_files(related_files, scheduler_address)

    def _re_register_paths(self, suffix):
        # self.hoge_path を dask worker space のファイルに変える

        # paths を更新
        #   suffix の付与は connect_method が auto の場合
        #   同名ファイルを開かないようにするための配慮
        #   output 等 == inputの場合、
        #   最初に input を rename しているので
        #   すでに rename されている
        self.input_xlsm_path = self._rename_and_get_path_on_worker_space(
            self._original_input_xlsm_path, suffix)
        self.output_xlsm_path = self._rename_and_get_path_on_worker_space(
            self._original_output_xlsm_path, suffix, True)
        self.constraint_xlsm_path = self._rename_and_get_path_on_worker_space(
            self._original_constraint_xlsm_path, suffix, True)
        self.procedure_xlsm_path = self._rename_and_get_path_on_worker_space(
            self._original_procedure_xlsm_path, suffix, True)
        self.setup_xlsm_path = self._rename_and_get_path_on_worker_space(
            self._original_setup_xlsm_path, suffix, True)
        self.teardown_xlsm_path = self._rename_and_get_path_on_worker_space(
            self._original_teardown_xlsm_path, suffix, True)

    def _setup_after_parallel(self, opt: AbstractOptimizer):
        """サブプロセス又はメインプロセスのサブスレッドで、最適化を開始する前の前処理"""

        # 最適化中は femtet の autosave を無効にする
        if self._with_femtet_autosave_setting:
            self._femtet_autosave_buffer = _get_autosave_enabled()
            _set_autosave_enabled(False)

        # Excel の場合はスレッドが変わっただけでも初期化が必要
        CoInitialize()

        # 元のファイルを保護するため space のファイルを使用
        suffix = self._get_file_suffix(opt)
        self._re_register_paths(suffix)

        # excel に繋ぐ
        with Lock('connect-excel'):
            self.connect_excel(self.excel_connect_method)
            sleep(1)

        # load_objective で optimizer に登録した Function は
        # 少なくともスレッドが異なるので
        # 現在のスレッドのオブジェクトに参照しなおす
        obj: Objective
        for obj_name, obj in opt.objectives.items():
            if isinstance(obj.fun, _ScapeGoatObjective):
                opt.objectives[obj_name].fun = self.objective_from_excel

        cns: Constraint
        for cns_name, cns in opt.constraints.items():
            if isinstance(cns.fun, _ScapeGoatObjective):
                opt.constraints[cns_name].fun = self.constraint_from_excel

        # excel の setup 関数を必要なら実行する
        if self.setup_procedure_name is not None:

            logger.info(
                _(
                    en_message='{procedure_kind} procedure {procedure_name} is running...',
                    jp_message='{procedure_kind} プロシージャ {procedure_name} を実行中です...',
                    procedure_kind='Setup',
                    procedure_name=self.setup_procedure_name
                )
            )

            with Lock('excel_setup_procedure'):
                try:
                    self.wb_setup.Activate()
                    sleep(0.1)
                    with watch_excel_macro_error(self.excel, timeout=self.procedure_timeout, restore_book=False):
                        self.excel.Run(
                            f'{self.setup_procedure_name}',
                            *self.setup_procedure_args
                        )

                    # 再計算
                    self.excel.CalculateFull()
                    sleep(1)

                except com_error as e:
                    raise RuntimeError(
                        _(
                            en_message='Failed to run macro {procedure_name}. The original message is: {exception}',
                            jp_message='マクロ {procedure_name} の実行に失敗しました。エラーメッセージ: {exception}',
                            procedure_name=self.setup_procedure_name,
                            exception=e
                        )
                    )

    def connect_excel(self, connect_method):

        # ===== 新しい excel instance を起動 =====
        # 起動
        logger.info(_(
            en_message='Launching and connecting to Microsoft Excel...',
            jp_message='Microsoft Excel を起動して接続しています...',
        ))
        if connect_method == 'auto':
            self.excel = Dispatch('Excel.Application')
        else:
            self.excel = DispatchEx('Excel.Application')
        logger.info(_(
            en_message='The connection to Excel is established.',
            jp_message='Excel への接続が確立されました。',
        ))
        # FemtetRef.xla を開く
        self.open_femtet_ref_xla()
        sleep(0.5)

        # 起動した excel の pid を記憶する
        self._excel_hwnd = self.excel.hWnd
        self._excel_pid = 0
        while self._excel_pid == 0:
            sleep(0.5)
            self._excel_pid = _get_pid(self.excel.hWnd)

        # 可視性の設定
        self.excel.Visible = self.visible
        self.excel.DisplayAlerts = self.display_alerts
        self.excel.Interactive = self.interactive
        sleep(0.5)

        # open
        self.excel.Workbooks.Open(self.input_xlsm_path)
        sleep(0.1)
        self.excel.Workbooks.Open(self.output_xlsm_path)
        sleep(0.1)
        self.excel.Workbooks.Open(self.constraint_xlsm_path)
        sleep(0.1)
        self.excel.Workbooks.Open(self.procedure_xlsm_path)
        sleep(0.1)
        self.excel.Workbooks.Open(self.setup_xlsm_path)
        sleep(0.1)
        self.excel.Workbooks.Open(self.teardown_xlsm_path)
        sleep(0.1)

        # book に参照設定を追加する
        self.add_femtet_macro_reference(self.wb_input)
        self.add_femtet_macro_reference(self.wb_output)
        self.add_femtet_macro_reference(self.wb_setup)
        self.add_femtet_macro_reference(self.wb_teardown)
        self.add_femtet_macro_reference(self.wb_constraint)

    def open_femtet_ref_xla(self):

        # get 64 bit
        xla_file_path = r'C:\Program Files\Microsoft Office\root\Office16\XLSTART\FemtetRef.xla'

        # if not exist, get 32bit
        if not os.path.exists(xla_file_path):
            xla_file_path = r'C:\Program Files (x86)\Microsoft Office\root\Office16\XLSTART\FemtetRef.xla'

        # certify
        if not os.path.exists(xla_file_path):
            raise FileNotFoundError(
                _(
                    en_message='Femtet XLA file ({xla_file_path}) not found. '
                               'Please run `Enable Macros` command.',
                    jp_message='Femtet XLA ファイル ({xla_file_path}) が見つかりません。'
                               '「マクロ機能の有効化」を実行してください。',
                    xla_file_path=xla_file_path
                )
            )

        # self.excel.Workbooks.Add(xla_file_path)
        self.excel.Workbooks.Open(xla_file_path, ReadOnly=True)

    @staticmethod
    def add_femtet_macro_reference(wb):

        # search
        ref_file_2 = os.path.abspath(util._get_femtetmacro_dllpath())
        contain_2 = False
        for ref in wb.VBProject.References:
            if ref.Description is not None:
                if ref.Description == 'FemtetMacro':  # FemtetMacro
                    contain_2 = True
                    break
        # add
        if not contain_2:
            wb.VBProject.References.AddFromFile(ref_file_2)

    @staticmethod
    def remove_femtet_ref_xla(wb):
        # search
        for ref in wb.VBProject.References:
            if ref.Description is not None:
                if ref.Description == 'FemtetMacro':  # FemtetMacro
                    wb.VBProject.References.Remove(ref)

    # ===== load =====
    def load_variables(self, opt: AbstractOptimizer, raise_if_no_keyword=True) -> None:

        df = ParseAsParameter.parse(
            self.input_xlsm_path,
            self.input_sheet_name,
            raise_if_no_keyword,
        )

        for i, row in df.iterrows():

            # use(optional)
            use = True
            if ParseAsParameter.use in df.columns:
                _use = row[ParseAsParameter.use]
                use = False if _is_cell_value_empty(_use) else bool(_use)  # bool or NaN

            # name
            name = str(row[ParseAsParameter.name])

            # if the variable is already added by
            # add_parameter or add_expression,
            # use it.
            if not self.force_override_when_load:
                if name in opt.variable_manager.get_variables():
                    continue

            # value
            value = float_(row[ParseAsParameter.value])

            # if 'choices' column exists and is not empty, use as Categorical.
            kind = 'Numerical'
            if ParseAsParameter.choices in df.columns:
                if row[ParseAsParameter.choices] != _EMPTY_CHOICE:
                    kind = 'Categorical'

            if kind == 'Numerical':

                # lb (optional)
                lb = None
                if ParseAsParameter.lb in df.columns:
                    lb = row[ParseAsParameter.lb]
                    lb = None if _is_cell_value_empty(lb) else float(lb)

                # ub (optional)
                ub = None
                if ParseAsParameter.ub in df.columns:
                    ub = row[ParseAsParameter.ub]
                    ub = None if _is_cell_value_empty(ub) else float(ub)

                # step (optional)
                step = None
                if ParseAsParameter.step in df.columns:
                    step = row[ParseAsParameter.step]
                    step = None if _is_cell_value_empty(step) else float(step)

                opt.add_parameter(
                    name=name,
                    initial_value=value,
                    lower_bound=lb,
                    upper_bound=ub,
                    step=step,
                    fix=not use,
                )

            elif kind == 'Categorical':

                # choices
                choices = row[ParseAsParameter.choices]
                assert choices != _EMPTY_CHOICE
                opt.add_categorical_parameter(
                    name=name,
                    initial_value=value,
                    choices=choices,
                    fix=not use,
                )

            else:
                raise NotImplementedError

    def load_objectives(self, opt: AbstractOptimizer, raise_if_no_keyword=True):

        df = ParseAsObjective.parse(
            self.output_xlsm_path,
            self.output_sheet_name,
            raise_if_no_keyword,
        )

        for i, row in df.iterrows():

            # use(optional)
            use = True
            if ParseAsObjective.use in df.columns:
                _use = row[ParseAsObjective.use]
                use = False if _is_cell_value_empty(_use) else bool(_use)  # bool or NaN

            # name
            name = str(row[ParseAsObjective.name])

            # if the objective is already added by
            # add_objective, use it.
            if not self.force_override_when_load:
                if name in opt.objectives.keys():
                    continue

            # direction
            direction = row[ParseAsObjective.direction]
            assert not _is_cell_value_empty(direction), 'direction is empty.'
            try:
                direction = float(direction)
            except ValueError:
                direction = str(direction).lower()
                assert direction in ['minimize', 'maximize']

            if use:
                # objective を作る
                opt.add_objective(
                    name=name,
                    direction=direction,
                    fun=_ScapeGoatObjective(),
                    kwargs=dict(name=name),
                )

    def load_constraints(self, opt: AbstractOptimizer, raise_if_no_keyword=False):

        df = ParseAsConstraint.parse(
            self.constraint_xlsm_path,
            self.constraint_sheet_name,
            raise_if_no_keyword,
        )

        for i, row in df.iterrows():

            # use(optional)
            use = True
            if ParseAsConstraint.use in df.columns:
                _use = row[ParseAsConstraint.use]
                use = False if _is_cell_value_empty(_use) else bool(_use)  # bool or NaN

            # name
            name = str(row[ParseAsConstraint.name])

            # if the constraint is already added by
            # add_constraint, use it.
            if not self.force_override_when_load:
                if name in opt.constraints.keys():
                    continue

            # lb (optional)
            lb = None
            if ParseAsConstraint.lb in df.columns:
                lb = row[ParseAsConstraint.lb]
                lb = None if _is_cell_value_empty(lb) else float(lb)

            # ub (optional)
            ub = None
            if ParseAsConstraint.ub in df.columns:
                ub = row[ParseAsConstraint.ub]
                ub = None if _is_cell_value_empty(ub) else float(ub)

            # strict (optional)
            strict = True
            if ParseAsConstraint.strict in df.columns:
                _strict = row[ParseAsConstraint.strict]
                strict = True if _is_cell_value_empty(_strict) else bool(_strict)  # bool or NaN

            # using_fem (optional)
            calc_before_solve = True
            if ParseAsConstraint.calc_before_solve in df.columns:
                _calc_before_solve = row[ParseAsConstraint.calc_before_solve]
                calc_before_solve = True if _is_cell_value_empty(_calc_before_solve) else bool(
                    _calc_before_solve)  # bool or NaN

            if use:
                opt.add_constraint(
                    name=name,
                    lower_bound=lb,
                    upper_bound=ub,
                    strict=strict,
                    fun=_ScapeGoatObjective(),
                    kwargs=dict(name=name),
                    using_fem=not calc_before_solve,
                )

    def objective_from_excel(self, _, name: str):
        r = 1 + search_r(self.output_xlsm_path, self.output_sheet_name, name)
        c = 1 + search_c(self.output_xlsm_path, self.output_sheet_name, ParseAsObjective.value)
        v = self.sh_output.Cells(r, c).value
        return float(v)

    def constraint_from_excel(self, _, name: str):
        r = 1 + search_r(self.constraint_xlsm_path, self.constraint_sheet_name, name)
        c = 1 + search_c(self.constraint_xlsm_path, self.constraint_sheet_name, ParseAsConstraint.value)
        v = self.sh_constraint.Cells(r, c).value
        return float(v)

    # ===== Workbook and WorkSheet =====
    # unpicklable object を excel だけにするため
    # これらのオブジェクトは動的に取得
    def _get_wb(self, path) -> CDispatch:
        for wb in self.excel.Workbooks:
            if wb.Name == os.path.basename(path):
                return wb
        else:
            raise WorkbookNotOpenedError(f'{path} is not opened.')

    @staticmethod
    def _get_sh(wb: CDispatch, name) -> CDispatch:
        for sh in wb.WorkSheets:
            if sh.Name == name:
                return sh
        else:
            raise WorkSheetNotFoundError(
                _(
                    en_message='{xla_file_path} not found. Please check the '
                               '"Enable Macros" command was executed.',
                    jp_message='{xla_file_path}が見つかりません。'
                               '"マクロを有効にする" command が実行'
                               'されたか確認してください。',
                    xla_file_path=wb.Name
                )
            )

    @property
    def wb_input(self) -> CDispatch:
        return self._get_wb(self.input_xlsm_path)

    @property
    def sh_input(self) -> CDispatch:
        return self._get_sh(self.wb_input, self.input_sheet_name)

    @property
    def wb_output(self) -> CDispatch:
        return self._get_wb(self.output_xlsm_path)

    @property
    def sh_output(self) -> CDispatch:
        return self._get_sh(self.wb_output, self.output_sheet_name)

    @property
    def wb_constraint(self) -> CDispatch:
        return self._get_wb(self.constraint_xlsm_path)

    @property
    def sh_constraint(self) -> CDispatch:
        return self._get_sh(self.wb_constraint, self.constraint_sheet_name)

    @property
    def wb_procedure(self) -> CDispatch:
        return self._get_wb(self.procedure_xlsm_path)

    @property
    def wb_setup(self) -> CDispatch:
        return self._get_wb(self.setup_xlsm_path)

    @property
    def wb_teardown(self) -> CDispatch:
        return self._get_wb(self.teardown_xlsm_path)

    # ===== update =====
    def update_parameter(self, x: TrialInput) -> None:

        COMInterface.update_parameter(self, x)

        # excel シートの変数更新
        if self.use_named_range:
            for key, variable in self.current_prm_values.items():
                try:
                    self.sh_input.Range(key).value = variable.value
                except com_error:
                    logger.warning(
                        _(
                            en_message='The cell address specification by named range is failed. '
                                       'The process changes the specification method to table based.',
                            jp_message='名前範囲によるセルアドレス指定に失敗しました。'
                                       '処理は指定方法をテーブルに変更します。',
                        )
                    )
                    self.use_named_range = False
                    break

        if not self.use_named_range:  # else にしないこと
            for name, variable in self.current_prm_values.items():
                r = 1 + search_r(self.input_xlsm_path, self.input_sheet_name, name)
                c = 1 + search_c(self.input_xlsm_path, self.input_sheet_name, ParseAsParameter.value)
                self.sh_input.Cells(r, c).value = variable.value

        # 再計算
        self.excel.CalculateFull()

    def update(self) -> None:

        if self.procedure_name is None:
            return

        # マクロ実行
        try:
            self.wb_procedure.Activate()
            sleep(0.1)
            with watch_excel_macro_error(self.excel, timeout=self.procedure_timeout):
                self.excel.Run(
                    f'{self.procedure_name}',
                    *self.procedure_args
                )

            # 再計算
            self.excel.CalculateFull()

        except com_error as e:
            raise SolveError(
                # 変換後(あなたの出力)
                _(
                    en_message='Failed to run macro {procedure_name}. '
                               'The original message is: {exception}',
                    jp_message='マクロ {procedure_name} の実行に失敗しました。 '
                               'エラーメッセージ: {exception}',
                    procedure_name=self.procedure_name,
                    exception=e
                )
            )

    # ===== close =====
    def _close_workbooks(self):
        # workbook を閉じる
        with watch_excel_macro_error(self.excel, timeout=10, restore_book=False):
            self.wb_input.Close(_SaveChanges := False)

        def close_wb_if_needed(attrib_name_):
            try:
                # self.wb_output などとすると
                # 関数呼び出しの時点でエラーになる
                wb = getattr(self, attrib_name_)
            except WorkbookNotOpenedError:
                pass
            else:
                with watch_excel_macro_error(self.excel, timeout=10, restore_book=False):
                    wb.Close(False)

        close_wb_if_needed('wb_output')
        close_wb_if_needed('wb_constraint')
        close_wb_if_needed('wb_procedure')
        close_wb_if_needed('wb_setup')
        close_wb_if_needed('wb_teardown')

    def close(self):

        # 無駄に不具合に遭う可能性があるので
        # 参照設定は解除しない

        if not hasattr(self, 'excel'):
            return

        if self.excel is None:
            return

        # 終了処理を必要なら実施する
        if self.teardown_procedure_name is not None:

            logger.info(_(
                en_message='{procedure_kind} procedure {procedure_name} is running...',
                jp_message='{procedure_kind} プロシージャ {procedure_name} が実行中...',
                procedure_kind='Teardown',
                procedure_name=self.teardown_procedure_name
            ))

            with Lock('excel_teardown_procedure'):
                try:
                    self.wb_teardown.Activate()
                    sleep(0.1)
                    with watch_excel_macro_error(self.excel, timeout=self.procedure_timeout, restore_book=False):
                        self.excel.Run(
                            f'{self.teardown_procedure_name}',
                            *self.teardown_procedure_args
                        )

                    # 再計算
                    self.excel.CalculateFull()

                except com_error as e:
                    raise RuntimeError(
                        _(
                            en_message='Failed to run macro {procedure_name}. '
                                       'The original message is: {exception}',
                            jp_message='マクロ {procedure_name} の実行に失敗しました。 '
                                       'エラーメッセージ： {exception}',
                            procedure_name=self.teardown_procedure_name,
                            exception=e
                        )
                    )

        # excel プロセスを終了する
        if self.terminate_excel_when_quit:

            logger.info(_(
                en_message='Terminating Excel process...',
                jp_message='Excel プロセスを終了しています...',
            ))

            already_terminated = not hasattr(self, 'excel')
            if already_terminated:
                return

            # ワークブックを閉じる
            self._close_workbooks()

            # excel の終了
            with watch_excel_macro_error(self.excel, timeout=10, restore_book=False):
                self.excel.Quit()
            del self.excel

            # ここで Excel のプロセスが残らず落ちる
            gc.collect()
            logger.info(_(
                en_message='Excel process is terminated.',
                jp_message='Excel プロセスは終了しました。',
            ))

            if self._with_femtet_autosave_setting:
                logger.info(_(
                    en_message='Restore Femtet setting of autosave.',
                    jp_message='Femtet の自動保存設定を復元しています。',
                ))
                _set_autosave_enabled(self._femtet_autosave_buffer)

        # そうでない場合でもブックは閉じる
        else:
            self._close_workbooks()


# main thread で作成した excel への参照を含む関数を
# 直接 thread や process に渡すと機能しない
class _ScapeGoatObjective:

    # for type hint
    def __call__(self) -> float:
        pass

    @property
    def __globals__(self):
        return dict()


def _is_same_path(p1, p2):
    _p1 = os.path.abspath(p1).lower()
    _p2 = os.path.abspath(p2).lower()
    return _p1 == _p2


def _is_cell_value_empty(cell_value):
    if isinstance(cell_value, str):
        return cell_value == ''
    elif isinstance(cell_value, int) \
            or isinstance(cell_value, float):
        return np.isnan(cell_value)
    elif cell_value is None:
        return True
    else:
        return False
