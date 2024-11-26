import os
from pathlib import Path

import pandas as pd
import numpy as np
from dask.distributed import get_worker

from win32com.client import DispatchEx, Dispatch
from win32com.client.dynamic import CDispatch
from femtetutils import util

# noinspection PyUnresolvedReferences
from pythoncom import CoInitialize, CoUninitialize
# noinspection PyUnresolvedReferences
from pywintypes import com_error

from pyfemtet.opt import FEMInterface
from pyfemtet.core import SolveError
from pyfemtet.opt.optimizer.parameter import Parameter

from pyfemtet.dispatch_extensions import _get_pid, dispatch_specific_femtet
from pyfemtet.dispatch_extensions._impl import _NestableSpawnProcess


class ExcelInterface(FEMInterface):

    input_xlsm_path: Path  # 操作対象の xlsm パス
    input_sheet_name: str  # 変数セルを定義しているシート名
    output_xlsm_path: Path  # 操作対象の xlsm パス (指定しない場合、input と同一)
    output_sheet_name: str  # 計算結果セルを定義しているシート名 (指定しない場合、input と同一)

    # FIXME: related_file_paths: dict[Path]  # 並列時に個別に並列プロセスの space にアップロードする必要のあるパス

    procedure_name: str  # マクロ関数名（or モジュール名.関数名）
    procedure_args: list  # マクロ関数の引数

    excel: CDispatch  # Excel Application
    wb_input: CDispatch  # システムを構成する Workbook
    sh_input: CDispatch  # 変数の定義された WorkSheet
    wb_output: CDispatch  # システムを構成する Workbook
    sh_output: CDispatch  # 計算結果の定義された WorkSheet (sh_input と同じでもよい)

    visible: bool = True  # excel を可視化するかどうか
    display_alerts: bool = True  # ダイアログを表示するかどうか

    _load_problem_from_me: bool = True  # TODO: add_parameter() 等を省略するかどうか。定義するだけでフラグとして機能する。
    _excel_pid: int
    _excel_hwnd: int

    def __init__(
            self,
            input_xlsm_path: str or Path,
            input_sheet_name: str,
            output_xlsm_path: str or Path = None,
            output_sheet_name: str = None,
            procedure_name: str = None,
            procedure_args: list or tuple = None,
            connect_method: str = 'auto',  # or 'new'

    ):

        # 初期化
        self.input_xlsm_path = None  # あとで取得する
        self.input_sheet_name = input_sheet_name
        self.output_xlsm_path = None  # あとで取得する
        self.output_sheet_name = output_sheet_name or self.input_sheet_name
        self.procedure_name = procedure_name or 'FemtetMacro.FemtetMain'
        self.procedure_args = procedure_args or []
        assert connect_method in ['new', 'auto']
        self.connect_method = connect_method

        # dask サブプロセスのときは space 直下の input_xlsm_path を参照する
        try:
            worker = get_worker()
            space = worker.local_directory
            self.input_xlsm_path = Path(os.path.join(space, os.path.basename(input_xlsm_path))).resolve()
            self.output_xlsm_path = Path(os.path.join(space, os.path.basename(output_xlsm_path))).resolve()

        # main プロセスの場合は絶対パスを参照する
        except ValueError:
            self.input_xlsm_path = Path(os.path.abspath(input_xlsm_path)).resolve()
            if output_xlsm_path is None:
                self.output_xlsm_path = self.input_xlsm_path
            else:
                self.output_xlsm_path = Path(os.path.abspath(output_xlsm_path)).resolve()

        # 先に femtet を起動
        util.execute_femtet()

        # Femtet が Dispatch 可能になるまで捨てプロセスで待つ
        p = _NestableSpawnProcess(target=wait_femtet)
        p.start()
        p.join()

        # サブプロセスでの restore のための情報保管
        kwargs = dict(
            input_xlsm_path=self.input_xlsm_path,
            input_sheet_name=self.input_sheet_name,
            output_xlsm_path=self.output_xlsm_path,
            output_sheet_name=self.output_sheet_name,
            procedure_name=self.procedure_name,
            procedure_args=self.procedure_args,
            connect_method='new',  # subprocess で connect する際は new を強制する
        )
        super().__init__(**kwargs)

    def _setup_before_parallel(self, client) -> None:
        # メインプロセスで、並列プロセスを開始する前に行う前処理

        input_xlsm_path: Path = self.kwargs['input_xlsm_path']
        output_xlsm_path: Path = self.kwargs['output_xlsm_path']

        client.upload_file(str(input_xlsm_path), False)
        if input_xlsm_path.resolve() != output_xlsm_path.resolve():
            client.upload_file(str(output_xlsm_path), False)

    def connect_excel(self, connect_method):

        # ===== 新しい excel instance を起動 =====
        # 起動
        if connect_method == 'auto':
            self.excel = Dispatch('Excel.Application')
        else:
            self.excel = DispatchEx('Excel.Application')

        # 起動した excel の pid を記憶する
        self._excel_hwnd = self.excel.hWnd
        self._excel_pid = _get_pid(self.excel.hWnd)

        # 可視性の設定
        self.excel.Visible = self.visible
        self.excel.DisplayAlerts = self.display_alerts

        # 開く
        self.excel.Workbooks.Open(str(self.input_xlsm_path))
        for wb in self.excel.Workbooks:
            if wb.Name == os.path.basename(self.input_xlsm_path):
                self.wb_input = wb
                break
        else:
            raise RuntimeError(f'Cannot open {self.input_xlsm_path}')

        # シートを特定する
        for sh in self.wb_input.WorkSheets:
            if sh.Name == self.input_sheet_name:
                self.sh_input = sh
                break
        else:
            raise RuntimeError(f'Sheet {self.input_sheet_name} does not exist in the book {self.wb_input.Name}.')

        if self.input_xlsm_path.resolve() == self.output_xlsm_path.resolve():
            self.wb_output = self.wb_input

        else:
            # 開く (output)
            self.excel.Workbooks.Open(str(self.output_xlsm_path))
            for wb in self.excel.Workbooks:
                if wb.Name == os.path.basename(self.output_xlsm_path):
                    self.wb_output = wb
                    break
            else:
                raise RuntimeError(f'Cannot open {self.output_xlsm_path}')

        # シートを特定する (output)
        for sh in self.wb_output.WorkSheets:
            if sh.Name == self.output_sheet_name:
                self.sh_output = sh
                break
        else:
            raise RuntimeError(f'Sheet {self.output_sheet_name} does not exist in the book {self.wb_output.Name}.')

        # book に参照設定を追加する
        self.add_femtet_ref_xla(self.wb_input)
        self.add_femtet_ref_xla(self.wb_output)

    def add_femtet_ref_xla(self, wb):

        # search
        ref_file_2 = os.path.abspath(util._get_femtetmacro_dllpath())
        contain = False
        for ref in wb.VBProject.References:
            if ref.Description is not None:
                if ref.Description == 'FemtetMacro':  # FemtetMacro
                    contain = True
        # add
        if not contain:
            wb.VBProject.References.AddFromFile(ref_file_2)

    def _setup_after_parallel(self, *args, **kwargs):
        # サブプロセス又はメインプロセスのサブスレッドで、最適化を開始する前の前処理

        CoInitialize()

        # excel に繋ぎなおす
        self.connect_excel(self.connect_method)

        # load_objective は 1 回目に呼ばれたのが main thread なので
        # subprocess に入った後でもう一度 load objective を行う
        from pyfemtet.opt.optimizer import AbstractOptimizer
        from pyfemtet.opt._femopt_core import Objective
        opt: AbstractOptimizer = kwargs['opt']
        obj: Objective
        for obj_name, obj in opt.objectives.items():
            if isinstance(obj.fun, ScapeGoatObjective):
                opt.objectives[obj_name].fun = self.objective_from_excel

    def update(self, parameters: pd.DataFrame) -> None:

        # params を作成
        params = dict()
        for _, row in parameters.iterrows():
            params[row['name']] = row['value']

        # excel シートの変数更新
        for key, value in params.items():
            self.sh_input.Range(key).value = value

        # 再計算
        self.excel.CalculateFull()  # 他に適したメソッドもあるかも。

        # マクロ実行
        try:
            self.excel.Run(
                f'{self.wb_input.Name}!{self.procedure_name}',
                *self.procedure_args
            )

        # FIXME: エラーハンドリング
        #   com_error をキャッチするか、
        #   sh_out に解析結果を書くようにして、
        #   それが FALSE なら SolveError を raise する。
        except ...:
            raise SolveError('Excelアップデートに失敗しました')

        # 再計算
        self.excel.CalculateFull()  # 他に適したメソッドもあるかも。

    def quit(self):
        self.wb_input.Close(SaveChanges := False)
        if self.input_xlsm_path.name != self.output_xlsm_path.name:
            self.wb_output.Close(SaveChanges := False)
        self.excel.Quit()
        del self.excel

        # import gc
        # gc.collect()

        # quit した後ならば femtet を終了できる
        # excel の process の完全消滅を待つ
        from time import sleep
        while self._excel_pid == _get_pid(self._excel_hwnd):
            print('エクセルの終了待ち...')
            sleep(1)

        # 正確だが時間がかかる
        femtet_pid = util.get_last_executed_femtet_process_id()
        Femtet, caught_pid = dispatch_specific_femtet(femtet_pid)
        assert femtet_pid == caught_pid
        Femtet.Exit(True)



    # 直接アクセスしてもよいが、ユーザーに易しい名前にするためだけのプロパティ
    @property
    def output_sheet(self) -> CDispatch:
        return self.sh_output

    @property
    def input_sheet(self) -> CDispatch:
        return self.sh_input

    @property
    def output_workbook(self) -> CDispatch:
        return self.wb_output

    @property
    def input_workbook(self) -> CDispatch:
        return self.wb_input

    def load_parameter(self, opt) -> None:
        from pyfemtet.opt.optimizer import AbstractOptimizer, logger
        opt: AbstractOptimizer

        df = pd.read_excel(
            self.input_xlsm_path,
            self.input_sheet_name,
            header=0,
            index_col=None,
        )

        for i, row in df.iterrows():
            try:
                name = row['name']
                value = row['current']
                lb = row['lower']
                ub = row['upper']
                step = row['step']
            except KeyError:
                logger.warn('列名が「name」「current」「lower」「upper」「step」になっていません。この順に並んでいると仮定して処理を続けます。')
                name, value, lb, ub, step, *_residuals = row.iloc[0]

            name = str(name)
            value = float(value)
            lb = float(lb) if not np.isnan(lb) else None
            ub = float(ub) if not np.isnan(ub) else None
            step = float(step) if not np.isnan(step) else None

            prm = Parameter(
                name=name,
                value=value,
                lower_bound=lb,
                upper_bound=ub,
                step=step,
                pass_to_fem=True,
                properties=None,
            )
            opt.variables.add_parameter(prm)

    def load_objective(self, opt):
        from pyfemtet.opt.optimizer import AbstractOptimizer, logger
        from pyfemtet.opt._femopt_core import Objective
        opt: AbstractOptimizer

        df = pd.read_excel(
            self.output_xlsm_path,
            self.output_sheet_name,
            header=0,
            index_col=None,
        )

        for i, row in df.iterrows():
            try:
                name = row['name']
                _ = row['current']
                direction = row['direction']
                value_column_index = list(df.columns).index('current')
            except KeyError:
                logger.warn('列名が「name」「current」「direction」になっていません。この順に並んでいると仮定して処理を続けます。')
                name, _, direction, *_residuals = row.iloc[0]
                value_column_index = 1

            name = str(name)

            # direction は minimize or maximize or float
            try:
                # float or not
                direction = float(direction)

            except ValueError:
                # 'minimize' or 'maximize
                direction = str(direction).lower()
                assert (direction == 'minimize') or (direction == 'maximize')

            # objective を作る
            opt.objectives[name] = Objective(
                fun=ScapeGoatObjective(),
                name=name,
                direction=direction,
                args=(i, value_column_index, ),  # 参照なのでこれで良い? parallel で問題になるかも
                kwargs=dict(),
            )

    def objective_from_excel(self, i: int, value_column_index: int):
        r = i + 2  # header が 1
        c = value_column_index + 1
        v = self.sh_output.Cells(r, c).value
        return float(v)


from time import sleep
def wait_femtet():
    Femtet = Dispatch('FemtetMacro.Femtet')
    while Femtet.hWnd <= 0:
        sleep(1)
        Femtet = Dispatch('FemtetMacro.Femtet')


# main thread で作成した excel への参照を含む関数を
# 直接 thread や process に渡すと機能しない
class ScapeGoatObjective:
    def __call__(self, *args, fem: ExcelInterface or None = None, **kwargs):
        fem.objective_from_excel(*args, **kwargs)

    @property
    def __globals__(self):
        return tuple()


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(__file__))

    dg_fem = ExcelInterface(
        input_xlsm_path=r"io_and_solve.xlsm",
        input_sheet_name='input',
        output_sheet_name='output',
        procedure_name='FemtetMacro.FemtetMain',
        connect_method='auto',
    )

    dg_df = pd.DataFrame(
        dict(
            name=['w', 'h', 'd'],
            value=[10, 10, 10],
        )
    )

    # fem.update(df)

    from threading import Thread

    # fem.excel = None
    # fem.sh_input = None
    # fem.sh_output = None
    # fem.wb_input = None
    # fem.wb_output = None

    # import gc
    # gc.collect()

    def update_with_restore_com(fem_, df_):
        fem_.__init__(**fem_.kwargs)
        fem_.update(df_)
        print(fem_.output_sheet.Cells)

    t = Thread(
        target=update_with_restore_com,
        args=(dg_fem, dg_df,)
    )

    t.start()
    t.join()

