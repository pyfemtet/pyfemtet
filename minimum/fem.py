from femtetutils import util
from dask.distributed import get_worker
import os
from pythoncom import CoInitialize, CoUninitialize
from minimum.dispatch_extension import launch_and_dispatch_femtet



class FEM:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def update(self, x):
        pass

    def setup_before_parallel(self, client):
        pass


class Femtet(FEM):

    def __init__(self, path):

        # win32com の初期化
        CoInitialize()

        # femtet 起動, 接続
        self.Femtet, self.pid = launch_and_dispatch_femtet()

        # ファイルを開く
        try:
            worker = get_worker()
            space = worker.local_directory
            self.Femtet.LoadProject(os.path.join(space, os.path.basename(path)), True)
        except ValueError:
            self.Femtet.LoadProject(path, True)

        # restore 用データの保存
        super().__init__(
            path=path,
        )

    def update(self, parameters):
        self.Femtet.Gaudi.Activate()
        for i, row in parameters.iterrows():
            self.Femtet.UpdateVariable(row['name'], row['value'])
        self.Femtet.Gaudi.ReExecute()
        self.Femtet.Gaudi.Redraw()
        self.Femtet.Gaudi.Mesh()
        self.Femtet.Solve()
        self.Femtet.OpenCurrentResult()

    def setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['path'],
            False
        )


    def __del__(self):
        util.close_femtet(self.Femtet.hWnd, 1, True)
        # CoUninitialize()  # Win32 exception occurred releasing IUnknown at 0x0000022427692748
