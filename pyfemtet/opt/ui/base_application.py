import threading
from time import sleep

import pandas as pd

from dash_application import MultiPageApplication

from pyfemtet.opt._femopt_core import History



class BasePyFemtetApplication(MultiPageApplication):

    def __init__(self, history: History):
        super().__init__()
        self.history = history
        self._df = history.get_df()

    def get_df(self) -> pd.DataFrame:
        """Get df from history.

        Please note that history.get_df() accesses Actor,
        but the dash callback cannot access to Actor,
        so the _df should be updated by self._sync().

        """
        return self._df

    def run(self, host=None, debug=False, launch_browser=False):
        # _sync も app と同様 **デーモンスレッドで** 並列実行
        sync_thread = threading.Thread(
            target=self._sync,
            args=(),
            kwargs={},
            daemon=True,
        )
        sync_thread.start()
        super().run(host, debug, launch_browser)

    def _sync(self):
        while True:
            # df はここでのみ上書きされ、dashboard から書き戻されることはない
            self._df = self.history.get_df()

            # status は...
            print('status の共有方法をまだ実装していません。')

            # lock が release されていれば、これ以上 sync が実行される必要はない
            if not self.lock_to_terminate.locked():
                break
            sleep(1)
