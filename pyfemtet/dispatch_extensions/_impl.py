from time import time, sleep
from multiprocessing import current_process

from tqdm import tqdm

from win32com.client import Dispatch, CDispatch

from femtetutils import util

from pyfemtet._i18n import Msg
from pyfemtet._util.dask_util import *
from pyfemtet._util.process_util import *
from pyfemtet.logger import get_module_logger


__all__ = [
    'launch_and_dispatch_femtet',
    'dispatch_femtet',
    'dispatch_specific_femtet',
    'DispatchExtensionException',
]


logger = get_module_logger('dispatch', False)


DISPATCH_TIMEOUT = 120


class DispatchExtensionException(Exception):
    pass


class FemtetNotFoundException(DispatchExtensionException):
    """Raises when (specific) Femtet process doesn't exist."""


class FemtetConnectionTimeoutError(DispatchExtensionException):
    """Raises when connection trials is timed out."""


def _get_subprocess_log_prefix():
    return f'({current_process().name}) '


def launch_and_dispatch_femtet(
        timeout=DISPATCH_TIMEOUT,
        strictly_pid_specify=True
) -> tuple[CDispatch, int]:
    """Launch Femtet by new process and connect to it.

    The wrapper for Dispatch() but returns PID with IFemtet.

    Args:
        timeout (int, optional): Raises an error if the connection is not established within the specified timeout.
        strictly_pid_specify (bool, optional): Attempts to establish a connection to the launched Femtet strictly.
            This may result in slower processing due to process exclusivity handling.

    Raises:
        FemtetConnectionTimeoutError: Couldn't connect Femtet process for some reason (i.e. Femtet.exe is not launched).

    Returns:
        tuple[IFemtet, int]: An object for controlling Femtet and the PID of the Femtet being controlled.

    """

    # launch femtet
    util.execute_femtet()
    pid = util.get_last_executed_femtet_process_id()
    logger.debug(f'Target pid is {pid}.')
    for _ in tqdm(range(5), Msg.WAIT_FOR_LAUNCH_FEMTET):
        sleep(1)

    # dispatch femtet
    if strictly_pid_specify:
        Femtet, pid = dispatch_specific_femtet(pid, timeout)
    else:
        # worker process なら排他処理する
        with Lock('simply-dispatch-femtet'):
            Femtet, pid = dispatch_femtet()
    return Femtet, pid


def dispatch_femtet(timeout=DISPATCH_TIMEOUT, subprocess_log_prefix='') -> tuple[CDispatch, int]:
    """Connect to existing Femtet process.

    The wrapper for Dispatch() but returns PID with IFemtet.

    Args:
        timeout (int, optional): Raises an error if the connection is not established within the specified timeout.
        subprocess_log_prefix (str, optional): A prefix to output in logs. Typically used only for internal processing.

    Raises:
        FemtetConnectionTimeoutError: Couldn't connect Femtet process for some reason (i.e. Femtet.exe is not launched).

    Returns:
        tuple[IFemtet, int]: An object for controlling Femtet and the PID of the Femtet being controlled.

    """

    # Dispatch
    if subprocess_log_prefix:
        logger.debug('%s'+'Try to connect Femtet.', subprocess_log_prefix)
    else:
        logger.info(Msg.TRY_TO_CONNECT_FEMTET)
    Femtet = Dispatch('FemtetMacro.Femtet')
    logger.debug('%s'+'Dispatch finished.', subprocess_log_prefix)

    # timeout 秒以内に接続が確立するか
    start = time()
    while True:
        hwnd = Femtet.hWnd

        # 接続が確立
        if hwnd > 0:
            logger.debug('%s'+f'Dispatched hwnd is {hwnd} and'
                              f'its pid is {_get_pid(hwnd)}.'
                              f'Connection established.', subprocess_log_prefix)
            break
        else:
            logger.debug('%s'+f'Dispatched hwnd is {hwnd}.'
                              f'Waiting for establishing connection.', subprocess_log_prefix)

        # 接続がタイムアウト
        if time()-start > timeout:
            raise FemtetConnectionTimeoutError(f'Connection trial with Femtet is timed out in {timeout} sec')

        sleep(1)

    # pid を取得
    pid = _get_pid(hwnd)
    if subprocess_log_prefix:
        logger.debug('%s'+f'Successfully connected. The pid of Femtet is {pid}.', subprocess_log_prefix)
    else:
        logger.info(Msg.F_FEMTET_CONNECTED(pid))

    return Femtet, pid


def _block_other_femtets(
        target_pid,
        subprocess_idx,
        connection_flags,
        timeout_main_wait_for_us,
        lock_inter_subproc,
        # lock_main,
):
    """Target 以外の pid を持つ Femtet と接続し、他の（メインの）プロセスが接続できないようにする。

    ただし、Femtet の Dispatch ルールは一意なので、
    ターゲットの Femtet と接続するまでの他のプロセスだけをブロックする。
    接続可能な Femtet がなければ timeout するまで接続を試行しようとするが、
    ひとつでも timeout すれば、それ以降のプロセスは dispatch する必要がない。

    """

    # Dispatch を行う
    if connection_flags[subprocess_idx]:
        # 他のタイムアウトしたプロセスによってスキップフラグを立てられた場合
        my_pid = 0
    else:
        lock_inter_subproc.acquire()
        try:
            logger.debug('%s'+f'Subprocess started.', _get_subprocess_log_prefix())
            _, my_pid = dispatch_femtet(2, _get_subprocess_log_prefix())
        except DispatchExtensionException:
            logger.debug('%s'+f'Connection failed. The other subprocesses will skip.', _get_subprocess_log_prefix())
            my_pid = 0
            # ひとつでもここに来れば、それ以降は探しても無駄なのでスキップフラグを立てる
            for i in range(len(connection_flags)-1):
                connection_flags[i] = True
        finally:
            lock_inter_subproc.release()

    # # target なら Dispatch の終了を通知する前に main を lock する
    # if my_pid == target_pid:
    #     lock_main.acquire()

    # Dispatch の終了を通知
    connection_flags[subprocess_idx] = True

    # # pid が目的のものなら他の subprocesses すべてが Dispatch 終了後に開放する
    # pid が目的のものなら即刻開放する
    if my_pid == target_pid:
        # while True:
        #     start = time()
        #     if all(connection_flags[:-1]):
        #         break
        #     if time()-start > timeout_main_wait_for_us:
        #         return -1
        #     sleep(1)
        logger.debug('%s'+f'Release {my_pid}.', _get_subprocess_log_prefix())
        # lock_main.release()
        return 0

    # my_pid が非正なら何にも関与しないので即刻終了する
    elif my_pid <= 0:
        logger.debug('%s'+'Failed or timed out to connect Femtet.', _get_subprocess_log_prefix())
        return 1

    # そうでなければメインプロセスが Dispatch を終えるまで解放しない
    else:
        logger.debug('%s'+f'Connected to {my_pid}. Keep connection to block.', _get_subprocess_log_prefix())
        start = time()
        while True:
            if connection_flags[-1]:
                logger.debug('%s'+'Main process seems to connect target Femtet.', _get_subprocess_log_prefix())
                return 1
            if time()-start > timeout_main_wait_for_us:
                logger.warning(
                    '%s'+f'Timed out to wait for the main process to {timeout_main_wait_for_us}',
                    _get_subprocess_log_prefix()
                )
                return -1
            sleep(1)


def dispatch_specific_femtet(pid, timeout=DISPATCH_TIMEOUT) -> tuple[CDispatch, int]:

    with Lock('dispatch-specific-femtet'):
        return _dispatch_specific_femtet_core(pid, timeout)


def _dispatch_specific_femtet_core(pid, timeout=DISPATCH_TIMEOUT) -> tuple[CDispatch, int]:

    if timeout < 5:
        raise ValueError(f'Timeout to dispatch specific femtet should equal or be over 5.')

    # 存在する Femtet プロセスの列挙
    pids = _get_pids('Femtet.exe')
    logger.info(f'PID of existing Femtet processes: {pids}')
    if not (pid in pids):
        raise FemtetNotFoundException(f"Femtet (pid = {pid}) doesn't exist.")

    logger.info(Msg.F_SEARCHING_FEMTET_WITH_SPECIFIC_PID(pid))

    # 子プロセスの準備
    with _NestableSyncManager() as manager:
        # フラグの準備
        connection_flags = manager.list()
        lock_inter_subproc = manager.Lock()
        # lock_main = manager.Lock()
        for _ in range(len(pids)+1):  # [-1]は自プロセス用のフラグ
            connection_flags.append(False)

        # 目的以外の Femtet をブロックする子プロセスを開始
        logger.debug('Start subprocess to block Femtet other than target pid.')
        processes = []
        for subprocess_id in tqdm(range(len(pids)), 'Specifying connection...'):
            p = _NestableSpawnProcess(
                target=_block_other_femtets,
                args=(
                    pid,
                    subprocess_id,
                    connection_flags,
                    10,
                    lock_inter_subproc,
                    # lock_main,
                ),
            )
            p.start()
            processes.append(p)

        # 子プロセスの Dispatch 完了を待つ
        start = time()
        while True:
            if all(connection_flags[:-1]):
                break
            if time()-start > timeout:
                # 子プロセスを終了する
                for p in processes:
                    p.terminate()
                    p.join()
                try:
                    lock_inter_subproc.release()
                except RuntimeError:
                    pass
                raise FemtetConnectionTimeoutError(
                    Msg.F_ERR_FEMTET_CONNECTION_TIMEOUT(
                        pid=pid,
                        timeout=timeout,
                    )
                )
            sleep(1)

        # # subprocesses の Dispatch 終了後、target_pid の解放を待つ
        # lock_main.acquire()

        # 子プロセスによるブロックが終了しているので Dispatch する
        try:
            logger.debug('Block process seems to be finished. Try Dispatch.')
            Femtet, my_pid = dispatch_femtet()
        except DispatchExtensionException as e:
            # lock_main.release()
            raise e

        # Dispatch 完了を通知
        connection_flags[-1] = True

        # サブプロセスすべての正常終了を待つ
        for p in processes:
            p.join()

    if my_pid != pid:  # pid の結果が違う場合
        txt = f'Target pid is {pid}, but connected pid is {my_pid}. '
        txt += f'The common reason is that THIS python process once connected {pid}.'
        logger.warning(txt)

    return Femtet, my_pid


def _debug():
    launch_and_dispatch_femtet(5)


if __name__ == '__main__':
    _Femtet, _my_pid = launch_and_dispatch_femtet(5)
    # _Femtet, _my_pid = dispatch_specific_femtet(pid=26124)
