import psutil
from pyfemtet.logger import get_logger

logger = get_logger('ui')

DEFAULT_PORT = 49000


def get_unused_port_number(start: int = DEFAULT_PORT) -> int:
    # "LISTEN" 状態のポート番号をリスト化
    used_ports = [conn.laddr.port for conn in psutil.net_connections() if conn.status == 'LISTEN']
    # start から順に調べ、空いているポート番号かどうかチェックする
    port = start
    for port in range(start, 65535 + 1):
        if port not in set(used_ports):
            break
    # ポート番号が指定と違えば警告する
    if port != start:
        logger.warning(f'Specified port "{start}" seems to be used. Port "{port}" is used instead.')

    return port
