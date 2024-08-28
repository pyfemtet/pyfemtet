import os
import csv
from shutil import copy
from time import sleep
from subprocess import run
from multiprocessing import Process
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from win32com.client import Dispatch
from femtetutils import util

from pyfemtet.opt import FEMOpt


class SuperSphere:
    def __init__(self, n):
        self.n = n

    def x(self, radius, *angles):
        assert len(angles) == self.n - 1, 'invalid angles length'

        out = []

        for i in range(self.n):
            if i == 0:
                out.append(radius * np.cos(angles[0]))
            elif i < self.n - 1:
                product = radius
                for j in range(i):
                    product *= np.sin(angles[j])
                product *= np.cos(angles[i])
                out.append(product)
            else:
                product = radius
                for j in range(i):
                    product *= np.sin(angles[j])
                out.append(product)
        return out


def _open_femprj(femprj_path):
    Femtet = Dispatch('FemtetMacro.Femtet')
    for _ in tqdm(range(5), 'wait for dispatch Femtet'):
        sleep(1)
    Femtet.LoadProject(femprj_path, True)


def launch_femtet(femprj_path):
    # launch Femtet externally
    print('Launching Femtet...')
    util.execute_femtet()
    pid = util.get_last_executed_femtet_process_id()
    for _ in tqdm(range(8), 'Wait for launch Femtet.'):
        sleep(1)

    # open femprj in a different process
    # to release Femtet for sample program
    print('Opening femprj...')
    if femprj_path:
        p = Process(
            target=_open_femprj,
            args=(femprj_path,),
        )
        p.start()
        p.join()


def taskkill_femtet():
    for _ in tqdm(range(3), 'wait before taskkill Femtet'):
        sleep(1)
    run(['taskkill', '/f', '/im', 'Femtet.exe'])
    for _ in tqdm(range(3), 'wait after taskkill Femtet'):
        sleep(1)


def find_latest_csv(dir_path=None):
    if dir_path is None: dir_path = '.'
    target = os.path.join(dir_path, '*.csv')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    out = sorted(files, key=lambda files: files[1])[-1]
    return os.path.abspath(out[0])


def py_to_reccsv(py_path, suffix=''):
    dst_csv_path = py_path + suffix
    dst_csv_path = dst_csv_path.replace(f'.py{suffix}', f'{suffix}.reccsv')
    return dst_csv_path


def record_result(src: FEMOpt or str, py_path, suffix=''):
    """Record the result csv for `is_equal_result`."""

    if isinstance(src, FEMOpt):  # get df directory
        src_csv_path = src.history_path
    else:
        src_csv_path = os.path.abspath(src)

    dst_csv_path = py_to_reccsv(py_path, suffix)
    copy(src_csv_path, dst_csv_path)


def _get_obj_from_csv(csv_path, encoding='cp932'):
    df = pd.read_csv(csv_path, encoding=encoding, header=2)
    columns = df.columns
    with open(csv_path, mode='r', encoding=encoding, newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        meta = reader.__next__()
    obj_indices = np.where(np.array(meta) == 'obj')[0]
    out = df.iloc[:, obj_indices]
    return out, columns


def is_equal_result(ref_path, dif_path, log_path):
    """Check the equality of two result csv files."""
    ref_df, ref_columns = _get_obj_from_csv(ref_path)
    dif_df, dif_columns = _get_obj_from_csv(dif_path)

    with open(log_path, 'a', newline='\n') as f:
        f.write('\n\n===== 結果の分析 =====\n\n')
        f.write(f'        \tref\tdif\n')
        f.write(f'---------------------\n')
        f.write(f'len(col)\t{len(ref_columns)}\t{len(dif_columns)}\n')
        f.write(f'len(df) \t{len(ref_df)}\t{len(dif_df)}\n')
        try:
            difference = (np.abs(ref_df.values - dif_df.values) / np.abs(dif_df.values)).mean()
            f.write(f'diff    \t{int(difference*100)}%\n')
        except Exception:
            f.write(f'diff    \tcannot calc\n')

    assert len(ref_columns) == len(dif_columns), '結果 csv の column 数が異なります。'
    assert len(ref_df) == len(dif_df), '結果 csv の row 数が異なります。'
    assert difference <= 0.05, '前回の結果との平均差異が 5% を超えています。'
