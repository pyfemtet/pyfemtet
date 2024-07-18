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


def _get_obj_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding='cp932', header=2)
    columns = df.columns
    with open(csv_path, mode='r', encoding='cp932', newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        meta = reader.__next__()
    obj_indices = np.where(np.array(meta) == 'obj')[0]
    out = df.iloc[:, obj_indices]
    return out, columns


def is_equal_result(csv1, csv2, result_save_to=None):
    """Check the equality of two result csv files."""
    df1, columns1 = _get_obj_from_csv(csv1)
    df2, columns2 = _get_obj_from_csv(csv2)

    if result_save_to is not None:
        import datetime
        with open(result_save_to, 'a', encoding='utf-8', newline='\n') as f:
            name = os.path.basename(csv1)
            content = [
                f'===== result of {name} =====\n',
                f'{datetime.datetime.now()}\n',
                f'----- column numbers -----\n',
                f'csv1: {len(columns1)} columns\n',
                f'csv2: {len(columns2)} columns\n',
                f'----- row numbers -----\n',
                f'csv1: {len(df1)} columns\n',
                f'csv2: {len(df2)} columns\n',
                f'----- difference -----\n',
                f'max difference ratio: {(np.abs(df1.values - df2.values) / np.abs(df2.values)).max() * 100}%\n',
                '\n'
            ]
            f.writelines(content)

    assert len(columns1) == len(columns2), '結果 csv の column 数が異なります。'
    assert len(df1) == len(df2), '結果 csv の row 数が異なります。'
    assert (np.abs(df1.values - df2.values) / np.abs(df2.values)).max() <= 0.01, '前回の結果と 1% を超える相違があります。'
