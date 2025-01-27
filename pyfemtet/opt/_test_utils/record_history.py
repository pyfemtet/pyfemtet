import os
import csv
from shutil import copy
from glob import glob

import numpy as np
import pandas as pd

from pyfemtet.opt import FEMOpt
from pyfemtet._message import encoding as ENCODING


def remove_extra_data_from_csv(csv_path, encoding=ENCODING):

    with open(csv_path, mode="r", encoding=encoding, newline="\n") as f:
        reader = csv.reader(f, delimiter=",")
        data = [line for line in reader]

    new_meta_data = data[0]
    new_meta_data[0] = ""
    data[0] = new_meta_data

    with open(csv_path, mode="w", encoding=encoding, newline="\n") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


def find_latest_csv(dir_path=None):
    if dir_path is None:
        dir_path = ""
    target = os.path.join(dir_path, "*.csv")
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    out = sorted(files, key=lambda files: files[1])[-1]
    return os.path.abspath(out[0])


def py_to_reccsv(py_path, suffix=""):
    dst_csv_path = py_path + suffix
    dst_csv_path = dst_csv_path.replace(f".py{suffix}", f"{suffix}.reccsv")
    return dst_csv_path


def record_result(src: FEMOpt or str, py_path, suffix=""):
    """Record the result csv for `is_equal_result`."""

    if isinstance(src, FEMOpt):  # get df directory
        src_csv_path = src.history_path
    else:
        src_csv_path = os.path.abspath(src)

    dst_csv_path = py_to_reccsv(py_path, suffix)
    copy(src_csv_path, dst_csv_path)


def _get_obj_from_csv(csv_path, encoding=ENCODING):
    df = pd.read_csv(csv_path, encoding=encoding, header=2)
    columns = df.columns
    with open(csv_path, mode="r", encoding=encoding, newline="\n") as f:
        reader = csv.reader(f, delimiter=",")
        meta = reader.__next__()
    obj_indices = np.where(np.array(meta) == "obj")[0]
    out: pd.DataFrame = df.iloc[:, obj_indices]
    out = out.dropna(axis=0)
    return out, columns


def is_equal_result(ref_path, dif_path, log_path=None, threashold=0.05):
    """Check the equality of two result csv files."""
    ref_df, ref_columns = _get_obj_from_csv(ref_path)
    dif_df, dif_columns = _get_obj_from_csv(dif_path)

    if log_path is not None:
        with open(log_path, "a", newline="\n", encoding=ENCODING) as f:
            f.write("\n\n===== 結果の分析 =====\n\n")
            f.write(f"        \tref\tdif\n")
            f.write(f"---------------------\n")
            f.write(f"len(col)\t{len(ref_columns)}\t{len(dif_columns)}\n")
            f.write(f"len(df) \t{len(ref_df)}\t{len(dif_df)}\n")
            try:
                difference = (
                    np.abs(ref_df.values - dif_df.values) / np.abs(dif_df.values)
                ).mean()
                f.write(f"diff    \t{int(difference*100)}%\n")
            except Exception:
                f.write(f"diff    \tcannot calc\n")

    else:
        difference = (
                np.abs(ref_df.values - dif_df.values) / np.abs(dif_df.values)
        ).mean()

    assert len(ref_columns) == len(dif_columns), "結果 csv の column 数が異なります。"
    assert len(ref_df) == len(dif_df), "結果 csv の row 数が異なります。"
    assert difference <= threashold*100, f"前回の結果との平均差異が {int(difference)}% で {int(threashold*100)}% を超えています。"


def _get_simplified_df_values(csv_path, exclude_columns=None):
    exclude_columns = exclude_columns if exclude_columns is not None else []

    with open(csv_path, 'r', encoding='cp932') as f:
        meta_header = f.readline()
    meta_header = 'removed' + meta_header.split('}"')[-1]
    meta_header = meta_header.split(',')

    df = pd.read_csv(csv_path, encoding='cp932', header=2)

    prm_names = []
    for meta_data, col in zip(meta_header, df.columns):
        if meta_data == 'prm':
            if col not in exclude_columns:
                prm_names.append(col)

    obj_names = []
    for meta_data, col in zip(meta_header, df.columns):
        if meta_data == 'obj':
            if col not in exclude_columns:
                obj_names.append(col)

    pdf = pd.DataFrame()

    for col in prm_names:
        pdf[col] = df[col]

    for col in obj_names:
        pdf[col] = df[col]

    return pdf.values.astype(float)


