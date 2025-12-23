import os
import shutil
import pyfemtet
import pytest


here = os.path.dirname(__file__)
root = os.path.dirname(pyfemtet.__file__)


@pytest.mark.femtet
def test_multiple_models():
    # Copy target path to temporary path
    target_path = os.path.join(root, "..", "samples", "opt", "advanced_samples", "multiple_models", "optimize_with_multiple_models.py")
    distination_path = os.path.join(here, "tmp_script.py")
    shutil.copyfile(target_path, distination_path)
    target_femprj = os.path.join(root, "..", "samples", "opt", "advanced_samples", "multiple_models", "cylinder-shaft-cooling.femprj")
    distination_femprj = os.path.join(here, "cylinder-shaft-cooling.femprj")
    shutil.copyfile(target_femprj, distination_femprj)

    # Replace optimize() arguments to make the test deterministic
    with open(distination_path, "r", encoding="utf-8") as f:
        content = f.read()
    history_path = os.path.join(here, "test_tmp.csv")
    content = content.replace(
        "    femopt.optimize(\n"
        "        n_trials=50,\n"
        "        confirm_before_exit=False,\n"
        "    )"
        "",
        "    femopt.optimize(\n"
        "        n_trials=3,\n"
        "        confirm_before_exit=False,\n"
        "        seed=42,\n"
        f"        history_path=r'{history_path}'\n"
        "    )"
        "",
    )
    with open(distination_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Remove previous temporary files
    if os.path.isfile(history_path):
        os.remove(history_path)
    if os.path.isfile(history_path.replace(".csv", ".db")):
        os.remove(history_path.replace(".csv", ".db"))

    # Execute the script
    from runpy import run_path
    run_path(distination_path, init_globals={}, run_name="__main__")

    # Check the result .reccsv
    from tests.utils.reccsv_processor import RECCSV
    from pyfemtet.opt.history import History


    history = History()
    history.load_csv(history_path, with_finalize=True)
    df = history.get_df()

    reccsv_path = os.path.join(here, "test_multiple_models")
    reccsv_checker = RECCSV(
        base_path=os.path.splitext(reccsv_path)[0],
        record=False,
    )
    reccsv_checker.check(
        check_columns_float=[
            'internal_radius',
            'cooling_area_radius',
            '0: 定常解析 / 温度[deg] / 最大値 / 全てのボディ属性',
            'difference_between_nearest_natural_frequency',
        ],
        check_columns_str=[],
        dif_df=df,
        rtol=0.01,
    )


@pytest.mark.excel
@pytest.mark.femtet
def test_excel_ui():
    # Copy target path to temporary path
    target_path = os.path.join(root, "..", "samples", "opt", "advanced_samples", "excel_ui", "femtet-macro.xlsm")
    distination_path = os.path.join(here, "tmp_femtet_macro.xlsm")
    shutil.copyfile(target_path, distination_path)
    target_core_py = os.path.join(root, "..", "samples", "opt", "advanced_samples", "excel_ui", "pyfemtet-core.py")
    distination_core_py = os.path.join(here, "pyfemtet-core.py")
    shutil.copyfile(target_core_py, distination_core_py)
    
    # Remove previous temporary files
    history_path = os.path.join(here, "test_excel_ui_history.csv")
    if os.path.isfile(history_path):
        os.remove(history_path)
    if os.path.isfile(history_path.replace(".csv", ".db")):
        os.remove(history_path.replace(".csv", ".db"))

    # Replace pyfemtet-core.py to disable confirm before exit
    with open(distination_core_py, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        "        confirm_before_exit=True,",
        "        confirm_before_exit=False,",
    )
    with open(distination_core_py, "w", encoding="utf-8") as f:
        f.write(content)

    # Execute the script
    import sys
    import gc
    from win32com.client import DispatchEx
    from pythoncom import CoInitialize
    from glob import glob
    try:
        CoInitialize()
        excel = DispatchEx("Excel.Application")
        excel.DisplayAlerts = False
        excel.Interactive = False
        excel.Visible = False
        for path in glob(r"C:\Program Files\Microsoft Office\root\Office16\XLSTART\*"):
            excel.Workbooks.Open(path)
        wb = excel.Workbooks.Open(distination_path)
        ws = wb.WorkSheets("最適化の設定")
        ws.Range("C11").Value = 3  # n_trials
        ws.Range("C13").Value = 42  # seed
        ws.Range("C14").Value = history_path
        ws.Range("C16").Value = sys.executable
        wb.Save()  # save settings
        excel.Run("call_pyfemtet")
        wb.Save()  # avoid dialog for unsaved book

        from time import sleep
        # call_pyfemtet is asynchronous
        for i in range(70):
            sleep(1)
            print(f"Waiting for optimization to complete... {i+1}/70s")

    finally:
        excel.Application.Quit()
        del excel
        gc.collect()

    # Check the result .reccsv
    from tests.utils.reccsv_processor import RECCSV
    from pyfemtet.opt.history import History
    history = History()
    history.load_csv(history_path, with_finalize=True)
    df = history.get_df()

    reccsv_path = os.path.join(here, "test_excel_ui_history.reccsv")
    reccsv_checker = RECCSV(
        base_path=os.path.splitext(reccsv_path)[0],
        record=False,
    )
    reccsv_checker.check(
        check_columns_float=[
            'section_radius',
            'インダクタンス',
        ],
        check_columns_str=[],
        dif_df=df,
        rtol=0.01,
    )


if __name__ == "__main__":
    test_multiple_models()
    test_excel_ui()
