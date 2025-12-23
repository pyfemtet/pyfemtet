import os
import shutil
import pyfemtet

here = os.path.dirname(__file__)
root = os.path.dirname(pyfemtet.__file__)


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


if __name__ == "__main__":
    test_multiple_models()
