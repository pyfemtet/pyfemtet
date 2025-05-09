import sys
import json
import subprocess
from pathlib import Path

here = Path(__file__).parent


def test_meta_script_yaml():
    subprocess.run(
        [
            sys.executable,
            '-m',
            'pyfemtet.opt.meta_script',
            f'--yaml_path={str(here / "test.yaml")}'
        ],
        cwd=str(here)
    ).check_returncode()


def test_meta_script():
    interface_class = 'NoFEM'
    interface_kwargs = json.dumps(dict())
    optimizer_class = 'OptunaOptimizer'
    optimizer_kwargs = json.dumps(dict(
        sampler_class='TPESampler',
        sampler_kwargs=dict(
            n_startup_trials=3,
        )
    ))
    seed = 42
    optimize_kwargs = json.dumps(dict(
        n_trials=3,
        confirm_before_exit=False,
    )).replace('false', 'False')  # 小文字はダメなので注意
    additional_module_paths = json.dumps([
        str((Path(__file__).parent / 'user_defined_function.py').absolute())
    ])

    subprocess.run(
        [
            sys.executable,
            '-m',
            'pyfemtet.opt.meta_script',
            f'--interface_class={interface_class}',
            f'--interface_kwargs={interface_kwargs}',
            f'--optimizer_class={optimizer_class}',
            f'--optimizer_kwargs={optimizer_kwargs}',
            f'--seed={seed}',
            f'--optimize_kwargs={optimize_kwargs}',
            f'--additional_module_paths={additional_module_paths}',
            f'--confirm_before_abnormal_termination=False',
        ],
    ).check_returncode()


if __name__ == '__main__':
    test_meta_script_yaml()
    test_meta_script()
