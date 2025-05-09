cd %~dp0
rem uv run python pyfemtet-core.py --help
rem uv run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3
rem uv run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3 --n_startup_trials=10 --timeout=3.14
rem uv run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3 --n_startup_trials=10 --timeout=5
rem uv run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3.14 --n_startup_trials=10 --timeout=5
uv run python pyfemtet-core.py ^
    �C���^�[�t�F�[�X.xlsm ^
    --input_sheet_name="�݌v�ϐ�" ^
    --output_sheet_name="�ړI�֐�" ^
    --constraint_sheet_name="�S���֐�" ^

    --n_parallel=1 ^
    --csv_path="test.csv" ^
    --procedure_name=FemtetMacro.FemtetMain ^
    --setup_procedure_name=PrePostProcessing.setup ^
    --teardown_procedure_name=PrePostProcessing.teardown ^

    --algorithm=Random ^
    --n_startup_trials=10 ^

pause
