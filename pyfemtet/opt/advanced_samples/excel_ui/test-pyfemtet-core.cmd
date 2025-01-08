cd %~dp0
rem poetry run python pyfemtet-core.py --help
rem poetry run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3
rem poetry run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3 --n_startup_trials=10 --timeout=3.14
rem poetry run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3 --n_startup_trials=10 --timeout=5
rem poetry run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3.14 --n_startup_trials=10 --timeout=5
poetry run python pyfemtet-core.py ^
    インターフェース.xlsm ^
    --input_sheet_name="設計変数" ^
    --output_sheet_name="目的関数" ^
    --constraint_sheet_name="拘束関数" ^

    --n_parallel=1 ^
    --csv_path="test.csv" ^
    --procedure_name=FemtetMacro.FemtetMain ^
    --setup_procedure_name=PrePostProcessing.setup ^
    --teardown_procedure_name=PrePostProcessing.teardown ^

    --algorithm=Random ^
    --n_startup_trials=10 ^

pause
