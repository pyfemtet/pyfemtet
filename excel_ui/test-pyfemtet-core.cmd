cd %~dp0
rem poetry run python pyfemtet-core.py --help
rem poetry run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3
rem poetry run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3 --n_startup_trials=10 --timeout=3.14
rem poetry run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3 --n_startup_trials=10 --timeout=5
rem poetry run python pyfemtet-core.py "xlsm" "femprj" "input_sheet" 3.14 --n_startup_trials=10 --timeout=5
@REM poetry run python pyfemtet-core.py ^
@REM     インターフェース.xlsm ^
@REM     --input_sheet_name="設計変数" ^
@REM     --output_sheet_name="目的関数" ^
@REM     --constraint_sheet_name="拘束関数" ^
@REM
@REM     --n_parallel=1 ^
@REM     --csv_path="test.csv" ^
@REM     --procedure_name=FemtetMacro.FemtetMain ^
@REM     --setup_procedure_name=PrePostProcessing.setup ^
@REM     --teardown_procedure_name=PrePostProcessing.teardown ^
@REM
@REM     --algorithm=Random ^
@REM     --n_startup_trials=10 ^
@REM
@REM pause

poetry run python "C:\Users\mm11592\Documents\myFiles2\working\1_PyFemtetOpt\PyFemtetDev3\pyfemtet\excel_ui\pyfemtet-core.py"  "C:\Users\mm11592\Documents\myFiles2\working\1_PyFemtetOpt\PyFemtetDev3\pyfemtet\excel_ui\インターフェース.xlsm" --input_sheet_name="設計変数" --output_sheet_name="目的関数" --n_parallel=1 --procedure_name=FemtetMacro.FemtetMain --setup_procedure_name=PrePostProcessing.setup --teardown_procedure_name=PrePostProcessing.teardown --algorithm=Random --constraint_sheet_name="拘束関数" --n_trials=20 & pause


