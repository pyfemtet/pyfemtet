
from pyfemtet.opt import FemtetInterface, OptunaOptimizer, FEMOpt


def main():

    femprj_path = r"C:\Users\mm11592\Documents\myFiles2\working\1_PyFemtetOpt\PyFemtetDev3\pyfemtet\pyfemtet\opt\visualization2\result_viewer\sample\sample.femprj"
    model_name = "解析モデル"
    fem = FemtetInterface(
        femprj_path=femprj_path,
        model_name=model_name,
        parametric_output_indexes_use_as_objective={
            0: "minimize",
            1: "minimize",
            2: "minimize",
        },
    )

    femopt = FEMOpt(fem=fem)

    femopt.add_parameter("w", 10, 1, 10)
    femopt.add_parameter("d", 10, 1, 10)
    femopt.add_parameter("h", 10, 1, 10)
    femopt.optimize(
        n_trials=10,
        n_parallel=1,
    )
    femopt.terminate_all()

if __name__ == '__main__':
    main()
