import os
import locale
from dataclasses import dataclass

loc, encoding = locale.getlocale()

if __name__ == '__main__':
    # set target to create .pot
    from gettext import gettext as _

else:
    # get translation
    if 'japanese' in loc.lower():
        from babel.support import Translations

        translations = Translations.load(
            os.path.join(os.path.dirname(__file__), 'locales'),
            locales='ja'
        )
        _ = translations.gettext

    else:
        def _(x):
            return x


@dataclass
class Message:
    # syntax
    HELLO = _('hello!')
    # This is ignored by babel
    def some_string(self, value): return _(f'value is {value}')

    # ===== common =====
    ENTER_TO_QUIT = _('Press enter to quit...')

    # ===== pyfemtet.opt femopt core =====
    ERR_CHECK_MINMAX = _('The magnitude relationship is incorrect.')
    ERR_CHECK_DIRECTION = _("The direction of the objective function must be 'minimize', 'maximize' or a number. ")
    ERR_CANNOT_ENCODING = _('The variable name, object name, or constraint name contains characters that cannot be encoded. Do not use environment-dependent characters.')
    ERR_PROBLEM_MISMATCH = _('The running configuration does not match the configuration in the csv file.')

    # ===== pyfemtet.opt FEMOpt =====
    ERR_NO_INITIAL_VALUE = _('Please specify initial_value.')
    ERR_CONTAIN_GOGH_ACCESS_IN_STRICT_CONSTRAINT = _('Constraint functions are evaluated before analysis is performed, '
                                                     'so do not access Femtet.Gogh. If you want your '
                                                     'constraints to include values after analysis is performed, set '
                                                     'the `strict` argument to False.')
    WARN_HISTORY_CSV_NOT_ACCESSIBLE = _('History csv file is in use and cannot be written to. '
                                        'Please free this file before exiting the program, '
                                        'otherwise history data will be lost.')
    OPTIMIZATION_FINISHED = _('Optimization finished. Results were saved in following:')
    ERR_NO_BOUNDS = _('No bounds specified.')
    CONFIRM_BEFORE_EXIT = _('The optimization is now complete. You can view the results on the monitor until you press Enter to exit the program.')

    # ===== pyfemtet.opt.interface =====
    ERR_RUN_JOURNAL_NOT_FOUND = _(r'"%UGII_BASE_DIR%\NXBIN\run_journal.exe" is not found. Make sure you have NX installed and the environment variable UGII_BASE_DIR is set.')
    ERR_MODEL_RECONSTRUCTION_FAILED = _('Model reconstruction failed.')
    ERR_MODEL_UPDATE_FAILED = _('Model update failed.')
    ERR_NO_MAKEPY = _('It was detected that the configuration of Femtet python macro constants has not been completed. The configuration was done automatically (python -m win32com.client.makepy FemtetMacro). Please restart the program.')
    ERR_FEMTET_CONNECTION_FAILED = _('Failed to connect to Femtet.')
    ERR_FEMTET_CRASHED_AND_RESTART_FAILED = _('The Femtet process crashed and could not be restarted successfully.')
    WARN_FEMTET_CRASHED_AND_TRY_RESTART = _('An abnormal termination of the Femtet process has been detected. Recovery will be attempted.')
    INFO_FEMTET_CRASHED_AND_RESTARTED = _('Femtet has been restarted and will perform analysis and attempt to recover.')
    ERR_NEW_FEMTET_BUT_NO_FEMPRJ = _("If you specify 'new' as the 'connect_method' argument, set the 'femprj_path' argument to existing femprj file path.")
    ERR_NO_SUCH_PARAMETER_IN_FEMTET = _('The specified variable is not included in the Femtet analysis model. Note the capitalization of the variable.')
    ERR_CANNOT_ACCESS_API = _('The following APIs are not accessible: ')
    CERTIFY_MACRO_VERSION = _('Macros may not be enabled in the installed version of Femtet. Please run the "Enable Macros" command from the start menu with administrator privileges in the same version of Femtet that is installed.')
    NO_ANALYSIS_MODEL_IS_OPEN = _('No analysis model is open')
    FEMTET_ANALYSIS_MODEL_WITH_NO_PARAMETER = _('The analysis model does not contain any variables.')
    ERR_FAILED_TO_UPDATE_VARIABLE = _('Failed to update variables:')
    WARN_IGNORE_PARAMETER_NOT_CONTAINED = _('The specified variable is not included in the analysis model and will be ignored.')
    ERR_RE_EXECUTE_MODEL_FAILED = _('Model history re-execute failed.')
    ERR_MODEL_REDRAW_FAILED = _('Model redraw failed.')
    ERR_MODEL_MESH_FAILED = _('Mesh generation failed')
    ERR_PARAMETRIC_SOLVE_FAILED = _('Parametric solve failed')
    ERR_SOLVE_FAILED = _('Solve failed.')
    ERR_OPEN_RESULT_FAILED = _('Open result failed.')
    ERR_CLOSE_FEMTET_FAILED = _('Failed to close Femtet.')
    ERR_FAILED_TO_SAVE_PDT = _('Failed to save result (.pdt) file.')
    ERR_FAILED_TO_SAVE_JPG = _('Failed to save screenshot (.jpg).')
    ERR_JPG_NOT_FOUND = _('Screenshot (.jpg) is not found.')
    ERR_UPDATE_SOLIDWORKS_MODEL_FAILED = _('Failed to update model in solidworks.')

    # ===== pyfemtet.opt.optimizer =====
    ERR_NOT_IMPLEMENTED = _('The following features are not supported by the specified optimization method. ')
    ERR_INCONSISTENT_PARAMETER = _('The parameter set does not match the one added with add_init_parameter.')
    INFO_EXCEPTION_DURING_FEM_ANALYSIS = _('An exception has occurred during FEM update. Current parameters are: ')
    INFO_INFEASIBLE = _('The constraints were not satisfied for the following sets of variables:')
    ERR_FEM_FAILED_AND_CANNOT_CONTINUE = _('Current parameter set cannot update FEM and this optimization method cannot skip current parameter set. The optimization process will be terminated.')
    WARN_INTERRUPTED_IN_SCIPY = _('Optimization has been interrupted. Note that you cannot acquire the OptimizationResult in case of `trust-constr`, `TNC`, `SLSQP` or `COBYLA`.')
    ERR_PARAMETER_CONSTRAINT_ONLY_BOTORCH = _('You can use parameter constraint only with BoTorchSampler.')
    WARN_SCIPY_DOESNT_NEED_SEED = _('Scipy is deterministic, so whether you set a seed or not will not change the results.')
    START_CANDIDATE_WITH_PARAMETER_CONSTRAINT = _('Start to candidate new parameter set with constraints. This process may take a long time.')
    WARN_UPDATE_FEM_PARAMETER_TOOK_A_LONG_TIME = _('Updating FEM parameter during evaluating constraints take a long time. Please consider not to use FEM variables in constraint functions and set `update_fem` to False.')

    # ===== pyfemtet.opt.visualization =====
    # control_femtet.py
    LABEL_CONNECT_FEMTET_BUTTON = _('Connect to Femtet')
    WARN_CSV_MODEL_NAME_IS_INVALID = _('Analysis model name described in csv does not exist in project.')
    WARN_HISTORY_CSV_NOT_READ = _('History csv is not read yet. Open your project manually.')
    WARN_INVALID_METADATA = _('Cannot read project data from csv. Open your project manually.')
    WARN_FEMPRJ_IN_CSV_NOT_FOUND = _('.femprj file described in csv is not found. Open your project manually.')
    WARN_MODEL_IN_CSV_NOT_FOUND_IN_FEMPRJ = _('Analysis model name is not specified. Open your model in the project manually.')
    # main_figure_creator.py
    LEGEND_LABEL_CONSTRAINT = _('Constraint')
    LEGEND_LABEL_FEASIBLE = _('feasible')
    LEGEND_LABEL_INFEASIBLE = _('infeasible')
    LEGEND_LABEL_OPTIMAL = _('Optimality')
    LEGEND_LABEL_NON_DOMI = _('non dominated')
    LEGEND_LABEL_DOMI = _('dominated')
    GRAPH_TITLE_HYPERVOLUME = _('Hypervolume Plot')
    GRAPH_TITLE_SINGLE_OBJECTIVE = _('Objective Plot')
    GRAPH_TITLE_MULTI_OBJECTIVE = _('Multi Objective Pair Plot')
    GRAPH_AXIS_LABEL_TRIAL = _('trial number')
    # main_graph.py
    TAB_LABEL_OBJECTIVE_PLOT = _('Objectives')
    TAB_LABEL_OBJECTIVE_SCATTERPLOT = _('Objectives (all)')
    # pm_graph.py
    TAB_LABEL_PREDICTION_MODEL = _('Prediction Model')
    LABEL_OF_CREATE_PREDICTION_MODEL_BUTTON = _(' Recalculate Model')
    LABEL_OF_REDRAW_PREDICTION_MODEL_GRAPH_BUTTON = _(' Redraw graph')
    LABEL_OF_AXIS1_SELECTION = _('Parameter')
    LABEL_OF_AXIS2_SELECTION = _('Parameter2')
    LABEL_OF_AXIS3_SELECTION = _('Objective')
    ERR_NO_HISTORY_SELECTED = _('No history selected.')
    ERR_NO_FEM_RESULT = _('No FEM result (yet).')
    ERR_NO_PREDICTION_MODEL = _('Prediction model is not calculated yet.')
    ERR_CANNOT_SELECT_SAME_PARAMETER = _('Cannot select same parameter')
    LABEL_SWITCH_PREDICTION_MODEL_3D = _('3D graph (two or more parameters required)')
    # pm_graph_creator
    GRAPH_TITLE_PREDICTION_MODEL = _('Prediction Model of Objective')
    LEGEND_LABEL_PREDICTION_MODEL = _('prediction model')
    LEGEND_LABEL_PREDICTION_MODEL_STDDEV = _('std. dev. of model')
    # process_monitor application
    PAGE_TITLE_PROGRESS = _('Progress')
    PAGE_TITLE_PREDICTION_MODEL = _('Prediction')
    PAGE_TITLE_WORKERS = _('Workers')
    PAGE_TITLE_OPTUNA_VISUALIZATION = _('Details')
    # process monitor pages
    DEFAULT_STATUS_ALERT = _('Optimization status will be shown here.')
    LABEL_AUTO_UPDATE = _('Auto-update graph')
    LABEL_INTERRUPT = _('Interrupt Optimization')
    # result viewer application
    PAGE_TITLE_RESULT = _('Result')
    # result viewer pages
    LABEL_OPEN_PDT_BUTTON = _('Open Result in Femtet')
    LABEL_RECONSTRUCT_MODEL_BUTTON = _('Reconstruct Model')
    LABEL_FILE_PICKER = _('Drag and drop or select csv file')
    ERR_NO_CONNECTION_ESTABLISHED = _('Connection to Femtet is not established. Launch Femtet and Open a project.')
    ERR_NO_SOLUTION_SELECTED = _('No result plot is selected.')
    ERR_FEMPRJ_IN_CSV_NOT_FOUND = _('The femprj file path in the history csv is not found or valid.')
    ERR_MODEL_IN_CSV_NOT_FOUND = _('The model name in the history csv is not found.')
    ERR_PDT_NOT_FOUND = _('.pdt file is not found. '
                          'Please check the .Results folder. '
                          'Note that .pdt file save mode depends on '
                          'the `save_pdt` argument of FemtetInterface in optimization script'
                          '(default to `all`).')
    ERR_FAILED_TO_OPEN_PREFIX = _('Failed to open ')
    ERR_NO_SUCH_MODEL_IN_FEMPRJ = _('Specified model is not in current project. '
                                    'Please check opened project. '
                                    'For example, not "analysis model only" but your .femprj file.')
    WARN_INCONSISTENT_FEMPRJ_PATH = _('.femprj file path of the history csv is invalid. Please certify matching between csv and opening .femprj file.')
    WARN_INVALID_MODEL_NAME = _('Analysis model name of the history csv is invalid. Please certify matching between csv and opening analysis model.')
    WARN_INCONSISTENT_MODEL_NAME = _('Analysis model name of the history csv and opened in Femtet is inconsistent. Please certify matching between csv and opening analysis model.')
    LABEL_TUTORIAL_MODE_SWITCH = _('tutorial mode')
    LABEL_LOAD_SAMPLE_CSV = _('Load Sample CSV')
    LOAD_CSV_POPOVER_HEADER = _('Load CSV')
    LOAD_CSV_POPOVER_BODY = _('Open your optimization result. Then connecting to femtet will start automatically. '
                              'Note that in tutorial mode, this button loads the ready-made sample csv and open sample femprj.')
    MAIN_GRAPH_POPOVER_HEADER = _('Main Graph')
    MAIN_GRAPH_POPOVER_BODY = _('Here the optimization history is shown. '
                                'Each plot represents single FEM result. '
                                'You can pick a result to open the corresponding result in Femtet. ')
    OPEN_PDT_POPOVER_HEADER = _('Open Result')
    OPEN_PDT_POPOVER_BODY = _('After pick a point in the main graph, '
                              'This button shows the corresponding FEM result in Femtet.')
    CONNECT_FEMTET_POPOVER_BODY = _('Re-connect to Femtet.')
    ERR_SAMPLE_CSV_NOT_FOUND = _('Sample csv is not found. Please consider to re-install pyfemtet by `py -m pip install pyfemtet -U --force-reinstall`')
    ERR_SAMPLE_FEMPRJ_NOT_FOUND = _('Sample femprj file is not found. Please consider to re-install pyfemtet by `py -m pip install pyfemtet -U --force-reinstall`')
    ERR_FEMPRJ_RESULT_NOT_FOUND = _('Sample femprj result folder is not found. Please consider to re-install pyfemtet by `py -m pip install pyfemtet -U --force-reinstall`')
    # DETAIL VISUALIZATION PAGES
    DETAIL_PAGE_TEXT_BEFORE_LOADING = _('Loading data...')
    DETAIL_PAGE_HISTORY_HEADER = _('Plots of objectives versus trials')
    DETAIL_PAGE_HISTORY_DESCRIPTION = _('The vertical axis is the objective, and the horizontal axis is the number of trials.')
    DETAIL_PAGE_PARALLEL_COOR_HEADER = _('Parallel coordinate plots')
    DETAIL_PAGE_PARALLEL_COOR_DESCRIPTION = _('The vertical axis is an objective or parameters, and one polyline indicates one result.')
    DETAIL_PAGE_CONTOUR_HEADER = _('The heatmap of objectives')
    DETAIL_PAGE_CONTOUR_DESCRIPTION = _('The axes are parameters, and the color shows objective value.')
    DETAIL_PAGE_SLICE_HEADER = _('The response of an objective versus one parameter')
    DETAIL_PAGE_SLICE_DESCRIPTION = _('The vertical axis is objective, and the horizontal axis is parameter.')

