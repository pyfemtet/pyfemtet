from .i18n import _


class Message:
    # rule
    TEST = _('Hello!')

    # ===== common =====
    ENTER_TO_QUIT = _('Press enter to quit...')
    CONSTRAINT = _('constraint')
    HIDDEN_CONSTRAINT = _('hidden constraint')

    # ===== pyfemtet.dispatch_extensions =====
    WAIT_FOR_LAUNCH_FEMTET = _('Waiting for launch femtet...')
    TRY_TO_CONNECT_FEMTET = _('Trying to connect Femtet...')

    @staticmethod
    def F_FEMTET_CONNECTED(pid):
        return _(
            'Successfully connected. The pid of Femtet is {pid}.',
            pid=pid
        )

    @staticmethod
    def F_SEARCHING_FEMTET_WITH_SPECIFIC_PID(pid):
        return _(
            'Searching Femtet (pid = {pid}) ...',
            pid=pid
        )

    @staticmethod
    def F_ERR_FEMTET_CONNECTION_TIMEOUT(pid, timeout):
        return _(
            'Connection to Femtet (pid = {pid}) was not'
            'established in {timeout} sec',
            pid=pid,
            timeout=timeout,
        )

    # ===== pyfemtet.opt femopt core =====
    ERR_CHECK_MINMAX = _('The magnitude relationship is incorrect.')
    ERR_CHECK_DIRECTION = _("The direction of the objective function must be 'minimize', 'maximize' or a number. ")
    ERR_CANNOT_ENCODING = _('The variable name, object name, or constraint name contains characters that cannot be encoded. Do not use environment-dependent characters.')
    ERR_PROBLEM_MISMATCH = _('The running configuration does not match the configuration in the csv file.')

    # ===== pyfemtet.opt FEMOpt =====
    ERR_NO_INITIAL_VALUE = _('Please specify initial_value.')
    ERR_CONTAIN_GOGH_ACCESS_IN_STRICT_CONSTRAINT = _(
        'Constraint functions are evaluated before analysis is performed, '
        'so do not access Femtet.Gogh. If you want your '
        'constraints to include values after analysis is performed, set '
        'the `strict` argument to False.')

    OPTIMIZATION_FINISHED = _('Optimization finished. Results were saved in following:')
    ERR_NO_BOUNDS = _('No bounds specified.')
    CONFIRM_BEFORE_EXIT = _('The optimization is now complete. You can view the results on the monitor until you press Enter to exit the program.')

    # ===== pyfemtet.opt.interface =====
    ERR_RUN_JOURNAL_NOT_FOUND = _(r'"%UGII_BASE_DIR%\NXBIN\run_journal.exe" is not found. Make sure you have NX installed and the environment variable UGII_BASE_DIR is set.')
    ERR_MODEL_RECONSTRUCTION_FAILED = _('Model reconstruction failed.')
    ERR_MODEL_UPDATE_FAILED = _('Model update failed.')
    ERR_FEMTET_CONNECTION_FAILED = _('Failed to connect to Femtet.')

    @staticmethod
    def F_ERR_FEMTET_CRASHED_AND_RESTART_FAILED(name):
        return _(
            'The Femtet process crashed and '
            'Python failed to restore Femtet. '
            'API: {name}',
            name=name
        )

    @staticmethod
    def F_WARN_FEMTET_CRASHED_AND_TRY_RESTART(name):
        return _(
            'An abnormal termination of the Femtet '
            'process has been detected. Recovery will '
            'be attempted. API: {name}',
            name=name,
        )

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
    ERR_OPEN_RESULT_FAILED = _('Failed to open result.')
    ERR_CLOSE_FEMTET_FAILED = _('Failed to close Femtet.')
    ERR_FAILED_TO_SAVE_PDT = _('Failed to save result (.pdt) file.')
    ERR_FAILED_TO_SAVE_JPG = _('Failed to save screenshot (.jpg).')
    ERR_JPG_NOT_FOUND = _('Screenshot (.jpg) is not found.')
    ERR_UPDATE_SOLIDWORKS_MODEL_FAILED = _('Failed to update model in solidworks.')
    INFO_POF_IS_LESS_THAN_THRESHOLD = _('Probability of feasibility is less than threshold.')
    INFO_TERMINATING_EXCEL = _('Terminating Excel process...')
    INFO_TERMINATED_EXCEL = _('Excel process is terminated.')
    INFO_RESTORING_FEMTET_AUTOSAVE = _('Restore Femtet setting of autosave.')
    ERR_PARAMETRIC_CSV_CONTAINS_ERROR = _('Failed to make output from Femtet. '
                                          'Please check output settings of Parametric Analysis.')

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

    # ===== optuna optimizer =====
    @staticmethod
    def F_WARN_INVALID_ARG_FOR_SAMPLER(key, sampler_name):
        return _(
            'The given argument {key} is not '
            'included in ones of {sampler_name}. '
            '{key} is ignored.',
            key=key, sampler_name=sampler_name,
        )

    # ===== scipy optimizer =====
    WARN_SCIPY_NELDER_MEAD_BOUND = _(
        'Sometimes Nelder-Mead cannot start optimization '
        'with the initial condition what is same with '
        'lower bounds or upper bounds.'
    )
    ERR_SCIPY_NOT_IMPLEMENT_CATEGORICAL = _(
        'Cannot use categorical parameter with ScipyOptimizer'
    )
    ERR_SCIPY_HARD_CONSTRAINT_VIOLATION = _(
        en_message='Hard constraint violation! scipy cannot continue '
                   'optimization. Only `SLSQP` supports optimization '
                   'with hard constraint optimization problem. '
                   'If you see this message even if you are using it, '
                   'please try to following:\n'
                   '- Use small `eps` by `options` argument.\n'
                   '- Set `constraint_enhancement`to the value that '
                   'it larger than the variation of the constraint '
                   'function when input variables within the range '
                   'of `eps`.',
        jp_message='hard constraint を扱うには'
                   'method に SLSQP を使用してください。'
                   'SLSQP を使用しているのにこのメッセージが表示される場合、\n'
                   '- options から eps を小さくするか、'
                   '- eps の値の範囲内で x が変動したときの'
                   '拘束関数の変動量を上回るように'
                   'constraint_enhancement を大きく\n'
                   'してみてください。',
    )
    ERR_SCIPY_HIDDEN_CONSTRAINT = _(
        en_message='ScipyOptimizer cannot continue optimization '
                   'when encountered the input variables that '
                   'break FEM model.',
        jp_message='ScipyOptimizer では解析ができない'
                   '設計変数の組合せをスキップできません。'
    )
    ERR_SCIPY_NOT_IMPLEMENT_SKIP = _('ScipyOptimizer cannot skip solve.')

    @staticmethod
    def F_ERR_SCIPY_METHOD_NOT_IMPLEMENT_HARD_CONSTRAINT(method):
        return _('{method} cannot handle hard constraint.', method=method)

    WARN_SCIPY_SLSQP_CANNOT_PROCESS_SOFT_CONSTRAINT = _(
        'SLSQP cannot handle soft constraint. '
        'The constraint is handled as a hard one.'
    )

    # ===== PoFBoTorchSampler =====
    WARN_USING_FEM_IN_NLC = _(
        'Accessing FEM API inside hard constraint '
        'function may be very slow.')
    WARN_NO_FEASIBLE_BATCH_INITIAL_CONDITION = _(
        'gen_batch_initial_conditions() failed to generate '
        'feasible initial conditions for acquisition '
        'function optimization sub-problem, '
        'so trying to use random feasible parameters '
        'as initial conditions.'
        'The constraint functions or solutions spaces '
        'may be too complicated.'
    )

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
    GRAPH_AXIS_LABEL_TRIAL = _('Succeeded trial number')
    LEGEND_LABEL_ALL_SOLUTIONS = _('All solutions')
    LEGEND_LABEL_OPTIMAL_SOLUTIONS = _('Transition of<br>optimal solutions')
    LEGEND_LABEL_OBJECTIVE_TARGET = _('Target value')
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
    # _process_monitor application
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
    DETAIL_PAGE_IMPORTANCE_HEADER = _('The importance of parameters evaluated by fANOVA')
    DETAIL_PAGE_IMPORTANCE_DESCRIPTION = _('The normalized relative importance of parameters. Please note that the importance is calculated from the overall relationship of the input-output response, rather than from a specific solution.')
