CONFIRM_BEFORE_ABNORMAL_TERMINATION = True

try:

    # for meta script
    import os
    import sys
    import importlib
    import pyfemtet

    if __name__ == '__main__':
        print(f'pyfemtet {pyfemtet.__version__} starting.')

    from pyfemtet._i18n import ENCODING

    from fire import Fire
    import yaml

    # for concrete script
    from optuna.samplers import *
    from pyfemtet.opt import FEMOpt
    from pyfemtet.opt.interface import *
    from pyfemtet.opt.optimizer import *

    class ContentContext:

        def __init__(self, content_):
            self.content = content_

        def __enter__(self):
            return self.content

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # for importing user-defined module
    def import_from_path(module_name, file_path):

        print(f'{file_path=}')

        # noinspection PyUnresolvedReferences
        spec = importlib.util.spec_from_file_location(module_name, file_path)

        print(f'{spec=}')

        # noinspection PyUnresolvedReferences
        module = importlib.util.module_from_spec(spec)

        print(f'{module=}')

        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


    def main(
            yaml_path: str = None,

            interface_class: str = None,  # including parameter definition excel
            interface_kwargs: str | dict = None,  # including Parametric Analysis Output
            optimizer_class: str = None,
            optimizer_kwargs: str | dict = None,
            seed: str = 'null',
            optimize_kwargs: str | dict = None,
            additional_module_paths: str | list[str] = None,

            confirm_before_abnormal_termination: bool = True,
    ):
        """

        Args:
            yaml_path:
                If this argument is passed, the other arguments will be ignored
                and load by .yaml file.
                The yaml file must contain the other arguments.

            interface_class: Class whose name ends with `~WithExcelSettingsInterface`.
            interface_kwargs: See documentation of each interface class.
            optimizer_class: OptunaOptimizer or ScipyOptimizer.
            optimizer_kwargs: See documentation of each optimizer class.
            seed: int or None.
            optimize_kwargs: See documentation of FEMOpt.optimize().
            additional_module_paths:
                The .py file paths containing user-defined objective and constraints functions.
                The module must contain __objective_functions__ or __constraint_functions__.
                They must be a `list` of `dict` that contains the required arguments of
                `FEMOpt.add_objective()` or `FEMOpt.add_constraint()` method.
                Note that the `args` argument of constraint function is reserved by meta_script
                and that value is fixed to `FEMOpt.opt`, so you can not use `args` argument
                in __constraint_functions__.
                Note that the functions should not contain the python variables defined outside
                global scope.
            confirm_before_abnormal_termination: Pause before termination if an error occurs. Default is True.

        """

        global CONFIRM_BEFORE_ABNORMAL_TERMINATION
        CONFIRM_BEFORE_ABNORMAL_TERMINATION = confirm_before_abnormal_termination

        # Override arguments,
        # if yaml is given.
        # ===================
        if yaml_path is not None:

            # check
            if os.path.isfile(yaml_path):
                # noinspection PyBroadException
                try:
                    context = open(yaml_path, 'r', encoding='utf-8')
                except Exception:
                    context = open(yaml_path, 'r', encoding=ENCODING)

            else:
                raise FileNotFoundError(yaml_path)

            # load
            with context as f:
                d = yaml.safe_load(f)
                interface_class = yaml.safe_dump(d['interface_class'], allow_unicode=True)
                interface_kwargs = yaml.safe_dump(d['interface_kwargs'], allow_unicode=True)
                optimizer_class = yaml.safe_dump(d['optimizer_class'], allow_unicode=True)
                optimizer_kwargs = yaml.safe_dump(d['optimizer_kwargs'], allow_unicode=True)
                seed = yaml.safe_dump(d['seed'], allow_unicode=True)
                optimize_kwargs = yaml.safe_dump(d['optimize_kwargs'], allow_unicode=True)
                additional_module_paths = yaml.safe_dump(d.get('additional_module_paths', None), allow_unicode=True)

        # Construct python variables
        # by evaluating string
        # or Fire arguments
        # ==========================

        # additional import
        # -----------------
        if yaml_path is not None:
            additional_module_paths_ = yaml.safe_load(additional_module_paths)
        else:
            additional_module_paths_ = [yaml.safe_load(p) for p in additional_module_paths]

        if additional_module_paths_ is not None:

            for i, path_ in enumerate(additional_module_paths_):

                # noinspection PyUnresolvedReferences
                additional_module = import_from_path(f'additional_module_{i}', path_)

                # from additional_module import *
                if hasattr(additional_module, '__all__'):
                    globals().update({key: getattr(additional_module, key) for key in additional_module.__all__})

        # Python variables
        # ----------------
        if yaml_path is not None:
            Interface = eval(yaml.safe_load(interface_class))
            interface_kwargs_ = yaml.safe_load(interface_kwargs)

            Optimizer = eval(yaml.safe_load(optimizer_class))
            optimizer_kwargs_ = yaml.safe_load(optimizer_kwargs)

            optimizer_kwargs_['sampler_class'] = eval(optimizer_kwargs_['sampler_class'])

            seed_ = yaml.safe_load(seed)

            optimize_kwargs_ = yaml.safe_load(optimize_kwargs)

        else:
            Interface = eval(interface_class)
            interface_kwargs_ = interface_kwargs

            Optimizer = eval(optimizer_class)
            optimizer_kwargs_ = optimizer_kwargs

            optimizer_kwargs_['sampler_class'] = eval(optimizer_kwargs_['sampler_class'])

            seed_ = seed

            optimize_kwargs_ = optimize_kwargs

        # meta-script
        # ===========

        # prepare common object
        fem = Interface(**interface_kwargs_)
        opt = Optimizer(**optimizer_kwargs_)
        femopt = FEMOpt(fem=fem, opt=opt)

        # add user-defined functions
        if additional_module_paths_ is not None:

            for i, path_ in enumerate(additional_module_paths_):

                # noinspection PyUnresolvedReferences
                additional_module = import_from_path(f'additional_module_{i}', path_)

                # import obj & cns
                d: dict
                if hasattr(additional_module, '__objective_functions__'):
                    for d in additional_module.__objective_functions__:
                        femopt.add_objective(
                            fun=d['fun'],
                            name=d.get('name'),
                            direction=d.get('direction'),
                            args=d.get('args'),
                            kwargs=d.get('kwargs'),
                        )
                if hasattr(additional_module, '__constraint_functions__'):
                    for d in additional_module.__constraint_functions__:
                        femopt.add_constraint(
                            fun=d['fun'],
                            name=d.get('name'),
                            lower_bound=d.get('lower_bound'),
                            upper_bound=d.get('upper_bound'),
                            strict=d.get('strict', True),
                            using_fem=d.get('using_fem'),
                            args=(femopt.opt,),
                            kwargs=d.get('kwargs'),
                        )

        # optimize
        femopt.set_random_seed(seed_)
        femopt.optimize(**optimize_kwargs_)


    if __name__ == '__main__':
        Fire(main)


except Exception as e:
    from traceback import print_exception

    print_exception(e)
    print()
    if CONFIRM_BEFORE_ABNORMAL_TERMINATION:
        input('Press Enter to quit...')

    raise e  # for test
