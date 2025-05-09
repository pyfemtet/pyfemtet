def objective(_, arg1):
    return arg1


__objective_functions__ = [
    dict(
        name='user_defined_objective',
        fun=objective,
        direction=0.,
        args=(1.,)
    ),
]
