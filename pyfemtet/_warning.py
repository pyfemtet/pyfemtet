import warnings
from functools import wraps


def show_experimental_warning(feature_name):
    warnings.warn(
        f"The function '{feature_name}' is experimental and may change in the future.",
        category=UserWarning,
        stacklevel=2
    )


def experimental_feature(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        show_experimental_warning(func.__name__)
        return func(*args, **kwargs)

    return wrapper


def experimental_class(cls):

    class Wrapper(cls):
        def __new__(cls, *args, **kwargs):
            warnings.warn(
                f"The class '{cls.__name__}' is experimental and may change in the future.",
                category=UserWarning,
                stacklevel=2
            )
            return super().__new__(cls)

    Wrapper.__name__ = cls.__name__
    Wrapper.__doc__ = cls.__doc__
    return Wrapper


def changed_feature(version, additional_message=None):

    def changed_feature_wrapper(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            version_string = f'in version {version}' if version is not None else 'in recent update'
            additional_message_string = ' ' + additional_message if additional_message is not None else ''

            warnings.warn(
                f"The behavior of function '{func.__name__}' is changed {version_string}." + additional_message_string,
                category=UserWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)

        return wrapper

    return changed_feature_wrapper


if __name__ == '__main__':

    # 使用例
    @experimental_feature
    def my_experimental_function():
        print("This is an experimental function.")

    class Sample:

        def __init__(self):
            show_experimental_warning("Sample")

        @experimental_feature
        def add(self, a, b):
            return a + b

    # 実行すると、警告が表示されます。
    my_experimental_function()
    sample = Sample()
    print(sample.add(1, 2))


    @changed_feature('0.1.0', 'See documentation of `Optimizer` for more details.')
    def product(a, b):
        return a * b

    print(product(1, 2))
