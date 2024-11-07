import warnings
from functools import wraps


def experimental_feature(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"The function '{func.__name__}' is experimental and may change in the future.", 
                      category=UserWarning, 
                      stacklevel=2)
        return func(*args, **kwargs)
    return wrapper


if __name__ == '__main__':

    # 使用例
    @experimental_feature
    def my_experimental_function():
        print("This is an experimental function.")

    # 実行すると、警告が表示されます。
    my_experimental_function()
