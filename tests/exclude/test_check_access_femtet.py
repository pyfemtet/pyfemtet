from pyfemtet.opt._femopt_core import (
    _is_access_gogh,
    _is_access_femtet
)


def some_normal_function_without_access_femtet(Femtet, args):
    return True


def some_normal_function_with_access_femtet(Femtet, args):
    print(Femtet.Version)
    return True


def some_normal_function_with_access_gogh(Femtet, args):
    print(Femtet.Gogh)
    return True


class SomeObject:
    def some_instance_method_with_gogh(self, Femtet, args):
        print(Femtet.Gogh)
        return True


if __name__ == '__main__':
    # without femtet
    ret = _is_access_gogh(some_normal_function_without_access_femtet)
    print(ret)
    assert ret == False

    # with femtet
    ret = _is_access_gogh(some_normal_function_with_access_femtet)
    print(ret)
    assert ret == False

    # with gogh
    ret = _is_access_gogh(some_normal_function_with_access_gogh)
    print(ret)
    assert ret == True

    # with gogh
    sample = SomeObject()
    ret = _is_access_gogh(sample.some_instance_method_with_gogh)
    print(ret)
    assert ret == True

    # with femtet
    ret = _is_access_femtet(some_normal_function_with_access_femtet)
    print(ret)
    assert ret == True
