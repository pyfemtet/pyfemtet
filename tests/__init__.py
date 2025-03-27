import os


def get(__file__, filename):
    return os.path.join(os.path.dirname(__file__), filename)
