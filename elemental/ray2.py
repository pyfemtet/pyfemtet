import ray

ray.init()


@ray.remote
class NameSpace:
    def __init__(self):
        self.global_var = 3

    def set_global_var(self, var):
        self.global_var = var

    def get_global_var(self):
        return self.global_var


class NameSpaceWrapper:
    def __init__(self):
        self.ns = NameSpace.remote()

    def get_global_var(self):
        return ray.get(self.ns.get_global_var.remote())

    def set_global_var(self, var):
        self.ns.get_global_var.remote(var)


# ns = NameSpace.remote()
# ray.get(ns.get_global_var.remote())  # これを ns.get_global_var() にしたい

nsw = NameSpaceWrapper()
nsw.get_global_var()
