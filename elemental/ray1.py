import ray

ray.init()

@ray.remote
class GlobalVarActor:
    def __init__(self):
        self.global_var = 3

    def set_global_var(self, var):
        self.global_var = var

    def get_global_var(self):
        return self.global_var

@ray.remote
class Actor:
    def __init__(self, global_var_actor):
        self.global_var_actor = global_var_actor

    def f(self):
        return 3 + ray.get(self.global_var_actor.get_global_var.remote())


global_var_actor = GlobalVarActor.remote()
actor = Actor.remote(global_var_actor)
ray.get(actor.f.remote())
