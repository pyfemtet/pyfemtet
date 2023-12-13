import ray
from win32com.client import Dispatch

# Initialize Ray
ray.init()

# Define a Sample class
class Sample:

    fem = None  # Class attribute to store PyIDispatch object

    def __init__(self):
        self.fem = Dispatch('excel.application')  # Your PyIDispatch object

        Sample.fem = None

    @ray.remote
    def my_task(sample_parallel_instance):
        local_fem = Sample.fem  # Use local variable to access fem without serialization
        # Perform computations using local_fem

        sample_parallel_instance.fem = Dispatch('excel.application')

        sample_parallel_instance.fem.This

    def main(self):
        self.my_task.remote(Sample())


if __name__ == '__main__':

    # Create an instance of Sample
    sample = Sample()

    sample.main()

