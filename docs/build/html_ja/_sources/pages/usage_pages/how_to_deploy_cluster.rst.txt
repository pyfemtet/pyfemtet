.. |dask| raw:: html

    <a href="https://docs.dask.org/en/stable/deploying.html" target="_blank">dask documentation</a>


Procedure for Running Cluster Calculations (Experimental Feature)
----------------------------------------------------------------------------

This page outlines the procedure for parallel computing an optimization program using ``pyfemtet.opt`` on multiple PCs.

.. note::
    
    **Here, the machine where the program is called is referred to as the 'local PC,' and the machine running the calculations is referred to as the 'calculation PC.'**
    It is acceptable to have multiple calculation PCs.
    The local machine can also be a calculation machine.
    Please perform '2. Setting Up Calculation PC' and '4. Launching the Worker' for each calculation PC.


.. tip::
    
    Parallel computing in pyfemtet depends on ``dask.distributed``. This document describes the behavior as of dask version 2023.12.1. For more details and the latest CLI command usage, please refer to |dask|.


1. Creating a Program

    Refer to :doc:`how_to_optimize_your_project` and create a program for optimization.


2. Setting Up Calculation PC

    - Please install Femtet on the calculation PC.
    - Please install the same version of Python as on the local PC on the calculation PC.
    - Please install the same version of pyfemtet and its dependencies as on the local PC on the calculation PC.

        - To install dependencies with specified versions, the following steps are convenient. Please execute the following steps from the command prompt. (Please do not execute the line starts with # as it is a comment.)

        .. code-block::

            # local PC
            py -m pip freeze > requirements.txt

        Transfer the file generated here, named requirements.txt, to the calculation PCs, and run the following command in the command prompt.

        .. code-block::
            
            # calculation PC            
            py -m pip install -r <path/to/requirements.txt>

        Then run the makepy command to set the macro constants for Femtet.

        .. code-block::
            
            # calculation PC            
            py -m win32com.client.makepy FemtetMacro


3. Launching the Scheduler (a process that manages processes on multiple calculation PCs)

    - Please run the following command on your local PC.

        .. code-block::

            # local PC
            dask scheduler 

        .. figure:: images/dask_scheduler.png

            Please make a note of the numbers displayed here, such as tcp://~~~:~~~.

        .. note::

            | If communication ports are restricted due to firewalls or other constraints,
            | ``dask scheduler --port your_port``
            | please use the above command (replace your_port with the port number).


4. Launching the Worker (a process that performs calculations)

    - Please run the following command on the calculation PCs.

        .. code-block::

            # calculation PC
            dask worker tcp://~~~:~~~ --nthreads 1 --nworkers -1

        If the screen updates on both scheduler and worker, and the text ``Starting established connection`` is displayed, the communication has been successful.

        .. note:: If communication is not possible for a certain period of time, a message indicating a timeout will be displayed on the Worker side.
        

5. Editing and executing programs

    - Include the address of the Scheduler in the program so that computational tasks are passed to the Scheduler during program execution.
    - Specify ``tcp://~~~:~~~`` for the argument ``scheduler_address`` in the FEMOpt constructor.

        .. code-block:: Python

            from pyfemtet.opt import FEMOpt

            ...  # Define objectives, constraints and so on.

            if __name__ == '__main__':

                femopt = FEMOpt(scheduler_address='tcp://~~~:~~~')

                ...  # Setup optimization problem.
        
                femopt.optimize()  # Connect cluster and start optimization
                femopt.terminate_all()  # terminate Shceduler and Workers started in procedure 3 and 4.


.. warning::

    If the program terminates abnormally due to errors, it is recommended to terminate the Scheduler and Worker once before retrying, and then proceed with steps 3 and 4 again.
