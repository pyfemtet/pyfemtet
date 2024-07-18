Procedure for executing optimization
----------------------------------------

This page demonstrates how to create a program for conducting optimal design using ``pyfemtet.opt`` in your own project.


1. Creating a Femtet project

    Create an analysis model on Femtet. **Register the parameters you want to optimize as variables.** For more details on analysis settings using parameters, please refer to Femtet Help / Project Creation / Variables.


2. Setting the objective function

    In optimization problems, the metric to be evaluated is referred to as the objective function. Please write the process of calculating the objective function from analysis results or model shapes using Python macros in Femtet.


    .. code-block:: python

        """Example to calculate max displacement (for your obejctive function).
        The scripts after Dispatch are Femtet's Python macros.
        """
        from win32com.client import Dispatch

        # Get object to control Femtet.
        Femtet = Dispatch("FemtetMacro.Femtet")

        # Open analysis result by Femtet.
        Femtet.OpenCurrentResult(True)
        Gogh = Femtet.Gogh

        # ex.) Get max displacement from analysis deresult.
        dx, dy, dz = Gogh.Galileo.GetMaxDisplacement()

    .. note::
        For the Python macro syntax in Femtet, please refer to the Femtet Macro Help or `Macro Examples <https://www.muratasoftware.com/support/macro/>`_.
    

3. Creating the main script

    Using the design variables and objective function defined above, create the main script.

    .. code-block:: python

        """The minimum code example to execute parameter optimization using PyFemtet."""

        from pyfemtet.opt import FEMOpt

        def max_displacement(Femtet):
            """Objective function"""
            Gogh = Femtet.Gogh
            dx, dy, dz = Gogh.Galileo.GetMaxDisplacement()
            return dy
            
        if __name__ == '__main__':
            # prepareing optimization object
            femopt = FEMOpt()

            # parameter setting
            femopt.add_parameter('w', 10, 2, 20)
            femopt.add_parameter('d', 10, 2, 20)

            # objective setting
            femopt.add_objective(max_displacement, direction=0)

            # run optimization
            femopt.optimize()

    .. note::
 
        For this script to actually work, you need a Femtet stress analysis project with variables ``w`` and ``d``.
 
 
    .. note::
 
        **The objective function must take a Femtet instance as the first argument,** since the ``FEMOpt`` instance intarcreates it internally.


    .. warning::
 
        Only perform ``add_parameter()`` on variables set with constant expressions within Femtet, and do not do it for variables set with string expressions. The string expressions will be lost.


4. Run the script.

    When the script is executed, the progress and results will be saved in a csv file. Each row in the csv file represents the result of one analysis attempt. The meaning of each column is as follows:

    ==================================  ============================================================================================================
      Columns                           Meaning
    ==================================  ============================================================================================================
    trial                               The number of the attempt
    <Variable name>                     The value of the variable specified in the script
    <Objective name>                    The calculation result of the objective function specified in the script
    <Objective name>_direction          The target of the objective function specified in the script
    <Constraint name>                   The calculation result of the constraint function specified in the script
    <Constraint name>_lb                The lower bound of the constraint function specified in the script
    <Constraint name>_ub                The upper bound of the constraint function specified in the script
    feasible                            Whether the attempt satisfies all constraints
    hypervolume                         The hypervolume up to that attempt (only when the objective function is 2 or more)
    message                             Special notes from the optimization process
    time                                Time when the attempt was completed
    ==================================  ============================================================================================================

    .. note:: Items enclosed in <> indicate that their content and number may vary depending on the script.

    .. note:: If the objective name and constraint name are not specified in the script, values such as obj_1, cns_1 will be automatically assigned.
