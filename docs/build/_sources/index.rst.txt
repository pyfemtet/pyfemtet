Welcome to PyFemtet's documentation!
====================================

.. |Femtet| raw:: html

    <a href="https://www.muratasoftware.com/" target="_blank">muratasoftware.com</a>

.. |Python| raw:: html

    <a href="https://www.python.org/" target="_blank">python.org</a>


Abstract
----------

**PyFemtet provides extensions for Femtet, a CAE software developed by Murata Software.**

- PyFemtet is an open-source library and can be used free of charge for both non-commercial and commercial purposes.
- This library is provided "as is" and without warranty of any kind.
- A license is required to use the Femtet main body. PyFemtet does not alter the license of the Femtet main body in any way.
- Please contact Murata Software for a trial version of Femtet for evaluation purposes.

    - --> |Femtet|


Main Features of PyFemtet
----------------------------

PyFemtet is a library that provides functionality using the Python macro interface of Femtet. Currently, the only feature of PyFemtet is design parameter optimization, which is implemented as a subpackage ``pyfemtet.opt``.

The optimization feature by ``pyfemtet.opt`` has the following characteristics:

- Single-objective and multi-objective optimization
- Real-time progress display with process monitoring
- Parallel computation with multiple instances of Femtet
- Result output in easy-to-analyze csv format for Excel and other tools


Examples
--------------------------------

.. grid:: 2

    .. grid-item-card:: Inductance of a solenoid coil
        :link: examples/gau_ex08/gau_ex08
        :link-type: doc
        :text-align: center

        .. image:: examples/gau_ex08/gau_ex08.png
            :scale: 50
        +++
        In magnetic field analysis, the self-inductance of a finite-length solenoid coil is set to a specific value.


    .. grid-item-card:: Resonant frequency of a circular patch antenna
        :link: examples/her_ex40/her_ex40
        :link-type: doc
        :text-align: center

        .. image:: examples/her_ex40/her_ex40.png
            :scale: 50
        +++
        In electromagnetic wave analysis, the resonant frequency of a circular patch antenna is set to a specific value.


.. tip::
    
    There are more examples in the :doc:`pages/examples` section.


Simple API
----------------------------

Below is an example of multi-objective optimization. You can set up the problem with ``add_parameter()`` and ``add_objective()``, and then execute it with ``optimize()``. For everything else, you can use the regular Femtet macro script. For more detailed examples, please check the :doc:`pages/usage` section.

.. code-block:: python

   from pyfemtet.opt import FEMOpt

   def max_displacement(Femtet):
       dx, dy, dz = Femtet.Gogh.Galileo.GetMaxDisplacement()
       return dy

   def volume(Femtet):
       w = Femtet.GetVariableValue('w')
       d = Femtet.GetVariableValue('d')
       h = Femtet.GetVariableValue('h')
       return w * d * h

   if __name__ == '__main__':
       femopt = FEMOpt()
       femopt.add_parameter('w', 10, 2, 20)
       femopt.add_parameter('d', 10, 2, 20)
       femopt.add_objective(max_displacement, name='max_displacement', direction=0)
       femopt.add_objective(volume, name='volume', direction='minimize')
       femopt.optimize(n_trials=20)


Install
---------------

.. note:: PyFemtet is only available for Windows.

.. note::
    
    In an environment where Python and Femtet are installed and Femtet macros are enabled, simply run ``pip install pyfemtet``. The following steps are for a full setup of Python, Femtet and PyFemtet.

1. **Installation of Femtet (version 2023.0 or later)**
    
    For first-time users, please consider using the trial version or personal edition. --> |Femtet|

    .. note::

        If you use Femtet that is not the latest version, some functions of PyFemtet cannot be used.

    
2. **Enabling Femtet macros**

    .. figure:: images/enableMacrosIcon.png


    .. note::

        Close Excel and Femtet before following this step.


    After installing Femtet, **please run 'EnableMacros' from the start menu.** This procedure requires administrator privileges.


    .. note::

        When you follow this step, the Femtet help window will appear. You do not need to follow the help window, so close it.


3. **Installation of 64bit Python (version 3.9.3 or later)**

    Download the installer from the link provided and run it.  --> |Python|

    .. tip::

        To download a version of Python that is not the latest, refer to the screenshot below and download the installer that suits your environment.

    .. tip::

        ``pyfemtet.opt`` is currently primarily developed in a Python 3.11 environment, so if you encounter any issues with installation or running the examples, consider setting up a Python 3.11 environment.

    .. figure:: images/python_download.png

    .. figure:: images/python_3.11.png
        :scale: 50%

        This screenshot shows an example of the location of the link to the installer for Python 3.11.7 for 64-bit Windows.

    .. figure:: images/python_install.png

        Installer screen.


4. **Installing** ``pyfemtet``

    Please launch the command prompt (``cmd``).

    .. figure:: images/launch_cmd.png

        Launch cmd via start button.


    Then run the following command in the command prompt. The download and installation of the library will begin.

    .. code-block::

        py -m pip install pyfemtet --no-warn-script-location


    .. figure:: images/pip_on_cmd.png

        Run pip command on command prompt.


    Once the installation is complete, after displaying "Successfully installed ", control will return to the command prompt.

    .. figure:: images/pip_while_install.png

        Installing

    .. figure:: images/pip_complete_install.png

        Installation completed

    .. note::

        Depending on the environment, installation may take about 5 minutes.

    .. note::

        At the end of the installation, you may see a message such as ``[notice] A new release of pip is available:`` . This is not an error and can be ignored without any issues.

5. **Setting Femtet Macro Constants**

    Please run the following command in the command prompt.::

        py -m win32com.client.makepy FemtetMacro

    Once the setting is complete, control will return to the command prompt.

    .. figure:: images/complete_makepy.png

        After makepy finishes

That's all.


.. tip::
    
    For verification, we recommend that you first view the samples in :doc:`pages/examples`.



Table of Contents
------------------------

.. toctree::
    :maxdepth: 2

    Home <self>
    pages/examples
    pages/usage
    pages/api
    pages/LICENSE
