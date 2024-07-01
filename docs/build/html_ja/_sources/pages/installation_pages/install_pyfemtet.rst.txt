Install PyFemtet
================

.. note::

    You will need administrator privileges to run this step.


Please download and run the script below. 

:download:`pyfemtet-installer.ps1 <../../pyfemtet-installer.ps1>` 

To run the .ps1 file, right click and select "run with powershell".


.. warning::

    This script assumes that your Python installation includes `py launcher`.


.. tip::

    If you are setting up pyfemtet in a virtual environment,
    change line 20 of the script to `$python_command = "python"`
    and run the script from the command line with your virtual environment active.


If this step fails, try the following steps manually:

.. toctree::
    :titlesonly:

    install_pyfemtet_manually
    setup_femtet_macro
    setup_com_constants
