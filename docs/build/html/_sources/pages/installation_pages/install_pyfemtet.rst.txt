Install PyFemtet
================

.. note::

    You will need administrator privileges to run this step.


Please download and run the script below. 

:download:`pyfemtet-installer.ps1 <../../pyfemtet-installer.ps1>` 

To run the .ps1 file, right click and select "run with powershell".


.. warning::

    This script assumes that your Python installation includes `py launcher`.
    If you have done a custom installation of Python, make sure that
    your installation configuration includes the `py launcher`.


.. tip::

    If you want to configure pyfemtet in a virtual environment,
    change line 20 of the script to the following, activate the
    virtual environment, and run the script from the command line: ::

        $python_command = "python"`


If this step fails, try the following steps manually:

.. toctree::
    :titlesonly:

    install_pyfemtet_manually
    setup_femtet_macro
    setup_com_constants
