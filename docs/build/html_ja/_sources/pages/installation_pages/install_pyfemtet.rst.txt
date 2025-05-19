Install PyFemtet
================

This step requires administrator privileges.

.. note::
    
    This procedure requires Python 3.11 or 3.12. If you are installing
    PyFemtet on a different version of Python, follow the manual
    instructions at the end of this page.


Please download and run the script below. 

:download:`pyfemtet-installer.ps1 <../../pyfemtet-installer.ps1>` 

To run the .ps1 file, right click and select "run with powershell".

.. note::

    This script will show some dialogs even if the installation fails.
    If a command prompt window disappears immediately and no dialog shown,
    you may fail to run .ps1 script itself.

    In such case, the following step and command may help to launch .ps1
    script correctly.

    1. Press Windows key, enter `cmd` and `Run as administrator`.

    2. Run the following command::

        powershell -ExecutionPolicy ByPass <path\to\downloaded\pyfemtet-installer.ps1>

    ***Security Warnings***

    This command temporarily relaxes the security requirements to run powershell script.
    Before executing this command, please ensure that the .ps1 file you specifying
    is the one downloaded from the above source.


.. warning::

    This script assumes that your Python installation includes `py launcher`.
    If you have done a custom installation of Python, **make sure that
    your installation configuration includes the py launcher.**


.. tip::

    If you want to configure pyfemtet in a virtual environment (or anyway without `py launcher`),
    **change line 24 of the script** to the following, activate the
    virtual environment, and run the script from the command line: ::

        $python_command = "python"


If this step fails, try the following steps manually:

.. toctree::
    :titlesonly:

    install_pyfemtet_manually
    setup_femtet_macro
    setup_com_constants
