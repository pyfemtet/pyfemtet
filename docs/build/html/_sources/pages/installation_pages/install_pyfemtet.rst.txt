Install PyFemtet
================

This step requires administrator privileges.

.. note::
    
    This procedure requires Python 3.11, 3.12 or 3.13. If you are installing
    PyFemtet on a different version of Python, follow the manual
    instructions at the end of this page.


1. Please download the following file.

    :download:`pyfemtet-installer.ps1 <../../pyfemtet-installer.ps1>`

2. Run the following command on **administrative** ``Command Prompt``. ::

    powershell -ExecutionPolicy Bypass -File <path/to/your/downloaded/.ps1>


.. note::

    Please don't forget to replace ``<path/to/your/downloaded/.ps1>`` to
    the path of the ``pyfemtet-installer.ps1`` file you downloaded.
    You can acquire the path by **shift+right-clicking** the file and
    select "Copy as a path".


.. warning::

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


.. _check-the-installation-pyfemtet-section:

Check the Installation of PyFemtet
----------------------------------

If you want to check the installation later, please follow the steps below.

1. Press the Windows key and open the Command Prompt.

    .. figure:: launch_cmd.png

2. Type ``py -m pip show pyfemtet`` and press Enter.
   (``py -m pip show pyfemtet-opt-gui`` for GUI tool)

3. If you get the following result, the setup was successful.
   (
   If you see the message
   ``'py' is not recognized as an internal or external command, operable program or batch file.``,
   Python is not installed.
   
   If you see the message ``WARNING: Package(s) not found: pyfemtet`` (or ``pyfemtet-opt-gui``),
   PyFemtet (or its GUI tool) is not installed.
   )

    .. figure:: pyfemtet_installed.png


Manual Installation
-------------------

If this step fails, try the following steps manually:

.. toctree::
    :titlesonly:

    install_pyfemtet_manually
    setup_femtet_macro
    setup_com_constants
