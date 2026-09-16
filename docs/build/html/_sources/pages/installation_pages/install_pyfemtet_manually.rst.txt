Manually Install PyFemtet
=========================

This page shows you how to install/update
``pyfemtet`` manually.


Launch command prompt
---------------------

.. figure:: launch_cmd.png


Input install command
---------------------

Run following command to install or update.
The both are the same command.

    (Only PyFemtet core library:)::

        py -m pip install pyfemtet -U --no-warn-script-location


    (With GUI tool to build script:)::

        py -m pip install pyfemtet pyfemtet-opt-gui -U --no-warn-script-location


.. figure:: pip_on_cmd.png


.. note::

    If you failed to this process
    and the warning or error message constaints
    ``ConnectionError``, ``EnvironmentError`` and so on,
    the proxy may be a problem.

    Please try to set the environment variable
    before running the install command.
    For example, ::

        set HTTP_PROXY=http://<user>:<password>@<host>:<port>
        set HTTPS_PROXY=http://<user>:<password>@<host>:<port>
        py -m pip install pyfemtet -U --no-warn-script-location


.. note::

    PyFemtet core library is published under MIT,
    but the GUI tool is published under LGPL-v3.


Wait for installation
---------------------

- Depending on the environment, the following installation screen may be displayed and may not change for several minutes or even several tens of minutes.

.. figure:: pip_while_install.png

    Installation Screen


- Wait until the "Installed successfully..." message appears on the screen.

.. figure:: pip_complete_install.png    


If you want to check the installation, please check the section below.

- :ref:`Check the installation of PyFemtet <check-the-installation-pyfemtet-section>`
