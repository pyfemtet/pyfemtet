Result Viewer
=============

``pyfemtet.opt`` has a GUI tool to visualize the optimization result.

.. figure:: images/pyfemtet-opt-result-viewer.png


Installation
------------

.. note:: 
    
    If you successfully ran `onestop-installer`, you already have the GUI tool.

    See :doc:`../installation_pages/install_pyfemtet` or :doc:`../installation_pages/install_pyfemtet_manually` .


.. tip::

    If you installed PyFemtet without using the installer,
    you can find the executable in the following location

    ```<Folder containing python.exe>\Scripts\pyfemtet-opt-result-viewer.exe```

    Please create a shortcut to this file. Do not copy or move it.


.. tip::

    The path to the folder containing python.exe can be found
    by opening the command prompt and executing the following command:

    ```py -c "import sys;print(sys.executable)"```

    If you have not installed the py launcher, please replace 'py' with 'python'.


.. note::

    If you installed PyFemtet using a virtual environment,
    you can use the viewer by executing the 'pyfemtet-opt-result-viewer'
    command in the command line.


Launch GUI
----------

Double click desktop icon.

.. figure:: images/pyfemtet-opt-result-viewer-desktop-icon.png


Usage
-----

Before launch tool, you run an optimization and get ``.csv`` file
that contains the optimization history.

Then Launch the tool and Femtet and load csv.

Finally, you can select each solution plot and
open corresponding FEM result in Femtet.

.. note::

    To learn usage with ready-made sample project, you can use `tutorial mode` in tool.
    It starts few minutes journy to explain how to use the tool.
