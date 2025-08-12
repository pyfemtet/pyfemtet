r"""In optimization, the model geometry is updated by changing the design parameters.

At this time, depending on how the model is created, there is a known issue where the topology numbers inside the CAD can change, resulting in unintended assignments of boundary conditions and mesh sizes.

Femtet and PyFemtet have experimentally implemented a Topology Matching feature based on the technique described in [1].

This sample demonstrates optimization using Topology Matching to address the problem where boundary conditions would otherwise be lost with conventional methods.


# Limitations

This feature currently supports only models with a single body.


# Prerequests

1. Femtet 2025.0.2 or later is required.

2. Topology Matching の利用には追加のモジュールが必要です。
To use Topology Matching, you need to install an additional module.
Please install it using the following command:
(The MIT-licensed library ``brepmatching`` will be installed.)

    py -m pip install -U pyfemtet[matching]

    or

    py -m pip install -U brepmatching


3. Please complete the following steps as preparation:
- Install SOLIDWORKS.
- Create a C:\temp folder.
- Place the following files in the same folder:
    - topology_matching.py (this file)
    - cad_ex01_SW_fillet.SLDPRT
    - cad_ex01_SW_fillet.femprj


For more details about the SolidWorks integration feature, please refer to the following page:
https://pyfemtet.readthedocs.io/en/stable/examples/Sldworks_ex01/Sldworks_ex01.html

[1]
Benjamin Jones, James Noeckel, Milin Kodnongbua, Ilya Baran, and Adriana Schulz. 2023.
B-rep Matching for Collaborating Across CAD Systems.
ACM Trans. Graph. 42, 4, Article 104 (August 2023), 13 pages.
https://doi.org/10.1145/3592125

"""
