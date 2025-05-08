Use pyfemtet.opt with Microsoft Excel
=======================================================

This sample demonstrates how to use PyFemtet without
migrating the existing Excel macro processes to Python.


Sample Files
-------------------------------------------------------

.. note::
   Keep the :download:`UI file (Japanese only)<femtet-macro.xlsm>` and
   :download:`core script<pyfemtet-core.py>` on same
   folder.

.. note::
   :download:`here<(ref) original_project.femprj>` is a
   sample file to create UI file's macro base.


How to run this Sample
-------------------------------------------------------

.. warning::
   To run macros from an xlsm file downloaded from the
   internet, you need to change the security settings.

Open the Excel file and check the settings listed on
the "最適化の設定" sheet. Then, press the
"call pyfemtet" button at the bottom of the same sheet.

.. note::
   The macros included in this xlsm file are based on
   the macro auto-generation feature from the original
   Femtet project. Therefore, when you run the Excel
   macro, Femtet will launch and automatically create
   the original analysis model. Additionally, this
   macro has been modified to read variable values from
   cells within the xlsm file and reflect them in the
   created analysis model. For detailed specifications,
   please refer to the "備考" sheet in Excel and the
   comments in the source code.   


Design Variables
--------------------------------------------------------

.. figure:: tapered_inductor.png
   :width: 500
   
   Appearance of the Model

================= ==============================================
Variable Name     Description
================= ==============================================
section_radius    Radius of wire
coil_radius       Bottom radius of coil
coil_pitch        Pitch of coil
n                 Number of turns
coil_radius_grad  Coil radius increment per pitch
================= ==============================================


Objective Function
--------------------------------------------------------

- Self-inductance (aim to 100 nH)
- Approx. Volume (minimize)


Execution Result of the Samples
--------------------------------------------------------
.. figure:: running_excel_migration.png
   :width: 300

   Screenshot in progress 


The pyfemtet-core.py performs optimization by controlling
the values of various cells in the xlsm file and executing
macros instead of directly running Femtet. This allows for
integration with PyFemtet without needing to rewrite all
existing code if you are already using Excel macros for an
automated design flow with Femtet.
