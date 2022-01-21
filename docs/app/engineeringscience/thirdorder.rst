.. _thirdorder:

thirdorder
===========

introduction
-------------

A Python script to help create input files for computing anhamonic interatomic force constants, harnessing the symmetries of the system to minimize the number of required DFT calculations. A second mode of operation allows the user to build the third-order IFC matrix from the results of those runs.

how to use thirdorder
----------------------

.. code:: bash

   singularity shell /lustre/opt/contribute/cascadelake/thirdorder/thirdorder.sif
   python /work/thirdorder/thirdorder_espresso.py scf.in

result
-----------------

.. code:: bash

   Singularity> python /work/thirdorder/thirdorder_espresso.py scf.in
   Usage:
   /work/thirdorder/thirdorder_espresso.py unitcell.in sow na nb nc cutoff[nm/-integer] supercell_template.in
   /work/thirdorder/thirdorder_espresso.py unitcell.in reap na nb nc cutoff[nm/-integer]
   Singularity>

reference
----------

-  `thirdorder website <https://bitbucket.org/sousaw/thirdorder/src/master/>`__
