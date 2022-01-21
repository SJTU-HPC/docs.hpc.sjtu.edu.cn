.. _thirdorder:

thirdorder
===========

introduction
-------------

A Python script to help create input files for computing anhamonic interatomic force constants, harnessing the symmetries of the system to minimize the number of required DFT calculations. A second mode of operation allows the user to build the third-order IFC matrix from the results of those runs.

Use thirdorder in Singularity
-----------------------------

You can use thirdorder after entering image.

.. code:: bash

   singularity shell /lustre/opt/contribute/cascadelake/thirdorder/thirdorder.sif
   python /work/thirdorder/thirdorder_espresso.py scf.in

following results show that you can use thirdorder correctly.
-------------------------------------------------------------

.. code:: bash

   Singularity> python /work/thirdorder/thirdorder_espresso.py scf.in
   Usage:
   /work/thirdorder/thirdorder_espresso.py unitcell.in sow na nb nc cutoff[nm/-integer] supercell_template.in
   /work/thirdorder/thirdorder_espresso.py unitcell.in reap na nb nc cutoff[nm/-integer]
   Singularity>

Usage
-----

The directory will contain the compiled module thirdorder_core.so, thirdorder_common.py and DFT-code specific interfaces (e.g. thirdorder_vasp.py). All are needed to run thirdorder. You can either use them from that directory (maybe including it in your PATH for convenience) or copying the .py files to a directory in your PATH and thirdorder_core.so to any location where Python can find it for importing.

Running thirdorder with VASP
-----------------------------

Any invocation of thirdorder_vasp.py requires a POSCAR file with a description of the unit cell to be present in the current directory. The script uses no other configuration files, and takes exactly five mandatory command-line arguments:

.. code:: bash

   thirdorder_vasp.py sow|reap na nb nc cutoff[nm/-integer]

The first argument must be either "sow" or "reap", and chooses the operation to be performed (displacement generation or IFC matrix reconstruction). The next three must be positive integers, and specify the dimensions of the supercell to be created. Finally, the "cutoff" parameter decides on a force cutoff distance. Interactions between atoms spaced further than this parameter are neglected. If cutoff is a positive real number, it is interpreted as a distance in nm; on the other hand, if it is a negative integer -n, the maximum distance among n-th neighbors in the supercell is automatically determined and the cutoff distance is set accordingly.

The following POSCAR describes the relaxed geometry of the primitive unit cell of InAs, a III-V semiconductor with a zincblende structure:

.. code:: bash

   InAs
   6.00000000000000
     0.0000000000000000    0.5026468896190005    0.5026468896190005
     0.5026468896190005    0.0000000000000000    0.5026468896190005
     0.5026468896190005    0.5026468896190005    0.0000000000000000
   In   As
   1   1
   Direct
     0.0000000000000000  0.0000000000000000  0.0000000000000000
     0.2500000000000000  0.2500000000000000  0.2500000000000000

Let us assume that such POSCAR is in the current directory and that thirdorder_vasp.py is in our PATH. To generate an irreducible set of displacements for a 4x4x4 supercell and up-to-third-neighbor interactions, we run

.. code:: bash

   thirdorder_vasp.py sow 4 4 4 -3

This creates a file called 3RD.SPOSCAR with the undisplaced supercell coordinates and 144 files with names following the pattern 3RD.POSCAR.*. It is the latter that need to be input to VASP. This step is completely system-dependent, but suppose that in ~/vaspinputs we have the required INCAR, POTCAR and KPOINTS files as well as a runvasp.sh script that can be passed to qsub. We can run a command sequence like:

.. code:: bash

   for i in 3RD.POSCAR.*;do
      s=$(echo $i|cut -d"." -f3) &&
      d=job-$s &&
      mkdir $d &&
      cp $i $d/POSCAR &&
      cp ~/vaspinputs/INCAR ~/vaspinputs/POTCAR ~/vaspinputs/KPOINTS $d &&
      cp ~/vaspinputs/runvasp.sh $d &&
      (cd $d && qsub runvasp.sh)
   done

Some time later, after all these jobs have finished successfully, we only need to feed all the vasprun.xml files in the right order to thirdorder_vasp.py, this time in reap mode:

.. code:: bash

   find job* -name vasprun.xml|sort -n|thirdorder_vasp.py reap 4 4 4 -3

If everything goes according to plan, a FORCE_CONSTANTS_3RD file will be created at the end of this run. Naturally, it is important to choose the same parameters for the sow and reap steps.

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
