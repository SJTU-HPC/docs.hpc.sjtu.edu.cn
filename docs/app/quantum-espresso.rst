#


Quantum ESPRESSO



--------------

简介
----

Quantum ESPRESSO is an integrated suite of computer codes for
electronic-structure calculations and materials modeling at the
nanoscale. It is based on density-functional theory, plane waves, and
pseudopotentials (both norm-conserving and ultrasoft).

Quantum ESPRESSO stands for opEn Source Package for Research in
Electronic Structure, Simulation, and Optimization. It is freely
available to researchers around the world under the terms of the GNU
General Public License.

Pi 上的 Quantum ESPRESSO
------------------------

查看 Pi 上已编译的软件模块:

.. code:: bash

   $ module avail espresso

调用该模块:

.. code:: bash

   $ module load quantum-espresso/6.5-intel-19.0.5-impi

Pi 上的 Slurm 脚本 slurm.test
-----------------------------

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH -J QE_test
   #SBATCH -p cpu
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   ulimit -s unlimited
   ulimit -l unlimited

   export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
   export I_MPI_FABRICS=shm:ofi

   module load quantum-espresso/6.5-intel-19.0.5-impi

   srun pw.x -i test.in

Pi 上提交作业
-------------

.. code:: bash

   $ sbatch slurm.test

参考资料
--------

-  `Quantum ESPRESSO 官网 <https://www.quantum-espresso.org/>`__
