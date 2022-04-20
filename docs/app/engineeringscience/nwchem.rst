.. _nwchem:

NWChem
======

简介
----

NWChem provides many methods for computing the properties of molecular and periodic systems using standard quantum mechanical descriptions of the electronic wavefunction or density. Its classical molecular dynamics capabilities provide for the simulation of macromolecules and solutions, including the computation of free energies using a variety of force fields. These approaches may be combined to perform mixed quantum-mechanics and molecular-mechanics simulations.

NWChem software can handle:

.. hlist::
   :columns: 2

   - Biomolecules, nanostructures, and solid-state
   - From quantum to classical, and all combinations
   - Ground and excited-states
   - Gaussian basis functions or plane-waves
   - Scaling from one to thousands of processors
   - Properties and relativistic effects

可用的版本
----------

+--------+---------+----------+-----------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                                    |
+========+=========+==========+===========================================================+
| 6.8.1  | |cpu|   | Spack    | nwchem/6.8.1-intel-19.0.4-impi                            |
+--------+---------+----------+-----------------------------------------------------------+

算例下载
--------

.. code-block:: bash

   wget https://nwchemgit.github.io/c240_631gs.nw

编辑文件，:abbr:`删除 (sed -i '6d' c240_631gs.nw)` 第6行的 ``scratch_dir /scratch`` 

运行示例
--------

.. code-block:: bash
   :emphasize-lines: 3,7,15

   #!/bin/bash
   #SBATCH --job-name=nwchem
   #SBATCH --partition=small
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=10

   ulimit -l unlimited
   ulimit -s unlimited

   module load intel/19.0.4
   module load nwchem/6.8.1-intel-19.0.4-impi

   mpirun -np $SLURM_NTASKS nwchem c240_631gs.nw > c240_631gs.out

.. warning:: 
   
   软件运行需要库环境，加载时会自动创建 ``~/.nwchemrc`` 文件。

使用如下脚本提交作业

.. code:: bash

   sbatch test.slurm

运行结果见 ``c240_631gs.out``

运行时间
--------

**π2.0**

+---------------+---------------+---------------+
| nwchem/6.8.1-intel-19.0.4-impi                |
+===============+===============+===============+
| 核数          | 10            | 20            |
+---------------+---------------+---------------+
| CPU_time      | 1h50m37s      | 54m44s        |
+---------------+---------------+---------------+

参考资料
--------

-  `NWChem 文档 <https://nwchemgit.github.io/>`__
