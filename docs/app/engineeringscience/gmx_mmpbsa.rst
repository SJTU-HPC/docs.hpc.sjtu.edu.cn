.. _gmx_mmpbsa:

gmx_MMPBSA
===========

简介
----

gmx_MMPBSA是一种基于AMBER的MMPBSA.py开发的新工具，旨在使用GROMACS文件执行端态自由能计算。它与所有GROMACS版本以及AmberTools20或21一起使用，与现有程序相比，它在兼容性、多功能性、分析和并行化方面都有改进。在当前版本中，gmx_MMPBSA支持多种不同的系统，包括但不限于：

    蛋白质 

    蛋白质配体 

    蛋白质DNA 

    金属蛋白肽 

    蛋白聚糖 

    膜蛋白 

    多组分系统（例如，蛋白质DNA RNA离子配体）

π 集群上使用gmx_MMPBSA
-------------------------------

一定要将数据、执行脚本gmx_MMPBSA.slurm和run1.sh放在同一目录下。

CPU版gmx_MMPBSA(GROMACS2019)
-----------------------------

gmx_MMPBSA.slurm内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=gmx_MMPBSA       
   #SBATCH --partition=small  
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gmx_MMPBSA/1.4.3-gcc-4.8.5-ambertools-20

   mpirun -np $SLURM_NTASKS gmx_MMPBSA 

run1.sh脚本内容如下：

.. code:: bash

   #!/bin/bash

   gmx_MMPBSA MPI -O -i mmpbsa.in -cs com.tpr -ci index.ndx -cg 1 13 -ct com_traj.xtc

CPU版gmx_MMPBSA(GROMACS2021)
-----------------------------

gmx_MMPBSA.slurm内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=gmx_MMPBSA       
   #SBATCH --partition=cpu  
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=10
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gmx_MMPBSA

   mpirun -np $SLURM_NTASKS gmx_MMPBSA_9.3.0 

run1.sh脚本内容如下：

.. code:: bash

   #!/bin/bash

   gmx_MMPBSA MPI -O -i mmpbsa.in -cs com.tpr -ci index.ndx -cg 1 13 -ct com_traj.xtc

使用如下指令提交：

.. code:: bash

   $ sbatch gmx_MMPBSA.slurm

参考资料
--------

-  `gmx_MMPBSA 官网 <https://valdes-tresanco-ms.github.io/gmx_MMPBSA/>`__
