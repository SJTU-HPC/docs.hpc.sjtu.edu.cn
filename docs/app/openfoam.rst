OpenFOAM
========

简介
----

OpenFOAM is a C++ toolbox for the development of customized numerical
solvers, and pre-/post-processing utilities for the solution of
continuum mechanics problems, most prominently including computational
fluid dynamics.

Pi 上的 OpenFOAM
----------------

查看 Pi 上已编译的软件模块:

.. code:: bash

   module av openfoam

调用该模块:

.. code:: bash

   module load openfoam/8

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点。脚本名称可设为 slurm.test

!!! example “cpu 队列 slurm 脚本示例 OpenFoam” \``\` #!/bin/bash

::

   #SBATCH --job-name=test           # 作业名
   #SBATCH --partition=cpu           # cpu 队列
   #SBATCH -n 80                     # 总核数 80
   #SBATCH --ntasks-per-node=40      # 每节点核数
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module purge
   module load openfoam/8

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 icoFoam -parallel
   ```

用下方语句提交作业

.. code:: bash

   sbatch slurm.test

参考链接
--------

-  `openfoam官方网站 <https://openfoam.org/>`__
-  `Singularity文档 <https://sylabs.io/guides/3.5/user-guide/>`__
