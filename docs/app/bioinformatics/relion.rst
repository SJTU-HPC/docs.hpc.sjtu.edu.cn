.. _relion:

Relion
======

简介
----

Relion 是由 MRC 的 Scheres 在 2012 年发布的针对单颗粒冷冻电镜图片进行处理的框架。

π 集群上的 Relion
---------------------

查看 π 集群上已编译的 GPU 版软件:

.. code:: bash

   $ module av relion

调用该模块及相关依赖:

.. code:: bash

   $ module load relion/3.0.8 或 $ module load relion/3.1.3-gcc-8.3.0-openmpi
   $ module load ghostscript/9.54.0-gcc-8.3.0
   $ module load openmpi/4.1.1-gcc-9.3.0
   $ module load cuda/10.2.89-intel-19.0.4

使用 GPU 版本的 Relion
----------------------

在 dgx2 队列上使用 1 块 gpu，并配比 6 cpu 核心。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=6
   #SBATCH --cpus-per-task=1
   #SBATCH --gres=gpu:1

   module load relion/3.0.8
   srun --mpi=pmi2 relion_refine_mpi (relion 的命令...)

使用以下指令提交作业

.. code:: bash

   $ sbatch slurm.test

使用 HPC Studio 启动可视化界面
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

参照\ `可视化平台 <../../../login/HpcStudio/>`__\ ，登录 HPC Studio，在顶栏选择 Relion：

|avater| |image1|

Relion的作业模板设置
~~~~~~~~~~~~~~~~~~~~

cpu队列 ``small_slurm.bash``

.. code:: bash

   #!/bin/bash
   #SBATCH --partition=small
   # set the job name
   #SBATCH --job-name=relion_job
   # send output to
   #SBATCH --output=XXXoutfileXXX
   #SBATCH --error=XXXerrfileXXX
   # this job requests x nodes
   #SBATCH --nodes=1
   # this job requests x tasks per node
   #SBATCH --ntasks-per-node=XXXmpinodesXXX
   #SBATCH --export=ALL
   #paste the "print command" from relion here
   mpirun -n XXXmpinodesXXX --mca mpi_cuda_support 0 XXXcommandXXX

gpu队列 ``xGPU_slurm.bash``

.. code:: bash

   #!/bin/bash
   #SBATCH --partition=dgx2
   # set the job name
   #SBATCH --job-name=relion_job
   # send output to
   #SBATCH --output=XXXoutfileXXX
   #SBATCH --error=XXXerrfileXXX
   # this job requests x nodes
   #SBATCH --nodes=1
   # this job requests x tasks per node
   #SBATCH --ntasks-per-node=XXXmpinodesXXX
   #SBATCH --cpus-per-task=6
   #SBATCH --time=48:00:00
   #SBATCH --gres=gpu:4
   #SBATCH --export=ALL
   #paste the "print command" from relion here
   mpirun -n XXXmpinodesXXX XXXcommandXXX

|image2|


参考资料
--------

-  `Relion 官网 <http://www2.mrc-lmb.cam.ac.uk/relion>`__
-  `Singularity 文档 <https://sylabs.io/guides/3.5/user-guide/>`__

.. |avater| image:: ../../img/relion2.png
.. |image1| image:: ../../img/relion1.png
.. |image2| image:: ../../img/relion3.png
