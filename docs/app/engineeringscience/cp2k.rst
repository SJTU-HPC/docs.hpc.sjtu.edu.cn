.. _cp2k:

CP2k
====

CP2K是一个量子化学和固态物理软件包，可以对固态，液态，分子，周期性，材料，
晶体和生物系统进行原子模拟。CP2K为不同的建模方法提供了通用框架。支持的理论
水平包括DFTB，LDA，GGA，MP2，RPA，半经验方法（AM1，PM3，PM6，RM1，
MNDO等）和经典力场（AMBER，CHARMM等）。CP2K可以使用NEB或二聚体方法
进行分子动力学，元动力学，蒙特卡洛，埃伦菲斯特动力学，振动分析，核心能谱，
能量最小化和过渡态优化的模拟。

更多信息请访问：https://www.cp2k.org/

语言：Fortran 2008

开源协议：GPL

一句话描述：是一个量子化学和固态物理软件包，可以对固态，液态，分子，周期
性，材料，晶体和生物系统进行原子模拟。

π 集群上的CP2K
-----------------

π 集群系统中已经预装 CP2K (GNU+cpu 版本)，可用以下命令查看和加载:

.. code:: bash

   $ module av cp2k
   $ module load cp2k/8.2-gcc-9.2.0-openblas   # CPU版本
   $ module load cp2k/8.2-gcc-8.3.0-openblas   # GPU版本


若不指定版本，将采用默认的 module（标记为 D）

π 集群上CPU版本cp2k示例脚本cp2k_cpu.slurm
------------------------------------------------

在 cpu 队列上，总共使用 40 核 (n = 40) 
cpu 队列每个节点配有 40核，所以这里使用了 1 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=cp2k_cpu_test
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load cp2k/8.2-gcc-9.2.0-openblas 
   module load openmpi/4.0.5-gcc-9.2.0

   ulimit -s unlimited
   ulimit -l unlimited

   INPUT_FILE=H2O-256.inp
   mpirun --allow-run-as-root -np $SLURM_NTASKS -x OMP_NUM_THREADS=1 cp2k.psmp ${INPUT_FILE}


π 集群上GPU版本cp2k示例脚本cp2k_gpu.slurm
--------------------------------------------

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=cp2k_gpu_test
   #SBATCH --partition=dgx2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=6
   #SBATCH --cpus-per-task=1
   #SBATCH --gres=gpu:1


   module load cp2k/8.2-gcc-8.3.0-openblas
   module load cuda/10.1.243-gcc-8.3.0
   module load openmpi/4.0.5-gcc-8.3.0

   ulimit -s unlimited
   ulimit -l unlimited

   INPUT_FILE=H2O-256.inp
   mpirun --allow-run-as-root -np $SLURM_NTASKS --mca opal_common_ucx_opal_mem_hooks 1 -x OMP_NUM_THREADS=1 cp2k.psmp ${INPUT_FILE}


并使用如下指令提交：

.. code:: bash

   $ sbatch cp2k_cpu.slurm
   $ sbatch cp2k_gpu.slurm


ARM集群上的cp2k
-------------------

ARM集群中已经预装了CP2K，可在 `ARM 节点 <../login/index.html#arm>`__\ 使用如下命令查看和加载：

.. code:: bash

   module av cp2k
   module load cp2k/4.1-cp2k-gcc-8.3.0
   module load cp2k/8.2-gcc-9.3.0-openblas-openmpi

ARM集群上Slurm脚本 cp2k.slurm
---------------------------------

示例脚本如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test
   #SBATCH --partition=arm128c256g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load cp2k/8.2-gcc-9.3.0-openblas-openmpi
   module load openmpi/4.0.3-gcc-9.3.0

   ulimit -s unlimited
   ulimit -l unlimited

   INPUT_FILE=H2O-256.inp
   mpirun --allow-run-as-root -np $SLURM_NTASKS -x OMP_NUM_THREADS=1 cp2k.psmp ${INPUT_FILE} 

在 `ARM 节点 <../login/index.html#arm>`__\ 使用如下命令提交作业：

.. code:: bash

   sbatch cp2k.slurm


参考资料
--------

-  `CP2K 官网 <https://manual.cp2k.org/#gsc.tab=0>`__
