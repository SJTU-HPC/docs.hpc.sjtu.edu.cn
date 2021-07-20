.. _vasp:

VASP
====

简介
----

vasp全称Vienna Ab-initio Simulation Package，是维也纳大学Hafner小组开发的进行电子结构计算和量子力学-分子动力学模拟软件包。它是目前材料模拟和计算物质科学研究中最流行的商用软件之一。

ARM版VASP
---------

ARM平台上运行脚本如下(vasp.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 2            
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   source /lustre/share/singularity/commercial-app/vasp/activate arm

   mpirun -n $SLURM_NTASKS vasp_std

使用如下指令提交：

.. code:: bash

   $ sbatch vasp.slurm

ARM平台上使用容器运行VASP
-------------------------

将共享目录下的vasp镜像拷贝到自己的用户目录下

.. code:: bash

   $ cp /lustre/share/singularity/aarch64/vasp/vasp.sif /your/own/vasp/img/

运行脚本如下(vasp.slurm):

.. code:: bash

   #!/bin/bash
   #SBATCH -J vasp_arm_singularity
   #SBATCH -p arm128c256g
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=60
   #SBATCH --exclusive
   module load openmpi/4.0.3-gcc-9.3.0-vasp
   VASP_PATH=/your/own/vasp 
   IMAGE_PATH=$VASP_PATH/img
   mpirun -np $SLURM_NTASKS -x OMP_NUM_THREADS=1 singularity exec -B $VASP_PATH/data:/mnt -B /lib64:/lib64 $IMAGE_PATH /mnt/./run.sh

vasp.slurm脚本中run.sh中的内容如下所示：

.. code:: bash

   #!/bin/bash
   vasp_std
