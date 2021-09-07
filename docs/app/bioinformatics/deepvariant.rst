DeepVariant
===========

DeepVariant is an analysis pipeline that uses a deep neural network to
call genetic variants from next-generation DNA sequencing data.
DeepVariant relies on Nucleus, a library of Python and C++ code for
reading and writing data in common genomics file formats (like SAM and
VCF) designed for painless integration with the TensorFlow machine
learning framework.

CPU 版本的 Singularity DeepVariant
----------------------------------

CPU版安装
^^^^^^^^^

申请计算节点，然后制作 singularity 镜像

.. code:: bash

   $ srun -p cpu -N 1 --exclusive --pty /bin/bash
   $ singularity build deepvariant.simg docker://google/deepvariant

用SLURM脚本提交CPU版DeepVariant作业
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 CPU 版本的 singluarity 镜像的 slurm.sh 如下：

.. code:: bash

   #!/bin/bash

   #SBATCH -J DeepVariant
   #SBATCH -p small
   #SBATCH -n 1
   #SBATCH --ntasks-per-node=1
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   ulimit -s unlimited
   ulimit -l unlimited

   IMAGE_PATH=/安装路径/deepvariant.simg

   singularity run $IMAGE_PATH /opt/deepvariant/bin/make_examples 

并使用如下指令提交：

.. code:: bash

   $ sbatch slurm.sh

交互式提交CPU版DeepVariant作业
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   srun -p cpu -N 1 --exclusive --pty /bin/bash
   export IMAGE_PATH=/安装路径/deepvariant.simg
   singularity run $IMAGE_PATH /opt/deepvariant/bin/make_examples

GPU 版本的 Singularity DeepVariant
----------------------------------

GPU版安装
^^^^^^^^^

申请计算节点，然后制作 singularity 镜像

.. code:: bash

   $ srun -p cpu -N 1 --exclusive --pty /bin/bash
   $ singularity build deepvariant.gpu.simg docker://google/deepvariant:0.10.0-gpu

用SLURM脚本提交GPU版作业
^^^^^^^^^^^^^^^^^^^^^^^^

使用GPU版本的 singluarity 镜像的 slurm.sh 如下：

.. code:: bash

   #!/bin/bash

   #SBATCH -J DeepVariant
   #SBATCH -p dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=6
   #SBATCH --gres=gpu:1
   #SBATCH --mem=MaxMemPerNode
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   ulimit -s unlimited
   ulimit -l unlimited

   IMAGE_PATH=/安装路径/deepvariant.gpu.simg

   singularity run $IMAGE_PATH /opt/deepvariant/bin/make_examples 

并使用如下指令提交：

.. code:: bash

   $ sbatch slurm.sh

交互式提交GPU版deepvarant作业
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   srun --ntasks-per-node=1 -p dgx2 --gres=gpu:1 -N 1 --pty /bin/bash
   export IMAGE_PATH=/安装路径/deepvariant.gpu.simg
   singularity run $IMAGE_PATH /opt/deepvariant/bin/make_examples

参考资料
--------

-  DeepVariant官网 https://github.com/google/deepvariant

