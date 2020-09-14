
TensorFlow
==========

简介
----

TensorFlow
是一个端到端开源机器学习平台。它拥有一个包含各种工具、库和社区资源的全面灵活生态系统，可以让研究人员推动机器学习领域的先进技术的发展，并让开发者轻松地构建和部署由机器学习提供支持的应用。

Pi 上的 TensorFlow
------------------

Pi 上可以用 miniconda 自行安装 TensorFlow，也可以用已预置的 singularity
``tensorflow-2.0.0`` 优化镜像。

使用 miniconda 安装 TensorFlow
------------------------------

创建名为 ``tf-env`` 的虚拟环境，激活虚拟环境，然后安装 TensorFlow

.. code:: bash

   $ module load miniconda3
   $ conda create -n tf-env
   $ source activate tf-env
   $ conda install pip
   $ conda install cudatoolkit=10.1 cudnn
   $ pip install tensorflow
   # 如需使用，可以选择安装keras
   $ pip install keras

提交 TensorFlow 作业
--------------------

示例：在 DGX-2 上使用 TensorFlow。作业使用单节点，分配 2 块 GPU，GPU:CPU
配比 1:6

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --mem=MaxMemPerNode
   #SBATCH --gres=gpu:2

   module load miniconda3
   source activate tf-env

   python -c 'import tensorflow as tf; \
              print(tf.__version__);   \
              print(tf.test.is_gpu_available());'

假设这个脚本文件名为
``tensorflow_conda.slurm``\ ，使用以下指令提交作业：

.. code:: bash

   $ sbatch tensorflow_conda.slurm

使用 singularity 容器中的 TensorFlow
------------------------------------

集群中已经预置了 `NVIDIA GPU CLOUD <https://ngc.nvidia.com/>`__
提供的优化镜像，通过调用该镜像即可运行 TensorFlow
作业，无需单独安装，目前版本为 ``tensorflow-2.0.0``\ 。该容器文件位于
``/lustre/share/img/tensorflow-2.0.0.simg``

使用 singularity 容器提交 TensorFlow 作业
-----------------------------------------

示例：在 DGX-2 上使用 TensorFlow 的容器。作业使用单节点并分配 2 块 GPU：

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --mem=MaxMemPerNode
   #SBATCH --gres=gpu:2

   IMAGE_PATH=/lustre/share/img/tensorflow-2.0.0.simg

   singularity run --nv $IMAGE_PATH python -c 'import tensorflow as tf; \
                                               print(tf.__version__);   \
                                               print(tf.test.is_gpu_available());'

假设这个脚本文件名为
``tensorflow_singularity.slurm``\ ，使用以下指令提交作业

.. code:: bash

   $ sbatch tensorflow_singularity.slurm

参考资料
--------

-  `TensorFlow 官网 <https://www.tensorflow.org/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__
-  `Singularity文档 <https://sylabs.io/guides/3.5/user-guide/>`__
