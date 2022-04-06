.. _tensorflow:

TensorFlow
==========

简介
----

TensorFlow
是一个端到端开源机器学习平台。它拥有一个包含各种工具、库和社区资源的全面灵活生态系统，可以让研究人员推动机器学习领域的先进技术的发展，并让开发者轻松地构建和部署由机器学习提供支持的应用。

π 超算上的 TensorFlow
----------------------------

π 超算上可以用 miniconda 自行安装 TensorFlow，也可以用已预置的 Singularity
``tensorflow-2.2.0`` 优化镜像。该镜像在π 超算及思源一号上均有部署。

+----------+----------------+----------+------------------------------------------------------+
|版本      |平台            |构建方式  |名称                                                  |
+==========+================+==========+======================================================+
| 2.2.0    |  |gpu|         | 容器     |/lustre/share/singularity/modules/tensorflow/2.2.0.sif|
+----------+----------------+----------+------------------------------------------------------+
| 2.2.0    |  |gpu|         | 容器     |/dssg/share/imgs/tensorflow/2.2.0.sif思源平台         |
+----------+----------------+----------+------------------------------------------------------+


使用 miniconda 安装 TensorFlow
------------------------------

创建名为 ``tf-env`` 的虚拟环境，激活虚拟环境，然后安装 TensorFlow

.. code:: bash

   $ module load miniconda3
   $ conda create -n tf-env python=3.6
   $ source activate tf-env
   $ conda install pip
   $ conda install cudatoolkit=10.1 cudnn
   $ pip install tensorflow=2.3.1
   # 如需使用，可以选择安装keras
   $ pip install keras

提交 TensorFlow 作业
--------------------

在 DGX-2 上使用 TensorFlow。作业使用单节点，分配 2 块 GPU，GPU:CPU
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


在 A100 上使用 TensorFlow。作业使用单节点，分配 2 块 GPU，GPU:CPU
配比 1:6

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p a100
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
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


使用 π 提供的 TensorFlow
-------------------------

集群中已经预置了 `NVIDIA GPU CLOUD <https://ngc.nvidia.com/>`__
提供的优化镜像，通过调用该镜像即可运行 TensorFlow 作业，无需单独安装，目前版本为 ``tensorflow-2.2.0``\ 。

查看π 超算上已编译的软件模块:

.. code:: bash

   module av tensorflow

调用该模块:

.. code:: bash

   module load tensorflow/2.2.0

以下 Slurm 脚本，在 dgx2 队列上使用 2 块 gpu，并配比 12 cpu 核心。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --gres=gpu:2

   module load tensorflow/2.2.0

   python -c 'import tensorflow as tf; \
              print(tf.__version__);   \
              print(tf.test.is_gpu_available());'

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test


查看思源一号上已编译的软件模块:

.. code:: bash
   
   module use /dssg/share/imgs
   module av tensorflow

调用该模块:

.. code:: bash
   
   module use /dssg/share/imgs
   module load tensorflow/2.2.0

以下 Slurm 脚本，在 A100 队列上使用 2 块 gpu，并配比 12 cpu 核心。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p a100
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --gres=gpu:2

   module use /dssg/share/imgs
   module load tensorflow/2.2.0

   python -c 'import tensorflow as tf; \
              print(tf.__version__);   \
              print(tf.test.is_gpu_available());'

使用如下指令提交：

.. code:: console

   $ sbatch slurm.test



算例测试
-------------------
超算中心提供了用来测试tensorflow的算例。

使用命令

.. code:: console

   $ cd ~
   $ git clone https://github.com/SJTU-HPC/HPCTesting.git


在π 超算上，使用如下脚本来提交该算例作业：

.. code:: bash

   #!/bin/bash
   #SBATCH -p dgx2
   #SBATCH -N 1
   #SBATCH -n 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --cpus-per-task 6 
   #SBATCH --gres gpu:1

   cd ~/HPCTesting/tensorflow/case1
   singularity  run --nv   /lustre/share/singularity/modules/tensorflow/2.2.0.sif python tf_test.py


在思源一号上，使用如下脚本来提交该算例作业：

.. code:: bash

   #!/bin/bash
   #SBATCH -p a100
   #SBATCH -N 1
   #SBATCH -n 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --cpus-per-task 6 
   #SBATCH --gres gpu:1

   cd ~/HPCTesting/tensorflow/case1
   singularity  run --nv /dssg/share/imgs/tensorflow/2.2.0.sif python tf_test.py

将以上脚本保存为 ``test.slurm`` ，使用 ``sbatch test.slurm`` 来交作业。

结果如下

.. code:: console

   Accuracy: mean=98.653 std=0.106, n=5

参考资料
--------

-  `TensorFlow 官网 <https://www.tensorflow.org/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__
