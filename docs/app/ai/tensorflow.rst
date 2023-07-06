.. _tensorflow:

TensorFlow
==========

简介
------

TensorFlow
是一个端到端开源机器学习平台。它拥有一个包含各种工具、库和社区资源的全面灵活生态系统，可以让研究人员推动机器学习领域的先进技术的发展，并让开发者轻松地构建和部署由机器学习提供支持的应用。

测试数据位置
--------------

.. code:: bash
   
   思源一号:
   /dssg/share/sample/tensorflow/tf_test.py
   
   π2.0:
   /lustre/share/samples/tensorflow/tf_test.py

集群上的TensorFlow
--------------------


+-------+-------+----------+---------------------------------------+
| 版本  | 平台  | 构建方式 | 导入方式                              |
+=======+=======+==========+=======================================+
| 2.8.2 | |gpu| | 容器     | module load tensorflow/2.8.2 思源一号 |
+-------+-------+----------+---------------------------------------+
| 2.4.1 | |gpu| | 容器     | module load tensorflow/2.4.1 pi2.0    |
+-------+-------+----------+---------------------------------------+


TensorFlow使用教程
--------------------

- `思源一号 TensorFlow`_

- `π2.0 TensorFlow`_


.. _思源一号 TensorFlow:

一. 思源一号 TensorFlow
--------------------------

1.1 预编译版本的使用方式
~~~~~~~~~~~~~~~~~~~~~~~~~

思源上拷贝数据到本地
^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   cd
   mkdir tensorflow
   cd tensorflow
   cp /dssg/share/sample/tensorflow/tf_test.py ./

思源上TensorFlow提交作业脚本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p a100
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=32
   #SBATCH --gres=gpu:2

   module load tensorflow/2.8.2

   python tf_test.py   


1.2 思源一号上自定义构建TensorFlow 2.x环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 miniconda 安装 TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

创建名为 ``tf-env`` 的虚拟环境，激活虚拟环境，然后安装 TensorFlow

.. code:: bash

   cd
   mkdir tensorflow
   cd tensorflow
   cp /dssg/share/sample/tensorflow/tf_test.py ./
   module load miniconda3
   conda create -n tf-env python=3.8.5
   source activate tf-env
   conda install tensorflow=2.8.2=gpu_py38h75b8afa_0
   #上述命令会自动安装如下依赖
   .cudatoolkit=11.3.1=h2bc3f7f_2
   .cudnn=8.2.1=cuda11.3_0
   .numpy=1.23.4=py38h14f4228_0
   
   conda install matplotlib=3.5.0=py38h06a4308_0
   conda install -c conda-forge sklearn-quantile=0.0.18=py38h3ec907f_0

思源一号上作业提交脚本如下所示
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 A100 上使用 TensorFlow。作业使用单节点，分配 2 块 GPU，GPU:CPU
配比 1:16

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p a100
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=32
   #SBATCH --gres=gpu:2

   module load miniconda3
   source activate tf-env

   python tf_test.py

1.3 思源一号上自定义构建TensorFlow 1.x环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 pip 安装NVIDIA A100 GPU上优化的TensorFlow 1.x

.. code:: bash

   cd
   mkdir tensorflow
   cd tensorflow
   cp /dssg/share/sample/tensorflow/tf_test.py ./
   module load miniconda3
   conda create -n tf-env python=3.8.5
   source activate tf-env
   pip install --user nvidia-pyindex
   pip install --user nvidia-tensorflow[horovod]

.. _π2.0 TensorFlow:

二. π2.0 TensorFlow
-----------------------

2.1 π2.0上预编译版本的使用方式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

首先拷贝数据到本地
^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   cd
   mkdir tensorflow
   cd tensorflow
   cp /lustre/share/samples/tensorflow/tf_test.py ./

使用如下脚本提交TensorFlow作业
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=12
   #SBATCH --gres=gpu:2

   module load tensorflow/2.4.1
   python tf_test.py


2.2 π2.0上自定义安装TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

π2.0上使用 miniconda 安装 TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

创建名为 ``tf-env`` 的虚拟环境，激活虚拟环境，然后安装 TensorFlow

.. code:: bash

   cd
   mkdir tensorflow
   cd tensorflow
   cp /lustre/share/samples/tensorflow/tf_test.py ./
   module load miniconda3
   conda create -n tf-env python=3.8.5
   source activate tf-env
   conda install tensorflow=2.4.1=gpu_py38h8a7d6ce_0
   #上述命令会自动安装如下依赖
   .cudatoolkit=10.1.243h6bb024c_0
   .cudnn=7.6.5=cuda10.1_0
   .numpy=1.23.4=py38h14f4228_0
   
   conda install matplotlib=3.4.2=py38h06a4308_0
   conda install -c conda-forge sklearn-quantile=0.0.18=py38h3ec907f_0

作业提交脚本如下
^^^^^^^^^^^^^^^^^^^

在 DGX2 上使用 TensorFlow。作业使用单节点，分配 2 块 GPU，GPU:CPU
配比 1:6

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=12
   #SBATCH --gres=gpu:2
   
   module load miniconda3
   source activate tf-env
   python tf_test.py

TensorFlow的运行结果
----------------------

思源一号 TensorFlow
~~~~~~~~~~~~~~~~~~~~~~

思源上预编译版本的运行结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   Accuracy: mean=98.653 std=0.083, n=5

思源上自定义编译版本的运行结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: console

   Accuracy: mean=98.645 std=0.134, n=5

π2.0 TensorFlow
~~~~~~~~~~~~~~~~~~~~~~

预编译版本的运行结果
^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   Accuracy: mean=98.698 std=0.089, n=5 

自定义编译版本的运行结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   Accuracy: mean=98.638 std=0.159, n=5

参考资料
--------

-  `TensorFlow 官网 <https://www.tensorflow.org/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__
