.. _tvm:

Apache TVM
=============

简介
----

Apache TVM是一个开源的机器学习编译框架。旨在帮助机器学习工程师在任何硬件平台优化、运行计算任务。


π 超算上的 TVM
----------------------

π 超算上部署了预编译版本的TVM-0.9.dev0。

+-------------+----------------+----------+---------------------------------------------------+
|版本         |平台            |构建方式  |名称                                               |
+=============+================+==========+===================================================+
| 0.9.dev0    |  |gpu|         | 容器     |/lustre/share/img/tvm-0.9.dev0.sif                 |
+-------------+----------------+----------+---------------------------------------------------+
| 1.6.dev0    |  |gpu|         | 容器     |/dssg/share/imgs/tvm/0.9.dev0.sif思源平台          |
+-------------+----------------+----------+---------------------------------------------------+



提交 TVM 作业
----------------------

在思源超算提交TVM作业：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name tvm
   #SBATCH -N 1
   #SBATCH -n 1 
   #SBATCH -p a100
   #SBATCH --gres gpu:1
   #SBATCH --cpus-per-task 6


   IMAGE_PATH=/dssg/share/imgs/tvm/0.9.dev0.sif

   singularity run --nv --env TVM_LOG_DEBUG=1 $IMAGE_PATH python ...


在闵行超算提交TVM作业：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name tvm
   #SBATCH -N 1
   #SBATCH -n 1 
   #SBATCH -p dgx2
   #SBATCH --gres gpu:1


   IMAGE_PATH=/lustre/share/img/tvm-0.9.dev0.sif

   singularity run --nv --env TVM_LOG_DEBUG=1 $IMAGE_PATH python ...
  

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test


TVM单卡测试
----------------------

算例下载

.. code:: console

   $ git clone https://github.com/SJTU-HPC/HPCTesting.git
   $ cd HPCTesting/tvm/case1


A100测试环境配置

.. code:: console

   $ singularity run /dssg/share/imgs/tvm/0.9.dev0.sif pip install -r requirements.txt


DGX2测试环境配置

.. code:: console

   $ singularity run /lustre/share/img/tvm-0.9.dev0.sif pip install -r requirements.txt


A100测试脚本：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name tvm
   #SBATCH -N 1
   #SBATCH -n 1 
   #SBATCH -p a100
   #SBATCH --gres gpu:1
   #SBATCH --cpus-per-task 6


   IMAGE_PATH=/dssg/share/imgs/tvm/0.9.dev0.sif

   singularity run --nv --env TVM_LOG_DEBUG=1 $IMAGE_PATH python test.py


DGX2测试脚本：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name tvm
   #SBATCH -N 1
   #SBATCH -n 1 
   #SBATCH -p dgx2
   #SBATCH --gres gpu:1


   IMAGE_PATH=/lustre/share/img/tvm-0.9.dev0.sif

   singularity run --nv --env TVM_LOG_DEBUG=1 $IMAGE_PATH python test.py
  

结果
.. code:: console

   optimized: {'mean': 518.929668366909, 'median': 512.8163313493133, 'std': 10.976080937596128}
   unoptimized: {'mean': 592.0918963477015, 'median': 587.7139701507986, 'std': 10.555673599042604}


参考资料
--------

- `Apache TVM 官网 <https://tvm.apache.org/>`__
