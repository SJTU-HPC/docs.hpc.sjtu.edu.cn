.. _keras:

Keras
=====

简介
----

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。

如果你在以下情况下需要深度学习库，请使用 Keras：
- 允许简单而快速的原型设计（由于用户友好，高度模块化，可扩展性）。
- 同时支持卷积神经网络和循环神经网络，以及两者的组合。
- 在 CPU 和 GPU 上无缝运行。


π 集群上的Keras安装方法
--------------------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install cudatoolkit=11.0
   pip install keras tensorflow-gpu==2.8.0


π 集群上的Slurm脚本 slurm.test
---------------------------------

在 dgx2 队列上，使用 1 张卡（gres=gpu:1），配合 6 核芯 (n = 6)

.. code:: bash

   #!/bin/bash

   #SBATCH -J keras
   #SBATCH --partition=dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 1
   #SBATCH --ntasks-per-node=1
   #SBATCH -N 1
   #SBATCH --cpus-per-task 6
   #SBATCH --gres=gpu:1
   #SBATCH --mem=MaxMemPerNode

   ulimit -l unlimited
   ulimit -s unlimited
   
   module load miniconda3
   source activate mypy

   python ...


在 A100 队列上，使用 1 张卡（gres=gpu:1），配合 6 核芯 (n = 6)

.. code:: bash

   #!/bin/bash

   #SBATCH -J keras
   #SBATCH --partition=a100
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 1
   #SBATCH --ntasks-per-node=1
   #SBATCH -N 1
   #SBATCH --cpus-per-task 6
   #SBATCH --gres=gpu:1
   #SBATCH --mem=MaxMemPerNode

   ulimit -l unlimited
   ulimit -s unlimited

   module load miniconda3
   source activate mypy

   python ...


π 集群上提交作业
------------------

.. code:: bash

   $ sbatch slurm.test


算例测试
-------------------
超算中心提供了用来测试keras的算例。

使用命令

.. code:: console

   $ cd ~
   $ git clone https://github.com/SJTU-HPC/HPCTesting.git
   $ cd HPCTesting/keras/case1
   $ conda env create -f environment.yml
   $ curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
   $ unzip -q kagglecatsanddogs_3367a.zip


在π 超算上，使用如下脚本来提交该算例作业：

.. code:: bash

   #!/bin/bash
   #SBATCH -p dgx2
   #SBATCH -N 1
   #SBATCH -n 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --cpus-per-task 6 
   #SBATCH --gres gpu:1

   cd ~/HPCTesting/keras/case1
   module load miniconda3
   source activate kerastest
   export LD_LIBRARY_PATH=~/.conda/envs/kerastest/lib/:$LD_LIBRARY_PATH
   python image_classification_from_scratch.py


在思源一号上，使用如下脚本来提交该算例作业：

.. code:: bash

   #!/bin/bash
   #SBATCH -p a100
   #SBATCH -N 1
   #SBATCH -n 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --cpus-per-task 6 
   #SBATCH --gres gpu:1

   cd ~/HPCTesting/keras/case1
   module load miniconda3
   source activate kerastest
   export LD_LIBRARY_PATH=~/.conda/envs/kerastest/lib/:$LD_LIBRARY_PATH
   python image_classification_from_scratch.py

将以上脚本保存为 ``test.slurm`` ，使用 ``sbatch test.slurm`` 来交作业。

参考资料
--------

-  `Keras 官网 <https://keras.io/>`__
