.. _deepmd-kit:

DeePMD-kit
==========

简介
----

DeePMD-kit是一种基于机器学习的分子动力学模拟方法，该方法是通过使用从头计算得到的数据对深度神经网络模型进行训练，从而得到通用的多体势能模型（DP模型）。由于其是基于第一性原理，而具有媲美量子力学的精度。其计算效率高，比第一性原理计算至少快5个数量级。目前DP模型已成功应用于水和含水体系，金属和合金，相图,高熵陶瓷，化学反应，固态电解质，离子液体等研究领域。


安装教程
----------

本案例使用 ``conda`` 进行安装

.. code:: bash

   srun -p small -n 4 --pty /bin/bash #申请计算资源
   module load miniconda3 #加载模块
   conda create -n deepmd deepmd-kit=*=*gpu libdeepmd=*=*gpu lammps-dp cudatoolkit=11.3 horovod -c https://conda.deepmodeling.org #创建名为deepmd的环境
   source activate deepmd #激活deepmd环境

测试
------

文档使用官方文档提供的甲烷燃烧模拟案例进行测试，介绍基于DP模型的MD模拟过程，测试所需要的文件可以通过以下链接下载https://github.com/tongzhugroup/Chapter13-tutorial

数据集准备
~~~~~~~~~~~~~~~~~~

在数据集准备过程中，需要考虑的是最终DP模型的准确性，即需要在哪个量子力学水平标记数据。本案例使用在MN15/6-31G**级别下通过Gaussian程序计算产生的势能和原子力，其中由于模拟过程需要处理诸多自由基及其反应，因此选择了对单/多参考系统具有良好精度的MN15。示例中已提供准备好的数据集。

.. code:: bash

   unzip Chapter13-tutorial-master.zip #解压下载的示例文件
   cd Chapter13-tutorial-master

DP模型训练
~~~~~~~~~~~

此步骤是提取经过训练的神经网络模型，生成冻结模型graph.pb，并压缩

.. code:: bash

   dp freeze -o graph.pb
   dp compress -i graph.pb -o graph_compressed.pb -t methane_param.json

基于DP模型的MD模拟
~~~~~~~~~~~~~~~~~~~~~~~~~~

冻结模型可用于反应MD模型，以探索甲烷燃烧过程的详细反应机理。此处的MD模拟过程由LAMMPS程序完成（LAMMPS已在生成deepmd环境时完成创建）。体系中包含100个甲烷分子和200个氧气分子，下载的data.methane文件中包含系统初始原子坐标信息。input.lammps文件则包含了调用LAMMPS程序的关键参数

.. code:: bash

   lmp -i input.lammps
   conda deactivate #退出deepmd环境 

参考资料
--------

-  `DeePMD-kit 官网 <https://docs.deepmodeling.com/projects/deepmd/en/master/index.html>`__
