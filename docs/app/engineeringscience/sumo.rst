.. _sumo:

SUMO
=====================

简介
---------------

SUMO，全称Simulation of Urban Mobility，是开源、微观、多模态的交通仿真软件，发展始于2000年。
它纯粹是微观的，可以针对每辆车进行单独控制，因此非常适合交通控制模型的开发。


CPU 版本 sumo (GUI)
---------------------------

.. code:: bash

   1. 用pi集群账号登录 https://studio.hpc.sjtu.edu.cn/ 平台
   2. 在网站通过  Interactive Apps -> Desktop -> Launch 进入桌面
   3. 打开终端，通过命令行来调用软件
   module purge
   module load sumo/1.10.0-sumo
   可用命令为 sumo-gui 及 netedit

