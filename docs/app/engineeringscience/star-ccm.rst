.. _star-ccm:

STAR-CCM+
=========

简介
----

Much more than just a CFD solver, STAR-CCM+ is an entire engineering
process for solving problems involving flow (of fluids or solids), heat
transfer and stress.

STAR-CCM+需自行安装
--------------------

STAR-CCM+
为商业软件，需要自行购买和安装。建议先和软件厂商工程师充分沟通如下问题：1)
能否安装在普通用户目录下；2) 是否支持浮动 License、是否需要
License服务器、License 服务器能否安装在虚拟机上；3)
能否提供用于运行作业的SLURM作业调度系统脚本

安装完成后，还需在 π 集群上设置以下内容：

.. code:: bash

   1. 清空 ~/.ssh/known_hosts 文件内容

   2. 执行下面两行
   ssh-keygen -t rsa
   ssh-copy-id -i ~/.ssh/id_rsa localhost

   3. 在 ~/.ssh/config 中头几行增加如下两行内容： 
          StrictHostKeyChecking no
          UserKnownHostsFile=/dev/null

   4. chmod 600 ~/.ssh/config

π 集群上的 Slurm 脚本 slurm.test
-------------------------------------

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH -J test
   #SBATCH -p cpu
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40

   module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

   ulimit -s unlimited
   ulimit -l unlimited

   cat /dev/null > machinefile
   scontrol show hostname $SLURM_JOB_NODELIST > machinefile

   starccm+ -power -mpi intel -machinefile './machinefile' -np $SLURM_NTASKS -rsh ssh -cpubind -batch run -batch-report YOURsample.sim

π 集群上提交作业
-------------------

.. code:: bash

   $ sbatch slurm.test

参考资料
--------

-  `STAR-CCM+ 网站 <https://www.femto.eu/star-ccm/>`__
