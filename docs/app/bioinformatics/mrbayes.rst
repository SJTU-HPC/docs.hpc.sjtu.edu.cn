.. _mrbayes:

MrBayes
=======

简介
----

MrBayes is a program for Bayesian inference and model choice across a
wide range of phylogenetic and evolutionary models. MrBayes uses Markov
chain Monte Carlo (MCMC) methods to estimate the posterior distribution
of model parameters.

π 集群上的MrBayes
---------------------------

查看 π 集群上已编译的软件模块:

.. code:: bash

   $ module spider mrbayes

调用该模块:

.. code:: bash

   $ module load mrbayes/3.2.7a-gcc-8.3.0-openmpi

π 集群上的Slurm脚本 slurm.test
--------------------------------

在 cpu 队列上，总共使用 16 核 (n = 16)
cpu 队列每个节点配有 40核，这里使用了 1 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=mrbayes
   #SBATCH --partition=cpu
   #SBATCH -n 16
   #SBATCH --ntasks-per-node=16
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module load mrbayes/3.2.7a-gcc-8.3.0-openmpi

   srun --mpi=pmi2 mb your_input_file

根据我们的测试，mrbayes最多只能使用16进程/节点的配置，请根据具体需要调整\ ``-n``\ 和\ ``--ntasks-per-node``\ 参数

π 集群上提交作业
------------------

.. code:: bash

   $ sbatch mrbayes_cpu_gnu.slurm

参考资料
--------

-  `MrBayes 官网 <http://nbisweden.github.io/MrBayes/>`__
