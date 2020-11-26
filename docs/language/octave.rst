.. _octave:

Octave
======

##简介
-------
Octave是一种采用高级编程语言的主要用于数值分析的软件。Octave有助于以数值方式解决线性和非线性问题，并使用与MATLAB兼容的语言进行其他数值实验。它也可以作为面向批处理的语言使用。因为它是GNU计划的一部分，所以它是GNU通用公共许可证条款下的自由软件。

π集群上的Octave
---------------------------------

查看π集群上已编译的软件模块：

.. code:: bash

   module av octave

调用该模块：
.. code:: bash
   
   module load octave/5.2.0

示例slurm脚本：在small队列上，总共使用4核（n=4），脚本名称设为slurm.test
small队列slurm脚本示例Octave：
.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=small
   #SBATCH -n 4
   #SBATCH --ntasks-per-ode=4
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module purge
   module load octave/5.2.0

   ulimit -s unlimited
   ulimit -l unlimited

   octave [FILE_NAME]

用下方语句提交作业：

.. code:: bash

   $ sbatch slurm.test


使用HPC Studio启动Octave可视化界面
----------------------------------

首先参照\ `可视化平台 <../../login/HpcStudio/>`__\ 开启远程桌面，并在远程桌面中启动终端，并输入以下指令：

.. code:: bash

  module load octave/5.2.0
  octave [FILE_NAME]

参考资料
--------

-  `Octave官方网站 <https://www.gnu.org/software/octave/>`__
-  
