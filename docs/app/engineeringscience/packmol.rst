.. _packmol:

packmol
=========

简介
----

Packmol 是一款开源软件，专门用于创建分子动力学（MD）模拟的初始配置。它的核心功能是将分子精确地放置在预定义的空间区域内，以确保短程排斥相互作用不会干扰模拟过程。Packmol 能够处理各种复杂的分子排列，例如层状、球形或管状的脂质层，并且支持多种输入文件格式，如PDB、TINKER、XYZ和MOLDY，这使得它具有广泛的使用范围。

可用的版本
----------

+----------+-------+-----------+------------------------------------+
| 集群     | 平台  |版本       | 模块名                             |
+==========+=======+===========+====================================+
| 思源一号 | |cpu| | 20.0.0    | packmol/20.0.0-gcc-11.2.0          |
+----------+-------+-----------+------------------------------------+
| Pi 2.0   | |cpu| | 20.0.0    | packmol/20.0.0-gcc-8.5.0           |
+----------+-------+-----------+------------------------------------+

集群上的 Packmol
-------------------

- `思源一号 Packmol`_

- `Pi 2.0 Packmol`_



.. _思源一号 Packmol:

思源一号 Packmol
---------------------

slurm 作业脚本示例：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=packmol
   #SBATCH --partition=64c512g
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module purge
   module load packmol/20.0.0-gcc-11.2.0
   
   packmol < packmol.inp



.. _Pi 2.0 Packmol:

π2.0 Packmol
----------------

slurm 作业脚本示例：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=packmol
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module purge
   module load packmol/20.0.0-gcc-8.5.0
   
   packmol < packmol.inp


运行官方算例
---------------------------------------

获取算例
~~~~~~~~
.. code:: bash

   wget https://m3g.github.io/packmol/examples/examples.tar.gz
   tar -xvf examples.tar.gz
   cd examples

算例信息：`bilayer`
~~~~~~~~~~~~~~~~~~~~
.. code:: bash

   Lipid bilayer with water over and below
   Commented input file: bilayer-comment.inp
   Clean input file: bilayer.inp
   Alternative input file using different constraints:	bilayer2.inp
   Molecules needed: palmitoil.pdb water.pdb
   Output file: bilayer.pdb
   Running time: 26 seconds
   Number of atoms: 7200
   Number of molecules:	1100
   Number of variables: 6600


思源一号
~~~~~~~~
.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=packmol
   #SBATCH --partition=64c512g
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module purge
   module load packmol/20.0.0-gcc-11.2.0
   
   packmol < bilayer.inp

π2.0
~~~~~
.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=packmol
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module purge
   module load packmol/20.0.0-gcc-8.5.0
   
   packmol < bilayer.inp


计算完成部分日志 
~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

   ################################################################################
   Solution written to file: bilayer.pdb
   --------------------------------------------------------------------------------
   Running time:    24.3353214      seconds. 
   --------------------------------------------------------------------------------

参考资料
--------

-  `Packmol 官网 <https://m3g.github.io/packmol/userguide.shtml>`__
