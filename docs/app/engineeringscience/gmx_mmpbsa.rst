.. _gmx_mmpbsa:

gmx_MMPBSA
=============

简介
----

gmx_MMPBSA是一种基于AMBER的MMPBSA.py开发的新工具，旨在使用GROMACS文件执行端态自由能计算。它与所有GROMACS版本以及AmberTools20或21一起使用，与现有程序相比，它在兼容性、多功能性、分析和并行化方面都有改进。在当前版本中，gmx_MMPBSA支持多种不同的系统，包括但不限于：蛋白质、蛋白质配体、蛋白质DNA、金属蛋白肽、蛋白聚糖、膜蛋白、 多组分系统（例如，蛋白质DNA RNA离子配体）

可用的版本
-----------

+--------+---------+----------+------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                               |
+========+=========+==========+======================================================+
| 1.5.2  | |cpu|   | 容器     | gmx_mmpbsa/1.5.2-gcc-9.3.0 思源一号                  |
+--------+---------+----------+------------------------------------------------------+
| 2020   | |cpu|   | 容器     | gmx_MMPBSA/1.4.3-gcc-9.3.0-ambertools-20-gromacs2021 |
+--------+---------+----------+------------------------------------------------------+
| 1.6.2  | |cpu|   | conda    | /lustre/share/conda_env/gmxMMPBSA                    |
+--------+---------+----------+------------------------------------------------------+

安装方法
----------------

conda安装gmx_MMPBSA

1. env.yml
   
.. code:: bash

   name:
   channels:
     - conda-forge
     - bioconda
     - defaults
   dependencies:
     - python=3.9
     - ambertools=21.12
     - mpi4py=3.1.3
     - compilers=1.2.0
     - gromacs==2022.4
     - git
     - pip
     - pip: 
       - pyqt5==5.15.6
       - gmx-mmpbsa
       - git+https://github.com/Valdes-Tresanco-MS/ParmEd.git@v3.4
       - pandas==1.2.2
       - seaborn<0.12
       - scipy>=1.6.1
       - matplotlib>=3.5.1
       - tqdm

2.安装命令：

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3/22.11.1
   source activate
   conda env create -n gmxMMPBSA --file env.yml
   conda activate gmxMMPBSA
   conda install -c conda-forge ocl-icd-system

3. 使用conda安装的gmx_MMPBSA的计算脚本，该方法只需要计算脚本不需要run1.sh

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=gmx_MMPBSA      
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load miniconda3/22.11.1
   source activate gmxMMPBSA

   mpirun -np 2 gmx_MMPBSA -O -i mmpbsa.in -cs com.tpr -ci index.ndx -cg 3 4 -ct com_traj.xtc -nogui


算例获取方式
-------------

.. code:: bash

   cd ~ && git clone https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA.git
   ls gmx_MMPBSA/docs/examples

为保证顺利运行，请将gmx_MMPBSA.slurm、run1.sh和数据放在同一目录下

集群上的gmx_MMPBSA
--------------------

- `思源一号 gmx_MMPBSA`_

- `π2.0 gmx_MMPBSA`_

.. _思源一号 gmx_MMPBSA:

思源一号上的gmx_MMPBSA（版本：1.5.2）
-------------------------------------

gmx_MMPBSA.slurm内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=gmx_MMPBSA      
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   ulimit -l unlimited
   ulimit -s unlimited
   
   module load gmx_mmpbsa/1.5.2-gcc-9.3.0
   gmx_MMPBSA_1.5.2

run1.sh脚本内容如下：

.. code:: bash

   #!/bin/bash
   gmx_MMPBSA MPI -O -i mmpbsa.in -cs com.tpr -ci index.ndx -cg 1 13 -ct com_traj.xtc -nogui

给run1.sh增加可执行权限

.. code:: bash

   chmod +x run1.sh

只有将gmx_MMPBSA.slurm、run1.sh和数据放在同一目录下才可正常运行。

使用如下命令提交：

.. code:: bash

   $ sbatch gmx_MMPBSA.slurm

.. _π2.0 gmx_MMPBSA:

π2.0 gmx_MMPBSA（版本：1.4.3）
------------------------------------

gmx_MMPBSA.slurm内容如下：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=gmx_MMPBSA      
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load gmx_MMPBSA/1.4.3-gcc-9.3.0-ambertools-20-gromacs2021
   mpirun gmx_MMPBSA_GROMACS2021

run1.sh脚本内容如下：

.. code:: bash

   #!/bin/bash 
   gmx_MMPBSA MPI -O -i mmpbsa.in -cs com.tpr -ci index.ndx -cg 1 13 -ct com_traj.xtc -nogui

给run1.sh增加可执行权限

.. code:: bash

   chmod +x run1.sh

只有将gmx_MMPBSA.slurm、run1.sh和数据放在同一目录下才可正常运行。

使用如下命令提交：

.. code:: bash

   $ sbatch gmx_MMPBSA.slurm

π2.0-KOS平台 gmx_MMPBSA（版本：1.6.2）
----------------------------------------------------

gmx_MMPBSA.slurm内容如下：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=gmx_MMPBSA      
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load miniconda3/22.11.1
   source activate /lustre/share/conda_env/gmxMMPBSA

   mpirun -np 2 gmx_MMPBSA -O -i mmpbsa.in -cs com.tpr -ci index.ndx -cg 3 4 -ct com_traj.xtc -nogui

该版本只需要gmx_MMPBSA.slurm和数据文件，不需要run1.sh即可正常运行。

使用如下命令提交：

.. code:: bash

   $ sbatch gmx_MMPBSA.slurm


运行结果
---------

+------+----------+----------+-------+-------+
| 平台 | 思源一号 | 思源一号 | pi2.0 | pi2.0 |
+======+==========+==========+=======+=======+
| 核数 | 64       | 128      | 40    | 80    |
+------+----------+----------+-------+-------+
| 时间 | 72s      | 65s      | 117s  | 75s   |
+------+----------+----------+-------+-------+


参考资料
--------

-  `gmx_MMPBSA 官网 <https://valdes-tresanco-ms.github.io/gmx_MMPBSA/>`__
