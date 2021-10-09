.. _elphbolt:

Elphbolt
========

简介
----
Elphbolt用于计算物理领域中的电子输运，采用的是严格电声耦合结合玻尔兹曼输运的方法

输入文件存放位置
-------------------------------

.. code:: bash

   #!/bin/bash

   workdir="./Si_6r4_300K_CBM_gcc/"
   inputdir="./input"

   mkdir $workdir
   cd $workdir

   #copy input file
   cp ../$inputdir/input.nml .

   #make soft link to the rest of the input data
   ln -s ../$inputdir/rcells_g .
   ln -s ../$inputdir/rcells_k .
   ln -s ../$inputdir/rcells_q .
   ln -s ../$inputdir/wsdeg_g .
   ln -s ../$inputdir/wsdeg_k .
   ln -s ../$inputdir/wsdeg_q .
   ln -s ../$inputdir/epwdata.fmt .
   ln -s ../$inputdir/epmatwp1 .
   ln -s ../$inputdir/FORCE_CONSTANTS_3RD .
   ln -s ../$inputdir/espresso.ifc2 .

运行脚本(在目录workdir下)
---------------------------------

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=elphbolt       
   #SBATCH --partition=small
   #SBATCH -N 1           
   #SBATCH --ntasks-per-node=4
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc/8.3.0-gcc-4.8.5
   module load openmpi/4.0.4-gcc-8.3.0
   module load netlib-lapack/3.8.0-gcc-8.3.0
   module load openblas/0.3.7-gcc-8.3.0
   module load elphbolt/1.0.0-gcc-8.3.0-openmpi-4.0.4
   
   cafrun -n 2 elphbolt.x

使用如下指令提交：

.. code:: bash

   $ sbatch elphbolt.slurm
