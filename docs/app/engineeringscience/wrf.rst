.. _wrf:

WRF
===

简介
----
WRF(Weather Research and Forecasting Model)模式是有美国环境预报中心(NCEP),
美国国家大气研究中心（NCAR）以及多个大学、研究所和业务部门联合研发的一种统一的中尺度天气预报模式。
WRF模式适用范围很广，从中小尺度到全球尺度的数值预报和模拟都有广泛的应用。

CPU版WRF(4.3.1)
----------------

WPS前期处理数据所需要的geo等数据的路径为：

.. code:: bash

   [user@login3 ~]$ls /scratch/share/stu/cheng2021/
   combine.obsgrid  EMIS  fnl_data  geo_data_WRF4

复制WPS和WRF环境到本地，并导入相应模块

.. code:: bash

   srun -p cpu -n 40 --pty /bin/bash
   cd $HOME
   mkdir wrf
   cd wrf
   cp -r /lustre/opt/contribute/cascadelake/wrf_cmaq/packet_1/WPS-4.3.1 ./
   cp -r /lustre/opt/contribute/cascadelake/wrf_cmaq/packet_1/WRF-master ./

   module load wrf_cmaq/5.3.3-wrf-4.3.1


ARM版WRF_bisheng 
-------------------------------

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1           
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load wrf/4.2-bisheng-1.3.3
   export WRF_HOME=/lustre/opt/contribute/kunpeng920/wrf/packet/WRF-4.2
   export PBV=CLOSE
   export OMP_PROC_BIND=TRUE
   export OMP_NUM_THREADS=4
   export OMP_STACKSIZE="16M"
   export WRF_NUM_TILES=128

   mpirun -np 32 --bind-to core --map-by ppr:32:node:pe=4 numactl -l $WRF_HOME/main/wrf.exe

ARM版WRF_bisheng_1
---------------------------------

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1           
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load 4.2-bisheng-1.3.3_1
   export WRF_HOME=/lustre/opt/contribute/kunpeng920/wrf/packet/WRF_1/WRF-4.2/main
   export PBV=CLOSE
   export OMP_PROC_BIND=TRUE
   export OMP_NUM_THREADS=4
   export OMP_STACKSIZE="16M"
   export WRF_NUM_TILES=128

   mpirun -np 32 --bind-to core --map-by ppr:32:node:pe=4 numactl -l $WRF_HOME/main/wrf.exe

ARM版WRF_OpenMPI
----------------

ARM平台上运行脚本如下

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1           
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   module load wrf/4.2-gcc-9.3.0-openmpi
   module load openmpi/4.0.3-gcc-9.3.0

   export WRF_HOME=/lustre/opt/kunpeng920/linux-centos7-aarch64/gcc-9.3.0/wrf-4.2-dvii6gqnopsssz5yytk4xcgrk2g2d2ob
   export PBV=CLOSE
   export OMP_PROC_BIND=TRUE
   export OMP_NUM_THREADS=4
   export OMP_STACKSIZE="16M"
   export WRF_NUM_TILES=128

   mpirun -np 32 --bind-to core --map-by ppr:32:node:pe=4 numactl -l $WRF_HOME/main/wrf.exe

使用如下指令提交：

.. code:: bash

   $ sbatch wrf.slurm
