.. _gift-bte:

Gift-BTE
========

Gift-ΒΤΕ 是对介观尺度的声子导热问题进行数值计算的C++软件，由上海交通大学密西根学院、未来技术学院鲍华课题组开发。该软件可以用于各种微纳结构的声子导热仿真，包括但不限于半导体器件、微纳多孔结构等。

可用版本
--------

+--------+---------+----------+-----------------------------+
| 版本   | 平台    | 构建方式 | 模块名                      |
+========+=========+==========+=============================+
| 1.0    | |cpu|   | 源码     | bte/1.0-openmpi-3.1.5  π2.0 |
+--------+---------+----------+-----------------------------+

算例获取
--------

.. code:: bash

   mkdir ~/bte && cd ~/bte
   cp -r /lustre/share/benchmarks/bte/input.tar.gz ./
   tar xf input.tar.gz

作业运行
--------

π2.0集群
~~~~~~~~

作业运行前数据、脚本所在目录如下所示：

.. code:: bash

   [hpc@login3 BTE]$ tree data/
   data/
   ├── input
   │   ├── FinFet_3D_2500.mphtxt
   │   ├── heatfile.dat
   │   ├── inputband_8.dat
   │   ├── inputbc_Finfet.dat
   │   ├── inputdata.dat
   │   └── inputmesh.dat
   ├── input.tar.gz
   └── run.slurm

运行脚本如下所示：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=bte-test
   #SBATCH --partition=cpu
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=32
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   export OMP_NUM_THREADS=1
   module load bte
   mpirun  BTEcmd

提交上述作业

.. code:: bash

   sbatch run.slurm

作业运行结束后的目录如下所示：

.. code:: bash

   [hpc@login3 BTE]$ tree data/
   data/
   ├── 9729078.err
   ├── 9729078.out
   ├── Boundary_heat_flux.dat
   ├── HeatFlux.dat
   ├── input
   │   ├── FinFet_3D_2500.mphtxt
   │   ├── heatfile.dat
   │   ├── inputband_8.dat
   │   ├── inputbc_Finfet.dat
   │   ├── inputdata.dat
   │   └── inputmesh.dat
   ├── Interface_emit_temp.dat
   ├── run.slurm
   ├── Tempcell1.dat
   ├── Tempcell2.dat
   └── Tempcell.dat

上述文件的具体含义可参考BTE官方网站：bte.sjtu.edu.cn.

文件内容最后一行显示如下内容，代表作业运行正确。

.. code:: bash

   [hpc@login3 data]$ tail -n 1 9729078.out 
   Time taken by iteration: 509080 milliseconds

运行结果
--------

π2.0
~~~~

+-------------------------------------------------+
|               bte/1.0-openmpi-3.1.5             |
+===================+=========+=========+=========+
| 核数              | 16      | 32      | 64      |
+-------------------+---------+---------+---------+
| 时间 milliseconds | 637674  | 618820  | 509080  |
+-------------------+---------+---------+---------+


