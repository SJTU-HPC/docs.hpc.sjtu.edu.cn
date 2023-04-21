.. _gromacs:

GROMACS
=======

简介
----

GROMACS
是一种分子动力学应用程序，可以模拟具有数百至数百万个粒子的系统的牛顿运动方程。GROMACS旨在模拟具有许多复杂键合相互作用的生化分子，例如蛋白质，脂质和核酸。

可用的版本
----------

+--------+-------+----------+----------------------------------------------------------+
| 版本   | 平台  | 构建方式 | 模块名                                                   |
+========+=======+==========+==========================================================+
| 2022.2 | |gpu| | spack    | gromacs/2022.2-intel-2021.4.0-cuda 思源一号              |
+--------+-------+----------+----------------------------------------------------------+
| 2022.3 | |gpu| | spack    | gromacs/2022.3-intel-2021.4.0-cuda 思源一号              |
+--------+-------+----------+----------------------------------------------------------+
| 2021.3 | |cpu| | spack    | gromacs/2021.3-intel-2021.4.0 思源一号                   |
+--------+-------+----------+----------------------------------------------------------+
| 2021.3 | |cpu| | spack    | gromacs/2021.3-gcc-11.2.0-cuda-openblas-openmpi 思源一号 |
+--------+-------+----------+----------------------------------------------------------+
| 2021.2 | |gpu| | spack    | gromacs/2021.2-intel-2021.4.0-cuda 思源一号              |
+--------+-------+----------+----------------------------------------------------------+
| 2021.4 | |gpu| | spack    | gromacs/2021.4-gcc-11.2.0-cuda-openblas 思源一号         |
+--------+-------+----------+----------------------------------------------------------+
| 2021.6 | |cpu| | spack    | gromacs/2021.6-gcc-9.2.0-openmpi-4.0.5                   |
+--------+-------+----------+----------------------------------------------------------+
| 2019.4 | |cpu| | spack    | gromacs/2019.4-gcc-9.2.0-openmpi                         |
+--------+-------+----------+----------------------------------------------------------+
| 2020   | |cpu| | 容器     | gromacs/2020-cpu-double                                  |
+--------+-------+----------+----------------------------------------------------------+
| 2022   | |arm| | 源码     | gromacs/2022-gcc-9.3.0                                   |
+--------+-------+----------+----------------------------------------------------------+

算例下载
---------

.. code:: bash

   mkdir ~/gromacs && cd ~/gromacs
   wget -c https://ftp.gromacs.org/pub/benchmarks/water_GMX50_bare.tar.gz
   tar xf water_GMX50_bare.tar.gz
   cd water-cut1.0_GMX50_bare/0768/    

数据目录如下所示：

.. code:: bash
         
   [hpc@login2 water-cut1.0_GMX50_bare]$ tree 0768/
   0768/
   ├── conf.gro
   ├── pme.mdp
   ├── rf.mdp
   └── topol.top
   
   0 directories, 4 files

集群上的GROMACS
----------------

- `一. 思源一号 GROMACS`_

- `二. π2.0 GROMACS`_

- `三. ARM GROMACS`_


.. _一. 思源一号 GROMACS:

一. 思源一号 GROMACS
--------------------

1.INTEL编译的版本
~~~~~~~~~~~~~~~~~~

预处理数据-CPU版本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=64_gromacs       
   #SBATCH --partition=64c512g  
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load oneapi
   module load gromacs/2021.3-intel-2021.4.0
   gmx_mpi grompp -f pme.mdp 

预处理数据-GPU版本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=gpu_gromacs       
   #SBATCH --partition=a100
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=16
   #SBATCH --gres=gpu:1 
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load oneapi
   module load gromacs/2022.2-intel-2021.4.0-cuda
   module load cuda/11.5.0
   gmx_mpi grompp -f pme.mdp 

提交作业脚本-CPU版本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=64_gromacs       
   #SBATCH --partition=64c512g  
   #SBATCH -N 2 
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load oneapi
   module load gromacs/2021.3-intel-2021.4.0
   mpirun gmx_mpi mdrun -dlb yes -v -nsteps 10000 -resethway -noconfout -pin on -ntomp 1 -s topol.tpr

提交作业脚本-GPU版本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=gpu_gromacs       
   #SBATCH --partition=a100
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=16
   #SBATCH --gres=gpu:1 
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load oneapi
   module load gromacs/2022.2-intel-2021.4.0-cuda
   module load cuda/11.5.0
   mpirun -n 1 gmx_mpi mdrun -dlb yes -v -nsteps 10000 -resethway -noconfout -pin on -ntomp 16 -gpu_id 0 -s topol.tpr 

2.GCC编译的版本
~~~~~~~~~~~~~~~~

预处理数据

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=64_gromacs
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load gcc/11.2.0
   module load openmpi/4.1.1-gcc-11.2.0
   module load gromacs/2021.3-gcc-11.2.0-cuda-openblas-openmpi
   gmx_mpi grompp -f pme.mdp 

提交预处理作业脚本。

.. code:: bash

   $ sbatch pre.slurm

运行结果如下所示：

.. code:: bash

   [hpchgc@login water]$ tree 0768
   0768
   ├── 9854405.err
   ├── 9854405.out
   ├── conf.gro
   ├── mdout.mdp
   ├── pme.mdp
   ├── pre.slurm
   ├── rf.mdp
   ├── topol.top
   └── topol.tpr

提交作业脚本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=64_gromacs
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc/11.2.0
   module load openmpi/4.1.1-gcc-11.2.0
   module load gromacs/2021.3-gcc-11.2.0-cuda-openblas-openmpi
   mpirun gmx_mpi mdrun -dlb yes -v -nsteps 10000 -resethway -noconfout -pin on -ntomp 1 -s topol.tpr
   
提交上述作业

.. code:: bash

   sbatch gromacs.slurm
   
运行结果如下所示：

.. code:: bash

   [hpchgc@sylogin1 64cores]$ tail -n 20 9853399.err
   vol 0.94  imb F  2% pme/F 0.92 step 10000, remaining wall clock time:     0 s


   Dynamic load balancing report:
    DLB was permanently on during the run per user request.
    Average load imbalance: 2.0%.
    The balanceable part of the MD step is 85%, load imbalance is computed from this.
    Part of the total run time spent waiting due to load imbalance: 1.7%.
    Steps where the load balancing was limited by -rdd, -rcon and/or -dds: X 0 % Y 0 %
    Average PME mesh/force load: 0.923
    Part of the total run time spent waiting due to PP/PME imbalance: 2.4 %


                  Core t (s)   Wall t (s)        (%)
          Time:     3052.051       47.699     6398.5
                    (ns/day)    (hour/ns)
   Performance:       18.117        1.325
   
   GROMACS reminds you: "The Stingrays Must Be Fat This Year" (Red Hot Chili Peppers)
  

.. _π2.0 GROMACS:

二. π2.0 GROMACS
------------------

1.gromacs/2021.6-gcc-9.2.0-openmpi-openmpi-4.0.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

提交预处理脚本

.. code:: bash

   #!/bin/bash

   #SBATCH -J gromacs_cpu_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load gromacs/2021.6-gcc-9.2.0-openmpi-4.0.5
   module load openmpi/4.0.5-gcc-9.2.0
   module load gcc/9.2.0

   ulimit -s unlimited
   ulimit -l unlimited
   gmx_mpi grompp -f pme.mdp 

提交运行作业脚本

.. code:: bash
            
   #!/bin/bash

   #SBATCH -J gromacs_cpu_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load gromacs/2021.6-gcc-9.2.0-openmpi-4.0.5
   module load openmpi/4.0.5-gcc-9.2.0
   module load gcc/9.2.0

   ulimit -s unlimited
   ulimit -l unlimited
   mpirun gmx_mpi mdrun -dlb yes -v -nsteps 10000 -resethway -noconfout -pin on -ntomp 1 -s topol.tpr

2.gromacs/2020-cpu-double 
~~~~~~~~~~~~~~~~~~~~~~~~~

提交预处理脚本

.. code:: bash

   #!/bin/bash

   #SBATCH -J gromacs_cpu_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   
   module load gromacs/2019.4-gcc-9.2.0-openmpi
   
   ulimit -s unlimited
   ulimit -l unlimited
   gmx_mpi grompp -f pme.mdp

提交运行作业脚本

.. code:: bash

   #!/bin/bash
   
   #SBATCH -J gromacs_cpu_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   
   module load gromacs/2020-cpu-double
   
   ulimit -s unlimited
   ulimit -l unlimited
   srun --mpi=pmi2 gmx_mpi_d mdrun -dlb yes -v -nsteps 10000 -resethway -noconfout -pin on -ntomp 1 -s topol.tpr

.. _ARM GROMACS:

三. ARM GROMACS
--------------------

1.module load gromacs/2022-gcc-9.3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

提交预处理脚本

.. code:: bash

   #!/bin/bash

   #!/bin/bash
   
   #SBATCH --job-name=test
   #SBATCH --partition=arm128c256g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load gromacs/2022-gcc-9.3.0
   
   gmx_mpi grompp -f pme.mdp

提交运行作业脚本

.. code:: bash
            
   #!/bin/bash

   #SBATCH --job-name=test
   #SBATCH --partition=arm128c256g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=128
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load gromacs/2022-gcc-9.3.0
   export OMP_NUM_THREADS=1
   mpirun gmx_mpi mdrun -dlb yes -v -nsteps 10000 -resethway -noconfout -pin on -ntomp 1 -s topol.tpr

运行结果如下所示(单位：ns/day，越高越好)
-----------------------------------------

1.GROMACS 思源一号
~~~~~~~~~~~~~~~~~~

+------------------------------------------------------+
|         gromacs/2021.3-intel-2021.4.0                |
+=============+=============+============+=============+
| 核数        | 64          | 128        | 192         |
+-------------+-------------+------------+-------------+
| Performance |  17.724     | 35.250     | 53.321      |
+-------------+-------------+------------+-------------+

+------------------------------------------------------+
|      gromacs/2021.3-gcc-11.2.0-cuda-openblas-openmpi |
+=============+=============+============+=============+
| 核数        | 64          | 128        | 192         |
+-------------+-------------+------------+-------------+
| Performance |  10.6259    | 32.798     | 55.635      |
+-------------+-------------+------------+-------------+

+-----------------------------------------+
|      gromacs/2022.3-intel-2021.4.0-cuda |
+=====================+===================+
| 卡数                |  1块A100          |
+---------------------+-------------------+
| Performance         |  37.081           |
+---------------------+-------------------+

2.GROMACS π2.0
~~~~~~~~~~~~~~~~

+----------------------------------------------+
|     gromacs/2021.6-gcc-9.2.0-openmpi-4.0.5   |
+=============+==========+==========+==========+
| 核数        | 40       | 80       | 120      |
+-------------+----------+----------+----------+
| Performance |  8.584   | 17.266   | 30.913   |
+-------------+----------+----------+----------+

+-----------------------------------------------+
|            gromacs/2020-cpu-double            |
+==============+==========+==========+==========+
| 核数         | 40       | 80       | 120      |
+--------------+----------+----------+----------+
| Performance  |  4.441   | 8.388    | 16.701   |
+--------------+----------+----------+----------+

3.GROMACS ARM
~~~~~~~~~~~~~~~~

+--------------------------------------------------+
|                gromacs/2022-gcc-9.3.0            |
+==============+===========+===========+===========+
| 核数         | 128       | 256       | 512       |
+--------------+-----------+-----------+-----------+
| Performance  |  7.754    | 15.466    | 30.650    |
+--------------+-----------+-----------+-----------+

参考资料
--------

- gromacs官方网站 http://www.gromacs.org/
