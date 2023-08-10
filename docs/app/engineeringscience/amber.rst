.. _amber:

Amber
======

简介
----

Amber 是分子动力学软件，用于蛋白质、核酸、糖等生物大分子的计算模拟。

Amber 为商业软件，根据 \ `Amber官网 <https://ambermd.org/GetAmber.php>`__\ ，非商业性使用的用户在官网登记，阅读并同意non-commerical license后即可下载软件安装包。

如需使用集群上的 Amber，请下载\ `non-commerical license <https://hpc.sjtu.edu.cn/Item/docs/AMBER_Software_License.pdf>`__\，按照下面模板发送邮件至 hpc 邮箱。

邮件模板： 我是超算账号xxx使用人，已经阅读了附件的License，同意该许可，并确认所有使用均为非商业性使用。

算例存放位置
--------------

.. code:: bash

   思源  ：/dssg/share/sample/amber
   π2.0 ：/lustre/share/sample/amber

算例结构如下

.. code:: bash

   tree amber:
   ├── inpcrd
   ├── mdin
   └── prmtop

集群上的Amber
--------------------

- `思源一号 Amber`_

- `π2.0 Amber`_

.. _思源一号 Amber:

思源一号上的Amber
-------------------------------------

思源上的Amber2022
~~~~~~~~~~~~~~~~~~

amber_GPU.slurm内容如下：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=Amber_gpu
   #SBATCH --partition=a100
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=16
   #SBATCH --gres=gpu:1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load amber/2022-cuda-11.5.119
   pmemd.cuda -O -i mdin -o mdout -p prmtop -c inpcrd 

amber_MPI.slurm内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=Amber_mpi     
   #SBATCH --partition=64c512g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load  amber/2022-intel-2021.4.0
   mpirun pmemd.MPI -O -i mdin -o mdout -p prmtop -c inpcrd

思源上的Amber2020
~~~~~~~~~~~~~~~~~~

amber_MPI.slurm内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=Amber_mpi     
   #SBATCH --partition=64c512g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load  amber/2020-intel-2021.4.0
   mpirun pmemd.MPI -O -i mdin -o mdout -p prmtop -c inpcrd

.. _π2.0 Amber:

π2.0上的Amber
-------------------------------------

π2.0上的Amber2022
~~~~~~~~~~~~~~~~~~~~


amber_GPU.slurm内容如下：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=Amber_gpu
   #SBATCH --partition=dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=6
   #SBATCH --gres=gpu:1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load amber/2022-cuda-10.1.243
   pmemd.cuda -O -i mdin -o mdout -p prmtop -c inpcrd 

amber_MPI.slurm内容如下：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test_amber
   #SBATCH --partition=cpu
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load amber/2022-intel-2021.4.0
   mpirun pmemd.MPI -O -i mdin -o mdout -p prmtop -c inpcrd

π2.0上的Amber2020
~~~~~~~~~~~~~~~~~~~

amber_GPU.slurm内容如下：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=Amber_gpu
   #SBATCH --partition=dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=6
   #SBATCH --gres=gpu:1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load amber/2020-cuda-10.2.89
   pmemd.cuda -O -i mdin -o mdout -p prmtop -c inpcrd 

amber_MPI.slurm内容如下：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test_amber
   #SBATCH --partition=cpu
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load amber/2020-intel-2021.4.0
   mpirun pmemd.MPI -O -i mdin -o mdout -p prmtop -c inpcrd

运行结果(单位为：s)
---------------------

Amber22在GPU上的运行结果

+-------------+-------------+------------+
| 平台        | 思源        | pi 2.0     |
+=============+=============+============+
| 核数        | 16core+1GPU | 6core+1GPU |
+-------------+-------------+------------+
| Time        |  60.57      | 60.99      |
+-------------+-------------+------------+

Amber20在GPU上的运行结果

+-------------+------------+
| 平台        | pi 2.0     |
+=============+============+
| 核数        | 6core+1GPU |
+-------------+------------+
| Time        | 60.94      |
+-------------+------------+

Amber22在CPU上的运行结果

+-------------+-------------+------------+------------+-----------+----------+------------+
| 平台        | 思源        | pi 2.0     | 思源       | pi 2.0    | 思源     | pi 2.0     |
+=============+=============+============+============+===========+==========+============+
| 核数        | 64          | 40         | 128        | 80        | 256      | 160        |   
+-------------+-------------+------------+------------+-----------+----------+------------+
| Time        |  446.36     | 722.14     | 311.67     | 428.30    | 306.37   | 315.61     |
+-------------+-------------+------------+------------+-----------+----------+------------+

Amber20在CPU上的运行结果

+-------------+-------------+------------+------------+-----------+----------+------------+
| 平台        | 思源        | pi 2.0     | 思源       | pi 2.0    | 思源     | pi 2.0     |
+=============+=============+============+============+===========+==========+============+
| 核数        | 64          | 40         | 128        | 80        | 256      | 160        |
+-------------+-------------+------------+------------+-----------+----------+------------+
| Time        | 441.83      | 694.00     | 309.26     | 430.69    | 306.83   | 312.89     |
+-------------+-------------+------------+------------+-----------+----------+------------+

参考资料
--------

-  `Amber 官网 <https://ambermd.org/>`__
