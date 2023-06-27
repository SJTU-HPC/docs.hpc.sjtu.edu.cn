.. _basilisk:

Basilisk
==========

简介
----
Basilisk也是一个开源软件，用于解决自适应直角坐标网格上的偏微分方程

Basilisk基本使用
------------------------------------
超算中使用Basilisk可以通过镜像直接调用，pi 2.0和思源一号上的软件镜像位置不同，脚本中要注意。
1. 在pi 2.0上，编写basilisk_test.slurm脚本，根据软件镜像所在位置进行使用：


.. code:: bash

  #!/bin/bash

  #SBATCH -J test
  #SBATCH -p small
  #SBATCH -o %j.out
  #SBATCH -e %j.err
  #SBATCH -n 1
  #SBATCH --cpus-per-task=1

  IMAGE_PATH=/lustre/share/img/basilisk/basilisk-8.5.0.sif
  singularity exec  $IMAGE_PATH qcc --version

2. 在思源一号上，编写basilisk_test.slurm脚本，根据软件镜像所在位置进行使用：

.. code:: bash

  #!/bin/bash

  #SBATCH -J test
  #SBATCH -p 64c512g
  #SBATCH -o %j.out
  #SBATCH -e %j.err
  #SBATCH -n 1
  #SBATCH --cpus-per-task=1

  IMAGE_PATH=/dssg/share/imgs/basilisk/basilisk-8.5.0.sif
  singularity exec  $IMAGE_PATH qcc --version

3. 提交脚本：

.. code:: bash

   sbatch basilisk_test.slurm 

4. 运行结束后可在.out文件中得到如下结果：

.. code:: bash

  cc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-4)
  Copyright (C) 2018 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

参考资料
--------


-  `Basilisk使用手册 <http://basilisk.fr/Tutorial>`__

