.. _gerris:

Gerris
======

简介
----
Gerris是求解描述流体流动的偏微分方程的开源软件。

Gerris基本使用
------------------------------------
1. 创建一个目录，并在该目录下编写如下vorticity.gfs文件：


.. code:: bash

  1 2 GfsSimulation GfsBox GfsGEdge {} {
  GfsTime { end = 50 }
  GfsRefine 6
  GfsInit {} {
    U = (0.5 - rand()/(double)RAND_MAX)
    V = (0.5 - rand()/(double)RAND_MAX)
  }  
  GfsOutputTime            { istep = 10 } stdout
  GfsOutputProjectionStats { istep = 10 } stdout
  }
  GfsBox {}
  1 1 right
  1 1 top

2. 在该目录下编写如下gerristest.slurm运行脚本：

.. code:: bash

  #!/bin/bash

  #SBATCH -J test
  #SBATCH -p small
  #SBATCH -o %j.out
  #SBATCH -e %j.err
  #SBATCH -n 1
  #SBATCH --cpus-per-task=1

  IMAGE_PATH=/lustre/share/img/x86/gerris/gerris.sif
  singularity exec  $IMAGE_PATH gerris2D vorticity.gfs

3. 提交脚本：

.. code:: bash

   sbatch gerristest.slurm 

4. 运行结束后可在.out文件中得到如下结果(部分)：

.. code:: bash

   step:       0 t:      0.00000000 dt:  1.888931e-02 cpu:      0.01000000 real:      0.01103200
  Approximate projection
    niter:    3
    residual.bias:   -4.374e-18  2.233e-18
    residual.first:   3.751e-01  4.525e-05     20
    residual.second:  4.633e-01  6.034e-05     20
    residual.infty:   1.419e+00  4.128e-04     15
  step:      10 t:      0.30436678 dt:  3.947231e-02 cpu:      0.18000000 real:      0.20647300
  MAC projection        before     after       rate
    niter:    2
    residual.bias:    1.746e-19  9.098e-20
    residual.first:   2.313e-02  9.074e-05     16
    residual.second:  3.342e-02  1.171e-04     17
    residual.infty:   2.310e-01  5.609e-04     20
  Approximate projection
    niter:    2
    residual.bias:   -5.667e-19  2.551e-19
    residual.first:   1.873e-02  5.536e-05     18
    residual.second:  2.423e-02  7.055e-05     19
    residual.infty:   1.063e-01  3.206e-04     18



参考资料
--------


-  `Gerris官网 <http://gfs.sourceforge.net/wiki/index.php/Main_Page>`__

