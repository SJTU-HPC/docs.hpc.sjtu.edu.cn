minife
======

简介
----

MiniFE是非结构隐式有限元代码的代理应用程序。它类似于HPCCG和pHPCCG，但提供了此类应用程序中各个步骤的更完整的垂直介绍。MiniFE旨在成为“非结构化隐式有限元或有限体积应用程序的最佳近似值求解程序”


测试平台
--------

- `思源一号 miniFE`_

- `π2.0 miniFE`_

.. _思源一号 miniFE:

思源一号miniFE
--------------

运行脚本如下所示：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=minife
   #SBATCH --partition=64c512g 
   #SBATCH -N 4
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.error
   
   module load minife/2.1.0-intel-2021.4.0
   
   mpirun miniFE.x nx=600 verify_solution=1


运行结果如下所示：

.. code:: bash

   CG solve: 
     Iterations: 200
     Final Resid Norm: 0.00431172
     WAXPY Time: 2.34283
     WAXPY Flops: 3.90096e+11
     WAXPY Mflops: 166506
     DOT Time: 2.37993
     DOT Flops: 1.728e+11
     DOT Mflops: 72607.1
     MATVEC Time: 12.7909
     MATVEC Flops: 2.34837e+12
     MATVEC Mflops: 183598
     Total: 
       Total CG Time: 17.5337
       Total CG Flops: 2.91127e+12
       Total CG Mflops: 166039
     Time per iteration: 0.0876683
   Total Program Time: 68.6005

结果显示部分最重要的值是CG求解的计算值 ```CG Mflops``` ,另外运行时间显示也是衡量并行速率的重要参考依据。

.. _π2.0 miniFE:

π2.0 miniFE
--------------

运行脚本如下所示：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=minife
   #SBATCH --partition=cpu
   #SBATCH -N 4
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.error
   
   module load minife/2.1.0-intel-2021.4.0
   
   mpirun miniFE.x nx=600 verify_solution=1


运行结果如下所示：

.. code:: bash

   CG solve: 
     Iterations: 200
     Final Resid Norm: 0.00423874
     WAXPY Time: 3.80524
     WAXPY Flops: 3.90096e+11
     WAXPY Mflops: 102515
     DOT Time: 1.65335
     DOT Flops: 1.728e+11
     DOT Mflops: 104515
     MATVEC Time: 19.1712
     MATVEC Flops: 2.34837e+12
     MATVEC Mflops: 122495
     Total: 
       Total CG Time: 24.6569
       Total CG Flops: 2.91127e+12
       Total CG Mflops: 118071
     Time per iteration: 0.123285
   Total Program Time: 75.0806   

结果显示部分最重要的值是CG求解的计算值 ```CG Mflops``` ,另外运行时间显示也是衡量并行速率的重要参考依据。

miniFE的运行结果比较
----------------------


思源一号上miniFE的运行结果
~~~~~~~~~~~~~~~~~~~~~~~~~~

+------+-----------+------------+
| 核数 | CG Mflops | Total time |      
+======+===========+============+
| 64   | 44365.2   | 118.973    |
+------+-----------+------------+
| 128  | 87730.4   | 81.8141    |
+------+-----------+------------+
| 256  | 166039    | 68.6005    |
+------+-----------+------------+

π2.0上miniFE的运行结果
~~~~~~~~~~~~~~~~~~~~~~~~

+------+-----------+------------+
| 核数 | CG Mflops | Total time |
+======+===========+============+
| 40   | 30003.6   | 175.854    |
+------+-----------+------------+
| 80   | 59460.3   | 107.695    |
+------+-----------+------------+
| 160  | 118071    | 75.0806    |
+------+-----------+------------+

参考资料

- miniFE https://github.com/Mantevo/miniFE/tree/v2.1.0

