.. _calculix:

Calculix
==========

简介
----

CalculiX是一个设计来利用有限元方法求解场问题的软件，其既能够运行在类Unix(包括Linux)平台上，也能在MS-Windows上运行。使用CalculiX，你可以构建有限元模型，对模型进行求解以及后处理。CalculiX的预处理器和后处理器基于openGL API开发而成。其解器能够进行线性和非线性计算，包括求解静态、动态和热力学问题的模块。



CalculiX使用说明
-----------------------------

在思源一号上自行安装并使用CalculiX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. 使用conda创建虚拟环境并安装CalculiX：

.. code::
        
  srun -p 64c512g -n 1 --pty /bin/bash
  module load miniconda3/4.10.3
  conda create --name calculix_test
  source  activate calculix_test
  conda install -c conda-forge calculix==2.17

2. 创建一个目录calculixtest并进入该目录：

.. code::
        
    mkdir calculixtest
    cd calculixtest

3. 在该目录下创建如下测试文件beam10p.inp(输入参数文件)：

.. code::
        
  **
  **   Structure: cantilever beam under pressure.
  **   Test objective: C3D10 elements.
  **
  *NODE,NSET=NALL
       1,            1.,            0.
       2,            0.,            0.
       3,            1.,            1.
       4,           0.5,            0.
       5,           0.5,           0.5
       6,            1.,           0.5
       7,            0.,            1.
       8,            0.,           0.5
       9,           0.5,            1.
      10,            1.,            0.,            8.
      11,            1.,            1.,            8.
      12,            0.,            0.,            8.
      13,            1.,           0.5,            8.
      14,           0.5,           0.5,            8.
      15,           0.5,            0.,            8.
      16,            0.,            1.,            8.
      17,           0.5,            1.,            8.
      18,            0.,           0.5,            8.
      19,            1.,            0.,            2.
      20,            1.,            0.,            1.
      21,           0.5,            0.,            1.
      22,            0.,            0.,            2.
      23,           0.5,            0.,            2.
      24,            0.,            0.,            1.
      25,            1.,            0.,            4.
      26,            1.,            0.,            3.
      27,           0.5,            0.,            3.
      28,            0.,            0.,            4.
      29,           0.5,            0.,            4.
      30,            0.,            0.,            3.
      31,            1.,            0.,            6.
      32,            1.,            0.,            5.
      33,           0.5,            0.,            5.
      34,            0.,            0.,            6.
      35,           0.5,            0.,            6.
      36,            0.,            0.,            5.
      37,            1.,            0.,            7.
      38,           0.5,            0.,            7.
      39,            0.,            0.,            7.
      40,            0.,            1.,            2.
      41,            0.,           0.5,            1.
      42,            0.,            1.,            1.
      43,            0.,           0.5,            2.
      44,            0.,            1.,            4.
      45,            0.,           0.5,            3.
      46,            0.,            1.,            3.
      47,            0.,           0.5,            4.
      48,            0.,            1.,            6.
      49,            0.,           0.5,            5.
      50,            0.,            1.,            5.
      51,            0.,           0.5,            6.
      52,            0.,           0.5,            7.
      53,            0.,            1.,            7.
      54,            1.,            1.,            2.
      55,           0.5,            1.,            1.
      56,            1.,            1.,            1.
      57,           0.5,            1.,            2.
      58,            1.,            1.,            4.
      59,           0.5,            1.,            3.
      60,            1.,            1.,            3.
      61,           0.5,            1.,            4.
      62,            1.,            1.,            6.
      63,           0.5,            1.,            5.
      64,            1.,            1.,            5.
      65,           0.5,            1.,            6.
      66,           0.5,            1.,            7.
      67,            1.,            1.,            7.
      68,            1.,           0.5,            1.
      69,            1.,           0.5,            2.
      70,            1.,           0.5,            3.
      71,            1.,           0.5,            4.
      72,            1.,           0.5,            5.
      73,            1.,           0.5,            6.
      74,            1.,           0.5,            7.
     149,      0.499871,      0.499871,       3.99987
     150,           0.5,           0.5,            2.
     151,           0.5,           0.5,            1.
     152,      0.249935,      0.249935,       2.99994
     153,      0.249935,      0.749935,       3.99994
     154,      0.249935,      0.249935,       3.99994
     155,      0.249935,      0.749935,       4.99994
     156,      0.249935,      0.249935,       4.99994
     157,      0.749935,      0.749935,       3.99994
     158,      0.249935,      0.749935,       2.99994
     159,      0.749935,      0.749935,       4.99994
     160,           0.5,           0.5,            7.
     161,           0.5,           0.5,            6.
     162,      0.749935,      0.249935,       2.99994
     163,      0.749935,      0.249935,       3.99994
     164,      0.749935,      0.249935,       4.99994
  *ELEMENT, TYPE=C3D10, ELSET=EALL
      37,       2,      22,      19,      40,      24,      23,      21,
      41,      43,     150
      38,       2,       3,      54,       1,       5,      56,     151,
       4,       6,      68
      39,      22,      44,      28,     149,      45,      47,      30,
     152,     153,     154
      40,      28,      34,     149,      48,      36,     156,     154,
      49,      51,     155
      41,      40,       7,      54,       2,      42,      55,      57,
      41,       8,     151
      42,      54,       2,      19,      40,     151,      21,      69,
      57,      41,     150
      43,      40,      44,     149,      58,      46,     153,     158,
      59,      61,     157
      44,      44,      48,     149,      62,      50,     155,     153,
      63,      65,     159
      45,      34,      16,      11,      48,      52,      17,     160,
      51,      53,      66
      46,      11,      62,      31,      34,      67,      73,      74,
     160,     161,      35
      47,      12,      11,      34,      16,      14,     160,      39,
      18,      17,      52
      48,      22,      19,      40,     149,      23,     150,      43,
     152,     162,     158
      49,       7,       3,      54,       2,       9,      56,      55,
       8,       5,     151
      50,      22,      28,      25,     149,      30,      29,      27,
     152,     154,     163
      51,       1,       2,      19,      54,       4,      21,      20,
      68,     151,      69
      52,      58,      19,      40,      54,      70,     150,      59,
      60,      69,      57
      53,      28,      25,     149,      31,      29,     163,     154,
      33,      32,     164
      54,      62,      48,      34,      11,      65,      51,     161,
      67,      66,     160
      55,     149,      48,      34,      62,     155,      51,     156,
     159,      65,     161
      56,      34,      11,      10,      31,     160,      13,      38,
      35,      74,      37
      57,      10,      11,      34,      12,      13,     160,      38,
      15,      14,      39
      58,      58,      25,      62,     149,      71,      72,      64,
     157,     163,     159
      59,      19,      25,      58,     149,      26,      71,      70,
     162,     163,     157
      60,      22,      19,     149,      25,      23,     162,     152,
      27,      26,     163
      61,      25,      62,     149,      31,      72,     159,     163,
      32,      73,     164
      62,      28,      48,     149,      44,      49,     155,     154,
      47,      50,     153
      63,      28,      34,      31,     149,      36,      35,      33,
     154,     156,     164
      64,      34,      62,      31,     149,     161,      73,      35,
     156,     159,     164
      65,      22,     149,      40,      44,     152,     158,      43,
      45,     153,      46
      66,      44,      62,     149,      58,      63,     159,     153,
      61,      64,     157
      67,     149,      58,      19,      40,     157,      70,     162,
     158,      59,     150
  *NSET,NSET=FIX
  1,4,2,6,5,8,3,9,7
  *NSET,NSET=LOAD
  10,15,12,13,14,18,11,17,16
  *BOUNDARY
  1,1,2
  3,1,1
  FIX,3,3
  *MATERIAL,NAME=EL
  *ELASTIC
  210000.,.3
  *SOLID SECTION,ELSET=EALL,MATERIAL=EL
  *STEP
  *STATIC
  *CLOAD
  LOAD,2,1.
  *NODE PRINT,NSET=NALL
  U,RF
  *EL PRINT,ELSET=EALL
  S
  *END STEP




4. 在该目录下创建如下作业提交脚本calculixtest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=calculixtest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  ccx beam10p

5. 使用如下命令提交作业：

.. code::

  sbatch calculixtest.slurm

6. 作业完成后在.out文件中可看到如下结果：

.. code::

  ************************************************************

  CalculiX Version 2.17, Copyright(C) 1998-2020 Guido Dhondt
  CalculiX comes with ABSOLUTELY NO WARRANTY. This is free
  software, and you are welcome to redistribute it under
  certain conditions, see gpl.htm

  ************************************************************

  You are using an executable made on Wed Oct 14 07:29:32 UTC 2020

  The numbers below are estimated upper bounds

  number of:

   nodes:          164
   elements:           67
   one-dimensional elements:            0
   two-dimensional elements:            0
   integration points per element:            4
   degrees of freedom per node:            3
   layers per element:            1

   distributed facial loads:            0
   distributed volumetric loads:            0
   concentrated loads:            9
   single point constraints:           12
   multiple point constraints:            1
   terms in all multiple point constraints:            1
   tie constraints:            0
   dependent nodes tied by cyclic constraints:            0
   dependent nodes in pre-tension constraints:            0

   sets:            4
   terms in all sets:          229

   materials:            1
   constants per material and temperature:            2
   temperature points per material:            1
   plastic data points per material:            0

   orientations:            0
   amplitudes:            2
   data points in all amplitudes:            2
   print requests:            3
   transformations:            0
   property cards:            0


   STEP            1

   Static analysis was selected

   Decascading the MPC's

   Determining the structure of the matrix:
   number of equations
   258
   number of nonzero lower triangular matrix elements
   6834

   Using up to 1 cpu(s) for the stress calculation.

   Using up to 1 cpu(s) for the symmetric stiffness/mass contributions.

   Factoring the system of equations using the symmetric spooles solver
   Using up to 1 cpu(s) for spooles.

   Using up to 1 cpu(s) for the stress calculation.


   Job finished

  ________________________________________

  Total CalculiX Time: 0.034328
  ________________________________________



在pi2.0上自行安装并使用Calculix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 使用conda创建虚拟环境并安装Calculix：

.. code::
        
  srun -p small -n 1 --pty /bin/bash
  module load miniconda3/4.8.2
  conda create --name calculix_test
  source  activate calculix_test
  conda install -c conda-forge calculix==2.17



2. 此步骤和上文完全相同；



3. 此步骤和上文完全相同；


4. 在该目录下创建如下作业提交脚本calculixtest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=calculixtest      
  #SBATCH --partition=small    
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  ccx beam10p

5. 使用如下命令提交作业：

.. code::

  sbatch calculixtest.slurm

6. 作业完成后在.out文件中可看到如下结果：

.. code::

  ************************************************************

  CalculiX Version 2.17, Copyright(C) 1998-2020 Guido Dhondt
  CalculiX comes with ABSOLUTELY NO WARRANTY. This is free
  software, and you are welcome to redistribute it under
  certain conditions, see gpl.htm

  ************************************************************

  You are using an executable made on Wed Oct 14 07:29:32 UTC 2020

  The numbers below are estimated upper bounds

  number of:

   nodes:          164
   elements:           67
   one-dimensional elements:            0
   two-dimensional elements:            0
   integration points per element:            4
   degrees of freedom per node:            3
   layers per element:            1

   distributed facial loads:            0
   distributed volumetric loads:            0
   concentrated loads:            9
   single point constraints:           12
   multiple point constraints:            1
   terms in all multiple point constraints:            1
   tie constraints:            0
   dependent nodes tied by cyclic constraints:            0
   dependent nodes in pre-tension constraints:            0

   sets:            4
   terms in all sets:          229

   materials:            1
   constants per material and temperature:            2
   temperature points per material:            1
   plastic data points per material:            0

   orientations:            0
   amplitudes:            2
   data points in all amplitudes:            2
   print requests:            3
   transformations:            0
   property cards:            0


   STEP            1

   Static analysis was selected

   Decascading the MPC's

   Determining the structure of the matrix:
   number of equations
   258
   number of nonzero lower triangular matrix elements
   6834

   Using up to 1 cpu(s) for the stress calculation.

   Using up to 1 cpu(s) for the symmetric stiffness/mass contributions.

   Factoring the system of equations using the symmetric spooles solver
   Using up to 1 cpu(s) for spooles.

   Using up to 1 cpu(s) for the stress calculation.


   Job finished

  ________________________________________

  Total CalculiX Time: 0.034328
  ________________________________________


  



参考资料
-----------

-  `Calculix 官网 <http://www.calculix.de/>`__
-  `Calculix 案例文件 <https://github.com/calculix/CalculiX-Examples>`__

