.. _nektar:

Nektar++
==========

简介
----

Nektar++ is a spectral/hp element framework designed to support the
construction of efficient high-performance scalable solvers for a wide
range of partial differential equations.

π 集群上的Nektar++
----------------------

查看 π 集群上已编译的软件模块:

.. code:: bash

   $ module avail Nektar

加载预安装的Nektar++
---------------------

π 集群系统中已经预装 nektar-5.0.0 (intel 版本)，可以用以下命令加载:

::

   $ module load nektar/5.0.0-intel-19.0.4-impi

Nektar++使用说明
-----------------------------

在这里，我们通过求解一个二维方形区域的对流扩散方程来来演示Nektar++的使用方法。该问题的具体定义如下：

|image1|

(1)  从Nektar++官网 https://www.nektar.info/ 的 GETTING STARTED->Tutorials->Basics->Advection-Diffusion->Introduction->Goals板块下载所需要的数据文件basics-advection-diffusion.tar.gz 并解压；
 
(2) 解压之后会得到两个文件夹 completed 以及 tutorial；

(3) 进入 completed 文件夹会看到如下几个文件：

  ADR_conditions.xml  

  ADR_mesh.geo  

  ADR_mesh.msh 

  ADR_mesh.xml 

  ADR_mesh_aligned.fld

  ADR_mesh_aligned.xml  

这几个文件定义了求解本问题所需要的几何信息、网格信息以及初始和边界条件。




(4) 在此目录下编写如下Nektar_run.slurm脚本：



.. code:: bash

   #!/bin/bash

   #SBATCH -J Nektar_test
   #SBATCH -p small
   #SBATCH -n 4
   #SBATCH --ntasks-per-node=4
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load nektar/5.0.0-intel-19.0.4-impi

   ulimit -s unlimited
   ulimit -l unlimited

   ADRSolver ADR_mesh_aligned.xml ADR_conditions.xml

(5) 使用如下指令提交：

.. code:: bash

   $ sbatch Nektar_run.slurm

(6) 然后即可在.out或者.err文件中看到如下结果：

.. code:: bash

  ========================================= 
                EquationType: UnsteadyAdvection 
                Session Name: ADR_mesh_aligned 
                Spatial Dim.: 2 
          Max SEM Exp. Order: 5 
              Expansion Dim.: 2 
              Riemann Solver: Upwind 
              Advection Type: 
             Projection Type: Discontinuous Galerkin 
                   Advection: explicit 
                   Diffusion: explicit 
                   Time Step: 0.001 
                No. of Steps: 1000 
         Checkpoints (steps): 100 
            Integration Type: ClassicalRungeKutta4 
  ========================================== 
  Initial Conditions: 
  - Field u: sin(k*x)*cos(k*y) 
  Writing: "ADR_mesh_aligned_0.chk" 
  Steps: 100      Time: 0.1          CPU Time: 0.435392s 
  Writing: "ADR_mesh_aligned_1.chk" 
  Steps: 200      Time: 0.2          CPU Time: 0.430588s 
  Writing: "ADR_mesh_aligned_2.chk" 
  Steps: 300      Time: 0.3          CPU Time: 0.428503s 
  Writing: "ADR_mesh_aligned_3.chk" 
  Steps: 400      Time: 0.4          CPU Time: 0.428529s 
  Writing: "ADR_mesh_aligned_4.chk" 
  Steps: 500      Time: 0.5          CPU Time: 0.430142s 
  Writing: "ADR_mesh_aligned_5.chk" 
  Steps: 600      Time: 0.6          CPU Time: 0.429481s 
  Writing: "ADR_mesh_aligned_6.chk" 
  Steps: 700      Time: 0.7          CPU Time: 0.433232s 
  Writing: "ADR_mesh_aligned_7.chk" 
  Steps: 800      Time: 0.8          CPU Time: 0.431088s 
  Writing: "ADR_mesh_aligned_8.chk" 
  Steps: 900      Time: 0.9          CPU Time: 0.427919s 
  Writing: "ADR_mesh_aligned_9.chk" 
  Steps: 1000     Time: 1            CPU Time: 0.436098s 
  Writing: "ADR_mesh_aligned_10.chk" 
  Time-integration  : 4.31097s 
  Writing: "ADR_mesh_aligned.fld" 
  ------------------------------------------- 
  Total Computation Time = 4s 
  ------------------------------------------- 
  L 2 error (variable u) : 0.00863475 
  L inf error (variable u) : 0.0390366

参考资料
--------

-  `Nektar 官网 <https://www.nektar.info/>`__




.. |image1| image:: ../../img/Nektar1.png
