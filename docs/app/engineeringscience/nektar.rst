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

   module avail Nektar

加载预安装的Nektar++
---------------------

π 集群系统中已经预装 nektar-5.0.0 (intel 版本)，可以用以下命令加载:

::

   module load nektar/5.0.0-intel-19.0.4-impi

Nektar++使用说明
-----------------------------

1. 在这里，我们通过求解一个二维方形区域的对流方程(单核串行)来演示Nektar++的使用方法。该问题的定义如下：

|image1|


其中，:math:`x_b  和  y_b  代表计算域边界，V_x=2，V_y=3，\kappa=2\pi`。

1.1 从 `Nektar 官网 <https://www.nektar.info/>`__ 的GETTING STARTED->Tutorials->Basics->Advection-Diffusion->Introduction->Goals板块下载所需要的数据文件basics-advection-diffusion.tar.gz并解压；
 
1.2 解压之后会得到两个目录completed以及tutorial；

1.3 进入completed目录会看到如下几个文件：

.. code:: bash

   ADR_conditions.xml  

   ADR_mesh.geo  

   ADR_mesh.msh 

   ADR_mesh.xml 

   ADR_mesh_aligned.fld

   ADR_mesh_aligned.xml  

*这几个文件定义了求解本问题所需要的几何信息、网格信息以及初始和边界条件。*




1.4 在此目录下编写如下Nektar_run.slurm脚本：



.. code:: bash

   #!/bin/bash

   #SBATCH -J Nektar_test
   #SBATCH -p small
   #SBATCH -n 1
   #SBATCH --ntasks-per-node=1
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load nektar/5.0.0-intel-19.0.4-impi

   ulimit -s unlimited
   ulimit -l unlimited

   ADRSolver ADR_mesh_aligned.xml ADR_conditions.xml

1.5 使用如下指令提交：

.. code:: bash

   sbatch Nektar_run.slurm

1.6 然后即可在.out或者.err文件中看到如下结果：

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



2. 可压缩圆柱绕流(多核并行)。

2.1 从 `Nektar 官网 <https://www.nektar.info/>`__ 的GETTING STARTED->Tutorials->Compressible Flow Solver->Subsonic Cylinder->Introduction->Goals板块下载所需要的数据文件cfs-CylinderSubsonic_NS.tar.gz并解压；
 
2.2 解压之后会得到两个目录completed以及tutorial；

2.3 在tutorial目录下编写以下Nektar_run.slurm脚本：


.. code:: bash

   #!/bin/bash

   #SBATCH -J Nektar_test
   #SBATCH -p cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load nektar/5.0.0-intel-19.0.4-impi
   module load openmpi/3.1.5-gcc-9.2.0 

   ulimit -s unlimited
   ulimit -l unlimited

   mpirun -np 32 CompressibleFlowSolver CylinderSubsonic_NS.xml

2.4 使用如下指令提交：

.. code:: bash

   sbatch Nektar_run.slurm

2.5 作业运行完毕后即可在.out或者.err文件中看到如下结果(部分)：

.. code:: bash

  =======================================================================
	        EquationType: NavierStokesCFE
	        Session Name: CylinderSubsonic_NS
	        Spatial Dim.: 2
	  Max SEM Exp. Order: 3
	      Expansion Dim.: 2
	      Riemann Solver: HLLC
	      Advection Type: 
	     Projection Type: Discontinuous Galerkin
	      Diffusion Type: 
	           Advection: explicit
	       AdvectionType: WeakDG
	           Diffusion: explicit
	           Time Step: 1e-05
	        No. of Steps: 60000
	 Checkpoints (steps): 400
	    Integration Type: RungeKutta
  =======================================================================
  =======================================================================
	        EquationType: NavierStokesCFE
	        Session Name: CylinderSubsonic_NS
	        Spatial Dim.: 2
	  Max SEM Exp. Order: 3
	      Expansion Dim.: 2
	      Riemann Solver: HLLC
	      Advection Type: 
	     Projection Type: Discontinuous Galerkin
	      Diffusion Type: 
  =======================================================================






在自己的目录下自行安装Nektar++
------------------------------------------



1. 执行以下从命令从GitHub上下载Nektar++源码：

.. code:: bash

   git clone http://gitlab.nektar.info/nektar/nektar.git nektar++

2. 下载完成后进入nektar++目录并通过源码编译安装(编译之前需要配置很多可选的编译选项，用户根据自己的具体情况自行选择即可)：

.. code:: bash

  cd nektar++
  mkdir build && cd build
  ccmake ../
  make
  make install





参考资料
--------



-  `Nektar 官网 <https://www.nektar.info/>`__




.. |image1| image:: ../../img/Nektar1.png
