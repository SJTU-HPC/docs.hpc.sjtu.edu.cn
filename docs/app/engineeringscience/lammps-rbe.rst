.. _lammps-rbe:

LAMMPS-RBE
==========

简介
----

LAMMPS-RBE是由上海交通大学上海应用数学中心团队基于LAMMPS二次开发的自研软件。该版本在长程力模拟中引入了先进的Random Batch Ewald算法，RBE使用基于动理学或连续介质理论的路径, 研究复杂环境中微纳系统的多体效应，并结合分子动力学进行多尺度建模和数学分析。LAMMPS-RBE突破了传统分子动力学在 CPU集群上可扩展性差的问题，可以使百万级别粒子以上的大尺度体系的计算成本降低一个数量级。

详细算法解释可以参阅: https://math.sjtu.edu.cn/faculty/xuzl/RBE.pdf

新增功能细节描述在本文底部。

可用的版本
----------

+----------+-------+----------+------------------------------------------------------------+
| 版本     | 平台  | 构建方式 | 模块名                                                     |
+==========+=======+==========+============================================================+
| 7Aug2019 | |cpu| | 源码     | lammps-rbe/20190807-oneapi-2021.4 思源一号                 |
+----------+-------+----------+------------------------------------------------------------+
| 7Aug2019 | |cpu| | 源码     | lammps-rbe/20190807-intel-parallel-studio-2020.1-impi π2.0 |
+----------+-------+----------+------------------------------------------------------------+
| 7Aug2019 | |cpu| | 容器     | lammps-rbe/20190807-oneapi-2021.1-impi π2.0                |
+----------+-------+----------+------------------------------------------------------------+

算例获取
--------

π2.0上数据获取
~~~~~~~~~~~~~~

.. code:: bash

   mkdir ~/lammps-rbe && cd ~/lammps-rbe
   cp -r /lustre/share/benchmarks/lammps-rbe/lammps-rbe.tar.gz .
   tar xf lammps-rbe.tar.gz
   cd RBE_Example/

思源一号上数据获取
~~~~~~~~~~~~~~~~~~

.. code:: bash

   mkdir ~/lammps-rbe && cd ~/lammps-rbe
   cp -r /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/lammps-rbe-2019.8.4/Lammps-RBE/RBE_Example ./
   cd  RBE_Example/

数据的目录结构如下所示：

.. code:: bash

   [hpc@login2 Lammps-RBE]$ tree RBE_Example/
   RBE_Example/
   ├── in.spce-bulk-nvt
   └── lmp.data
 
注意：
本文算例步数设置为40000，即文件 ``in.spce-bulk-nvt`` 最后一行内容为： ``run 40000``

运行核数要和文件 ``in.spce-bulk-nvt`` 中的参数 ``kspace_style	rbe 0.07 200 80`` 最后一个数字保持一致，比如本例中为： ``80``

不同集群上的 LAMMPS-RBE
-----------------------

- `思源一号 LAMMPS-RBE`_

- `π2.0 LAMMPS-RBE`_

.. _思源一号 LAMMPS-RBE:

思源一号
--------

作业脚本如下所示：

.. code:: bash

   #!/bin/bash
   
   #SBATCH -J lammps-rbe
   #SBATCH -p 64c512g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH -o RBE.out
   #SBATCH -e %j.err
   
   module load lammps-rbe/20190807-oneapi-2021.4
   mpirun lmp_intel_cpu_intelmpi -i in.spce-bulk-nvt

提交上述脚本

.. code:: bash

   sbatch lammps-rbe.slurm

运行结果如下所示：

.. code:: bash

   [hpc@node738 bte]$ tail -n 1 RBE_BAOBAO.log 
   Total wall time: 0:02:01

.. _π2.0 LAMMPS-RBE:

π2.0
----

源码编译版本
~~~~~~~~~~~~

lammps-rbe/20190807-intel-parallel-studio-2020.1-impi

作业脚本如下所示：

.. code:: bash

   #!/bin/bash
   
   #SBATCH -J lammps 
   #SBATCH -p cpu
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=40  
   #SBATCH --exclusive
   #SBATCH -o RBE.out
   #SBATCH -e %j.err  
   
   module load lammps-rbe/20190807-intel-parallel-studio-2020.1-impi
   mpirun lmp_intel_cpu_intelmpi -i in.spce-bulk-nvt

提交作业：

.. code:: bash

   $ sbatch lammps-rbe.slurm

运行结果如下：

.. code:: bash

   [hpc@login2 80cores_intel]$ tail -n 1 RBE_BAOBAO.log 
   Total wall time: 0:03:28

容器版本
~~~~~~~~

lammps-rbe/20190807-oneapi-2021.1-impi π2.0

运行脚本如下：

.. code:: bash

   #!/bin/bash
   
   #SBATCH -J lammps 
   #SBATCH -p cpu
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=40  
   #SBATCH --exclusive
   #SBATCH -o RBE.out
   #SBATCH -e %j.err  
   
   module load lammps-rbe/20190807-oneapi-2021.1-impi
   mpirun lammps-rbe -i in.spce-bulk-nvt

提交上述作业

.. code:: bash

   sbatch lammps-rbe.slurm

运行结果如下所示：

.. code:: bash

   [hpc@login2 80cores]$ tail -n 1 RBE_BAOBAO.log 
   Total wall time: 0:04:26

运行结果
--------

思源一号上的结果
~~~~~~~~~~~~~~~~

+-------------------------------------------------------+
|           lammps-rbe/20190807-oneapi-2021.4           |
+==============+============+=============+=============+
| 核数         | 64         | 128         | 256         |
+--------------+------------+-------------+-------------+
| wall time    | 0:03:10    | 0:02:02     | 0:01:26     |
+--------------+------------+-------------+-------------+

π2.0上的结果
~~~~~~~~~~~~

+-------------------------------------------------------+
| lammps-rbe/20190807-intel-parallel-studio-2020.1-impi |
+==============+============+=============+=============+
| 核数         | 40         | 80          | 160         |
+--------------+------------+-------------+-------------+
| wall time    | 0:06:09    | 0:03:28     | 0:02:09     |
+--------------+------------+-------------+-------------+

+-------------------------------------------------------+
|         lammps-rbe/20190807-oneapi-2021.1-impi        |
+==============+============+=============+=============+
| 核数         | 40         | 80          | 160         |
+--------------+------------+-------------+-------------+
| wall time    | 0:06:17    | 0:04:26     | 0:03:48     |
+--------------+------------+-------------+-------------+

新增功能
--------

同Lammps已有功能相比，该版本新增三个功能：

1. 基于Random Batch Ewald (RBE)算法的三维周期/二维准周期平板系统静电求解器，特别适用于多核模拟。
调用方式：在Lammps的input文件中加入下面命令（需和pair/lj/cut/coul/long配合使用，这点和PPPM算法相同），

kspace_style rbe arg1 arg2 arg3

其中kspace_style是Lammps固定指令，表示模拟中要计算静电相互作用；rbe是算法名称表示调用RBE算法计算静电；
arg1=alpha，是RBE算法里用于控制近远场比例的参数，该参数的选择和Ewald以及PPPM算法相同。如果希望相对误差是1e-4，那么需选取使得erfc(r_cut*sqrt{alpha})≈1e-4的alpha, 其中r_cut是在pair/lj/cut/coul/long中选取的静电近场截断；arg2=batch_size，是在傅里叶空间中做随机采样得到的样本数量，一般为几十至数百（越大越准确，越小计算速度越快）；arg3=sampling_core，用于采样的CPU核的数量，需>1且<总MPI数量，一般可选取和用户使用的MPI数量相同或MPI数量一半。两个使用案例（假设使用200个CPU核）：

pair_style      lj/cut/coul/long 10.0 10.0

kspace_style    rbe 0.07 500 100

或调用intel的近场计算

pair_style      lj/cut/coul/long/intel 12.0 12.0

kspace_style    rbe 0.05 200 100

如果希望处理二维周期且z方向是两块平板的系统，需要在input文件中定义平板的位置参数和kspace_modify slab 3,方法同LAMMPS官方文档中用PPPM算平板问题的方式一致。


2. 基于RBE2D算法的二维周期，Z方向具介电不匹配界面（Dielectric Interfaces）系统的静电求解器（包括界面带连续面电荷情形），特别适用于多核模拟，并且速度大幅超过其他处理Dielectric Interfaces的静电算法。
调用方式：在Lammps的input文件中加入下面命令

pair_style lj/cut/coul/long/rbed arg1 arg2 arg3 arg4

kspace_style Rbed arg1 arg2 arg3 arg4 arg5 arg6 arg7

pair_style和lj/cut/coul/long/rbed分别是Lammps固定指令（表示计算静电近场）和算法名称（表示使用RBE2D算法）；arg1=LJ_cut，是LJ相互作用的截断半径；arg2=Coul_cut，是静电相互作用的截断半径（需小于等于LJ截断半径，这点和LAMMPS原始设置相同）；arg3=gamma_top，arg4=gamma_down分别是上下界面的描绘介电不匹配程度的系数，取值范围都是[-1,+1]，定义分别为(ε_in-ε_top)/ (ε_in+ε_top)和(ε_in-ε_down)/ (ε_in+ε_down)，其中ε_in，ε_top和ε_down分别是盒子中间、盒子上方、盒子下方介质的相对介电常数。

kspace_style和Rbed分别是Lammps固定指令（表示计算静电远场）和算法名称（表示使用RBE2D算法）；arg1=alpha， arg2=batch_size，arg3=sampling_core同(1)中rbe指令对它们的定义相同；arg4=gamma_top，arg5=gamma_down和lj/cut/coul/long/rbed中对它们的定义相同；arg6=sigma_top, arg7=sigma_down分别代表上下表面的面电荷密度，单位是e/（长度单位的平方）。

一个使用案例（假设使用200个CPU核）：

pair_style lj/cut/coul/long/rbed 10 10 0.939 -0.939

kspace_style Rbed 0.079647 200 100 0.939 -0.939 0.08 -0.08

表示使用RBE2D计算一个上下界面介电系数分别为0.939和-0.939、上下界面分别带密度为0.08和-0.08的连续面电荷的系统的静电相互作用。LJ截断半径和静电截断半径均为10，alpha选择0.079647，每次在傅里叶空间抽取200个样本，使用其中100CPU核进行采样。


3. 基于Langevin动力学提出的新NPT系综控温控压器，好处是系统收敛到平衡的速度比LAMMPS自带的“fix npt”更快，目前支持各向同性和各向异性两种控压方式。
调用方式：在Lammps的input文件中加入下面命令

fix ID group-ID baoab temp arg1 arg2 arg3 keyword arg4 arg5 arg6

fix和temp是固定指令，baoab是控压算法名称，ID是用户为这条fix指定的名称，group-ID指定了这条fix能够作用的原子组的名称，ID和group-ID同LAMMPS本身对它们的设置相同，可参考LAMMPS官方文档中的fix指令说明；arg1=Tstart，arg2=Tstop分别设定了开始和结束时的外部温度；arg3=Tdamp是控温的阻尼系数，单位和时间单位相同，一般为5倍至100倍模拟的时间步长；keyword=iso or aniso表示控压是各向同性（三个方向耦合在一起，iso）或是各向异性（三个方向分别控压，aniso）进行；arg4=Pstart，arg5=Pstop分别设定了开始和结束时的外部压强；arg6=Pdamp是控压的阻尼系数，一般为数十至数百倍模拟的时间步长。

一个使用案例（假设模拟时间步长为1fs）：

fix 2 all baoab temp 298 298 5 iso 1.0 1.0 100

表示使用Langevin动力学对所有原子做各向同性控压，开始和结束的外部温度和外部压强分别为298K和1bar，控温和控压阻尼系数分别为5fs和100fs。该fix指令的名字被设定为2。
