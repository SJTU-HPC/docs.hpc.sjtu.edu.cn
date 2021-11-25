.. _lammps-rbe:

LAMMPS-RBE
==========

简介
----

LAMMPS-RBE是由上海交通大学自然科学研究院金石、徐振礼、洪亮三位教授团队联合开发的基于LAMMPS二次开发的自研软件。该版本在长程力模拟中引入了先进的Random Batch Ewald算法，RBE使用基于动理学或连续介质理论的路径, 研究复杂环境中微纳系统的多体效应，并结合分子动力学进行多尺度建模和数学分析。LAMMPS-RBE突破了传统分子动力学在 CPU集群上可扩展性差的问题，可以使百万级别粒子以上的大尺度体系的计算成本降低一个数量级。

π 集群上的 LAMMPS-RBE
----------------------

- `CPU版本 LAMMPS-RBE`_

.. _CPU版本 LAMMPS-RBE:

CPU版本
~~~~~~~

同Lammps已有功能相比，该版本新增三个功能：

(1) 基于Random Batch Ewald (RBE)算法的三维周期/二维准周期平板系统静电求解器，特别适用于多核模拟；
调用方式：在Lammps的input文件中加入下面命令（需和pair/lj/cut/coul/long配合使用，这点和PPPM算法相同），
kspace_style rbe arg1 arg2 arg3
其中kspace_style是Lammps固定指令，表示模拟中要计算静电相互作用；rbe是算法名称表示调用RBE算法计算静电；
arg1=alpha，是RBE算法里用于控制近远场比例的参数，该参数的选择和Ewald以及PPPM算法相同。如果希望相对误差是1e-4，那么需选取使得erfc(r_cut*sqrt{alpha})≈1e-4的alpha, 其中r_cut是在pair/lj/cut/coul/long中选取的静电近场截断；arg2=batch_size，是在傅里叶空间中做随机采样得到的样本数量，一般为几十至数百（越大越准确，越小计算速度越快）；arg3=sampling_core，用于采样的CPU核的数量，需>1且<总MPI数量，一般可选取和用户使用的MPI数量相同或MPI数量一半。
两个使用案例（假设使用200个CPU核）：
pair_style      lj/cut/coul/long 10.0 10.0
kspace_style    rbe 0.07 500 100
或调用intel的近场计算
pair_style      lj/cut/coul/long/intel 12.0 12.0
kspace_style    rbe 0.05 200 100
如果希望处理二维周期且z方向是两块平板的系统，需要在input文件中定义平板的位置参数和kspace_modify slab 3,方法同LAMMPS官方文档中用PPPM算平板问题的方式一致。
