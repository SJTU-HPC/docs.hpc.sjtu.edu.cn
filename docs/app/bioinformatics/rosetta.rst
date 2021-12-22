.. _rosetta:

Rosetta
=======

简介
----

Rosetta软件套件包括用于蛋白质结构计算建模和分析的算法。它使计算生物学取得了显著的科学进步，包括从头蛋白质设计、酶设计、配体对接以及生物大分子和大分子复合物的结构预测。

RoeTeta开发始于华盛顿大学David Baker博士的实验室，作为结构预测工具，但此后已经适应于解决常见的计算大分子问题。

Rosetta的重要作用如下所示：

理解大分子相互作用; 设计定制分子; 寻找构象和序列空间的有效方法; 为各种生物分子表示寻找广泛有用的能量函数

使用Rosetta的流程如下
---------------------

申请计算节点并导入rosetta软件

.. code:: bash

   srun -p small -n 2 --pty /bin/bash
   module load rosetta/3.12

1. 对输入结构进行预处理（refine）

.. code:: bash

   relax.mpi.linuxgccrelease -``in``:file:s input_files/from_rcsb/1qys.pdb @flag_input_relax

2. local dock

.. code:: bash
   
   docking_protocol.mpi.linuxgccrelease @flag_local_docking

3. 对得到的对接结果进行local refine

.. code:: bash

   docking_protocol.mpi.linuxgccrelease @flag_local_refine

4. global dock

.. code:: bash

   docking_protocol.mpi.linuxgccrelease @flag_global_docking

5. Docking flexible proteins

.. code:: bash

   docking_prepack_protocol.mpi.linuxgccrelease @flag_ensemble_prepack

prepack运行后，就可以执行柔性对接了，对接命令为：

.. code:: bash

   docking_protocol.mpi.linuxgccrelease @flag_ensemble_docking

参考资料
----------------

- Rosetta:https://www.rosettacommons.org/ 
