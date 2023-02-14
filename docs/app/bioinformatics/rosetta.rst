.. _rosetta:

Rosetta
=======

简介
----

Rosetta软件套件包括用于蛋白质结构计算建模和分析的算法。它使计算生物学取得了显著的科学进步，包括从头蛋白质设计、酶设计、配体对接以及生物大分子和大分子复合物的结构预测。

RoeTeta开发始于华盛顿大学David Baker博士的实验室，作为结构预测工具，但此后已经适应于解决常见的计算大分子问题。

Rosetta的重要作用如下所示：

理解大分子相互作用; 设计定制分子; 寻找构象和序列空间的有效方法; 为各种生物分子表示寻找广泛有用的能量函数

可用的版本
-------------------

+------+-------+----------+-----------------------+
| 版本 | 平台  | 构建方式 | 模块名                |
+======+=======+==========+=======================+
| 3.12 | |cpu| | 容器     | rosetta/3.12 思源一号 |
+------+-------+----------+-----------------------+
| 3.12 | |cpu| | 容器     | rosetta/3.12          |
+------+-------+----------+-----------------------+

算例下载
---------

.. code:: bash

   思源一号：
   mkdir ~/test_rosetta
   cd ~/test_rosetta
   cp -r /dssg/share/sample/rosetta/input_files ./
   mkdir output_files
   
   π2.0：
   mkdir ~/test_rosetta
   cd ~/test_rosetta
   cp -r /lustre/share/samples/rosetta/input_files ./
   mkdir output_files
   

集群上的Rosetta
----------------

- `一. 思源一号 Rosetta`_

- `二. π2.0 Rosetta`_

.. _一. 思源一号 Rosetta:

一. 思源一号 Rosetta
--------------------

1. 对输入结构进行预处理（refine） 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load rosetta/3.12 
   mpirun relax.mpi.linuxgccrelease -in:file:s input_files/1qys.pdb -nstruct 2 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false

输入与参数说明

::

   in:file:s                              #输入数据
   nstruct                                #nstruct可以提高模型结果的质量，如nstruct 10将会获得10个模型
   relax:constrain_relax_to_start_coords  #约束重原子，从而使得骨架较初始不会移动太多
   relax:ramp_constraints                 #设为false则不进行倾斜约束（进行整体约束该选项需要设置为false）
   use_input_sc                           #turns on inclusion of the current rotamer for the packer
   flip_HNQ                               #在氢键原子位置优化期间考虑翻转HIS，ASN，GLN
   no_optH                                #是否在PDB加载期间进行氢原子位置优化

2. 局部对接
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2.1 局部对接
""""""""""""""""""""""""""""""
.. code:: bash
  
   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load rosetta/3.12 
   mpirun docking_protocol.mpi.linuxgccrelease -in:file:s input_files/col_complex.pdb -in:file:native input_files/1v74.pdb -nstruct 1 -partners A_B -dock_pert 3 8 -ex1 -ex2aro -out:path:all output_files -out:suffix _local_dock

输入与参数说明

::

   in:file:s      #输入数据
   in:file:native #native file，与该文件进行计算比较
   nstruct        #请注意在进行实际数据分析时，此处的值应当至少为500
   partners       #partners A_B意味着，链B对接进入链A
   dock_pert      #dock_pert 3 8意味着，在开始单独的模拟之前随机的将配体（链B）进行一个3埃的平移和8度的旋转
   out:path:all   #输出路径
   out:suffix     #输出文件名后缀

2.2 对得到的对接结果进行局部优化 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load rosetta/3.12
   mpirun docking_protocol.mpi.linuxgccrelease -in:file:s input_files/1v74.pdb -nstruct 1 -docking_local_refine -use_input_sc -ex1 -ex2aro -out:file:fullatom -out:path:all output_files -out:suffix _local_refine

3. 全局对接
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

若没有蛋白结合位点的信息，则使用全局对接。全局对接假设蛋白质为球型，而更小的蛋白质配体围绕蛋白质受体。全局对接对小复合物相对较好（残基数小于450）

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load rosetta/3.12
   mpirun docking_protocol.mpi.linuxgccrelease -in:file:s input_files/col_complex.pdb -in:file:native input_files/1v74.pdb -unboundrot input_files/col_complex.pdb -nstruct 1 -partners A_B -dock_pert 3 8 -spin -randomize1 -randomize2 -ex1 -ex2aro -out:path:all output_files -out:suffix _global_dock

输入与参数说明

::

   unboundrot  #将指定结构的旋转异构体添加到旋转异构体库中
   nstruct     #请注意在进行实际数据分析时，此处的值应当为 10,000~100,000

4. Flexible Protein对接 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rosetta假设蛋白骨架为柔性的进行对接。Rosetta假设蛋白-蛋白结合过程前后构象发生了较大的变化，并对蛋白构象簇（ensembles）进行对接，而非一个配体构象和一个受体构象。

4.1 prepack 
""""""""""""""""""""""""""""""

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load rosetta/3.12
   ls input_files/COL_D_ensemble/*.pdb > COL_D_ensemblelist
   ls input_files/IMM_D_ensemble/*.pdb > IMM_D_ensemblelist
   mpirun docking_prepack_protocol.mpi.linuxgccrelease -in:file:s input_files/col_complex.pdb -in:file:native input_files/1v74.pdb -unboundrot input_files/col_complex.pdb -nstruct 1 -partners A_B -ensemble1 COL_D_ensemblelist -ensemble2 IMM_D_ensemblelist -ex1 -ex2aro -out:path:all output_files -out:suffix _ensemble_dock
   
4.2 柔性对接
""""""""""""""""""""""""""""""

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load rosetta/3.12
   mpirun docking_prepack_protocol.mpi.linuxgccrelease -in:file:s input_files/col_complex.pdb -in:file:native input_files/1v74.pdb -unboundrot input_files/col_complex.pdb -nstruct 1 -partners A_B -dock_pert 3 8 -ensemble1 COL_D_ensemblelist -ensemble2 IMM_D_ensemblelist -ex1 -ex2aro -out:path:all output_files -out:suffix _ensemble_dock

.. _π2.0 Rosetta:

二. π2.0 Rosetta
------------------------------------------

申请计算节点并导入rosetta软件

1. 对输入结构进行预处理（refine） _π2.0_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=small
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load rosetta/3.12
   mpirun relax.mpi.linuxgccrelease -in:file:s input_files/1qys.pdb -nstruct 2 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false

输入与参数说明

::

   in:file:s                              #输入数据
   nstruct                                #nstruct可以提高模型结果的质量，如nstruct 10将会获得10个模型
   relax:constrain_relax_to_start_coords  #约束重原子，从而使得骨架较初始不会移动太多
   relax:ramp_constraints                 #设为false则不进行倾斜约束（进行整体约束该选项需要设置为false）
   use_input_sc                           #turns on inclusion of the current rotamer for the packer
   flip_HNQ                               #在氢键原子位置优化期间考虑翻转HIS，ASN，GLN
   no_optH                                #是否在PDB加载期间进行氢原子位置优化

2. 局部对接  _π2.0_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2.1 局部对接 _π2.0_
""""""""""""""""""""""""""""""
.. code:: bash
   
   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=small
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load rosetta/3.12
   mpirun docking_protocol.mpi.linuxgccrelease -in:file:s input_files/col_complex.pdb -in:file:native input_files/1v74.pdb -nstruct 1 -partners A_B -dock_pert 3 8 -ex1 -ex2aro -out:path:all output_files -out:suffix _local_dock

输入与参数说明

::

   in:file:s      #输入数据
   in:file:native #native file，与该文件进行计算比较
   nstruct        #请注意在进行实际数据分析时，此处的值应当至少为500
   partners       #partners A_B意味着，链B对接进入链A
   dock_pert      #dock_pert 3 8意味着，在开始单独的模拟之前随机的将配体（链B）进行一个3埃的平移和8度的旋转
   out:path:all   #输出路径
   out:suffix     #输出文件名后缀

2.2 对得到的对接结果进行局部优化  _π2.0_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=small
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load rosetta/3.12
   mpirun docking_protocol.mpi.linuxgccrelease -in:file:s input_files/1v74.pdb -nstruct 1 -docking_local_refine -use_input_sc -ex1 -ex2aro -out:file:fullatom -out:path:all output_files -out:suffix _local_refine

3. 全局对接 _π2.0_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

若没有蛋白结合位点的信息，则使用全局对接。全局对接假设蛋白质为球型，而更小的蛋白质配体围绕蛋白质受体。全局对接对小复合物相对较好（残基数小于450）

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=small
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load rosetta/3.12
   mpirun docking_protocol.mpi.linuxgccrelease -in:file:s input_files/col_complex.pdb -in:file:native input_files/1v74.pdb -unboundrot input_files/col_complex.pdb -nstruct 1 -partners A_B -dock_pert 3 8 -spin -randomize1 -randomize2 -ex1 -ex2aro -out:path:all output_files -out:suffix _global_dock

输入与参数说明

::

   unboundrot  #将指定结构的旋转异构体添加到旋转异构体库中
   nstruct     #请注意在进行实际数据分析时，此处的值应当为 10,000~100,000

4. Flexible Protein对接  _π2.0_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rosetta假设蛋白骨架为柔性的进行对接。Rosetta假设蛋白-蛋白结合过程前后构象发生了较大的变化，并对蛋白构象簇（ensembles）进行对接，而非一个配体构象和一个受体构象。

4.1 prepack  _π2.0_
""""""""""""""""""""""""""""""

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=small
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load rosetta/3.12
   ls input_files/COL_D_ensemble/*.pdb > COL_D_ensemblelist
   ls input_files/IMM_D_ensemble/*.pdb > IMM_D_ensemblelist
   mpirun docking_prepack_protocol.mpi.linuxgccrelease -in:file:s input_files/col_complex.pdb -in:file:native input_files/1v74.pdb -unboundrot input_files/col_complex.pdb -nstruct 1 -partners A_B -ensemble1 COL_D_ensemblelist -ensemble2 IMM_D_ensemblelist -ex1 -ex2aro -out:path:all output_files -out:suffix _ensemble_dock
   
4.2 柔性对接  _π2.0_
""""""""""""""""""""""""""""""

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=rosetta
   #SBATCH --partition=small
   #SBATCH -N 1 
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load rosetta/3.12
   mpirun docking_prepack_protocol.mpi.linuxgccrelease -in:file:s input_files/col_complex.pdb -in:file:native input_files/1v74.pdb -unboundrot input_files/col_complex.pdb -nstruct 1 -partners A_B -dock_pert 3 8 -ensemble1 COL_D_ensemblelist -ensemble2 IMM_D_ensemblelist -ex1 -ex2aro -out:path:all output_files -out:suffix _ensemble_dock

运行结果
----------------

思源一号上的运行结果

.. code:: bash

   output_files/
   ├── 1v74_local_refine_0001.pdb
   ├─ col_complex_ensemble_dock_0001.pdb
   ├── col_complex_global_dock_0001.pdb
   ├── col_complex_local_dock_0001.pdb
   ├── score_ensemble_dock.sc
   ├── score_global_dock.sc
   ├── score_local_dock.sc
   └── score_local_refine.fasc

π2.0上的运行结果

.. code:: bash

   output_files/
   ├── 1v74_local_refine_0001.pdb
   ├─ col_complex_ensemble_dock_0001.pdb
   ├── col_complex_global_dock_0001.pdb
   ├── col_complex_local_dock_0001.pdb
   ├── score_ensemble_dock.sc
   ├── score_global_dock.sc
   ├── score_local_dock.sc
   └── score_local_refine.f

参考资料
----------------

- Rosetta:  https://www.rosettacommons.org/
