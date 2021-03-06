****
软件
****

π 集群有许多预建的软件模块，并且数量还在不断增长。欢迎您告诉我们您研究领域中流行的软件。由于收费很少甚至为零，因此开源软件的安装优先级更高。

软件分类
========

原子分子软件
------------

+--------------------------+-----------------+-----------------+
| Name                     | Version         | Platform        |
+==========================+=================+=================+
| :ref:`ABINIT`            | 8.10.3          | |cpu|           |
+--------------------------+-----------------+-----------------+
| :ref:`gromacs`           | 2020            | |cpu|           |
|                          |                 | |gpu| |arm|     |
+--------------------------+-----------------+-----------------+
| :ref:`lammps`            | 2020            | |cpu|           |
|                          |                 | |gpu| |arm|     |
+--------------------------+-----------------+-----------------+
| :ref:`NWchem`            | 6.8.1           | |cpu|           |
+--------------------------+-----------------+-----------------+
| :ref:`quantum-espresso`  | 6.6             | |cpu|           |
+--------------------------+-----------------+-----------------+
| :ref:`CP2k`              | 6.1             | |cpu|           |
+--------------------------+-----------------+-----------------+
| :ref:`SIESTA`            | 4.0.1           | |cpu|           |
+--------------------------+-----------------+-----------------+
| :ref:`VMD`               | 1.9.4           | |cpu| |studio|  |
+--------------------------+-----------------+-----------------+
| :ref:`OVITO`             | 3.2.1           | |cpu| |studio|  |
+--------------------------+-----------------+-----------------+
|      Paraview            | 0.4.1           | |cpu| |studio|  |
+--------------------------+-----------------+-----------------+


工程计算软件
-------------------

+------------------------+-----------------+-----------------+
| Name                   | Version         | Platform        |
+========================+=================+=================+
| :ref:`CESM`            | 1.2             | |cpu|           |
+------------------------+-----------------+-----------------+
| :ref:`Nektar`          | 5.0.0           | |cpu|           |
+------------------------+-----------------+-----------------+
| :ref:`Octave`          | 5.2.0           | |cpu| |studio|  |
+------------------------+-----------------+-----------------+
| :ref:`OpenFOAM`        | 7, 1712, 1812,  | |cpu|           |
|                        | 1912            |                 |
+------------------------+-----------------+-----------------+
|       STAR             | 2.7.0           | |cpu|           |
+------------------------+-----------------+-----------------+
| :ref:`star-ccm`        |                 | |cpu|           |
+------------------------+-----------------+-----------------+



AI 计算软件
-----------

+------------------------+-----------------+-----------------+
| Name                   | Version         | Platform        |
+========================+=================+=================+
| :ref:`PyTorch`         | 1.6.0           | |gpu|           |
+------------------------+-----------------+-----------------+
| :ref:`TensorFlow`      | 2.0.0           | |gpu|           |
+------------------------+-----------------+-----------------+
| :doc:`deepvariant`     | 10.0.130        | |cpu| |gpu|     |
+------------------------+-----------------+-----------------+
| :ref:`Keras`           |                 |                 |
+------------------------+-----------------+-----------------+


生信计算软件
-------------------
+------------------------+-----------------+-----------------+
| Name                   | Version         | Platform        |
+========================+=================+=================+
|       Bcftools         | 1.9.3           | |cpu|           |
+------------------------+-----------------+-----------------+
|       Bedtools2        | 2.27.1          | |cpu|           |
+------------------------+-----------------+-----------------+
|       Bismark          | 0.19.0          | |cpu|           |
+------------------------+-----------------+-----------------+
|       Bowtie           | 1.2.3           | |cpu|           |
+------------------------+-----------------+-----------------+
|       Bwa              | 0.7.17          | |cpu|           |
+------------------------+-----------------+-----------------+
|       Cdo              | 1.9.8           | |cpu|           |
+------------------------+-----------------+-----------------+
|       Cufflinks        | 2.2.1           | |cpu|           |
+------------------------+-----------------+-----------------+
| :doc:`deepvariant`     | 10.0.130        | |cpu| |gpu|     |
+------------------------+-----------------+-----------------+
|        Fastqc          | 0.11.7          | |cpu|           |
+------------------------+-----------------+-----------------+
|       Gatk             | 3.8             | |cpu|           |
+------------------------+-----------------+-----------------+
|       Geant4           | 10.6.2          | |cpu|           |
+------------------------+-----------------+-----------------+
|       Gmap-gsnap       | 2019-5-12       | |cpu|           |
+------------------------+-----------------+-----------------+
|       Graphmap         | 0.3.0           | |cpu|           |
+------------------------+-----------------+-----------------+
|       Hisat2           | 2.1.0           | |cpu|           |
+------------------------+-----------------+-----------------+
|       Lumpy-sv         | 0.2.13          | |cpu|           |
+------------------------+-----------------+-----------------+
|       Megahit          | 1.1.4           | |cpu|           |
+------------------------+-----------------+-----------------+
|       Metis            | 5.1.0           | |cpu|           |
+------------------------+-----------------+-----------------+
| :ref:`Mrbayes`         | 3.2.7a          | |cpu|           |
+------------------------+-----------------+-----------------+
|       Ncbi-rmblastn    | 2.2.28          | |cpu|           |
+------------------------+-----------------+-----------------+
|       Picard           | 2.19.0          | |cpu|           |
+------------------------+-----------------+-----------------+
| :ref:`R`               | 1.1.8, 3.6.2    | |cpu| |studio|  |
+------------------------+-----------------+-----------------+
| :ref:`Relion`          | 3.0.8           | |gpu| |studio|  |
+------------------------+-----------------+-----------------+
|       Rna-seqc         | 1.1.8           | |cpu|           |
+------------------------+-----------------+-----------------+
|       Salmon           | 0.14.1          | |cpu|           |
+------------------------+-----------------+-----------------+
|       SAMtools         | 1.9             | |cpu|           |
+------------------------+-----------------+-----------------+
|       SOAPdenovo2      | 240             | |cpu|           |
+------------------------+-----------------+-----------------+
|       SRAtoolkit       | 2.9.6           | |cpu|           |
+------------------------+-----------------+-----------------+
|       StringTie        | 1.3.4d          | |cpu|           |
+------------------------+-----------------+-----------------+
| :ref:`STRique`         |                 | |cpu|           |
+------------------------+-----------------+-----------------+
|       TopHat           | 2.1.2           | |cpu|           |
+------------------------+-----------------+-----------------+
|       VarDictJava      | 1.5.1           | |cpu|           |
+------------------------+-----------------+-----------------+
|       VSEARCH          | 2.4.3           | |cpu|           |
+------------------------+-----------------+-----------------+

编译器和库
==========

编译器

=============== =================== ========================= ========
模块名字        描述                提供版本                  默认版本
=============== =================== ========================= ========
:ref:`gnu`      GNU编译器集合       5.5, 7.4, 8.3, 9.2, 9.3    9.3
:ref:`intel`    Intel编译器套件     19.0.4, 19.0.5, 19.1.1     19.1.1
:doc:`cuda`     NVIDIA CUDA SDK     9.0, 10.0, 10.1, 10.2      10.2
:doc:`hpcsdk`   NVIDIA HPC SDK      20.11                      20.11
jdk             Java开发套件        12.0                       12.0
=============== =================== ========================= ========

MPI库

========= ========= ======== ========
模块名字  描述      提供版本 默认版本
========= ========= ======== ========
openmpi   OpenMPI   3.1.5    3.1.5
intel-mpi Intel MPI 2019.4   2019.4
========= ========= ======== ========

数学库

+-----------+---------------------+----------+----------+----------------------------+
| 模块名字  | 描述                | 提供版本 | 默认版本 | 备注                       |
+===========+=====================+==========+==========+============================+
| intel-mkl | Intel数学核心函数库 | 19.3     | 19.3     | 包含FFTW，BLAS，LAPACK实现 |
+-----------+---------------------+----------+----------+----------------------------+

软件使用
================

ENVIRONMENT MODULES可以帮助您在 π 集群上使用预构建的软件包。每个ENVIRONMENT
MODULES都是可以实时应用和不应用的一组环境设置。用户可以编写自己的模块。

本文档将向您介绍ENVIRONMENT MODULES的基本用法以及 π 集群上可用的软件模块。

调用 module
-------------------

======================= ================================
命令                    功能
======================= ================================
module use [PATH]       将[PATH]下的文件添加到模块列表中
module avail            列出所有模块
module load [MODULE]    加载[MODULE]
module unload [MODULE]  卸载[MODULE]
module whatis [MODULE]  显示有关[MODULE]的基本信息
module info [MODULE]    显示有关[MODULE]的详细信息
module display [MODULE] 显示有关[MODULE]的信息
module help             输出帮助信息
======================= ================================

``module use [PATH]``: 导入模块文件

``module use``\ 将在[PATH]下导入所有可用的模块文件，这些文件可在模块命令中使用。最终，\ ``[PATH]``\ 被添加到\ ``MODULEPATH``\ 环境变量的开头。

``module avail``: 列出所有模块

::

   ----------------------------------------------------- /lustre/share/spack/modules/cascadelake/linux-centos7-x86_64 -----------------------------------------------------
   bismark/0.19.0-intel-19.0.4                    intel/19.0.4-gcc-4.8.5                         openblas/0.3.6-intel-19.0.4
   bowtie2/2.3.5-intel-19.0.4                     intel-mkl/2019.3.199-intel-19.0.4              openmpi/3.1.4-gcc-4.8.5
   bwa/0.7.17-intel-19.0.4                        intel-mpi/2019.4.243-intel-19.0.4              openmpi/3.1.4-intel-19.0.4
   cuda/10.0.130-gcc-4.8.5                        intel-parallel-studio/cluster.2018.4-intel-18.0.4 perl/5.26.2-gcc-4.8.5
   cuda/10.0.130-intel-19.0.4                     intel-parallel-studio/cluster.2019.5-intel-19.0.5 perl/5.26.2-intel-19.0.4
   cuda/9.0.176-intel-19.0.4                      jdk/11.0.2_9-intel-19.0.4                      perl/5.30.0-gcc-4.8.5
   emacs/26.1-gcc-4.8.5                           miniconda2/4.6.14-gcc-4.8.5                    soapdenovo2/240-gcc-4.8.5
   gcc/5.5.0-gcc-4.8.5                            miniconda2/4.6.14-intel-19.0.4                 stream/5.10-intel-19.0.4
   gcc/8.3.0-gcc-4.8.5                            miniconda3/4.6.14-gcc-4.8.5                    tophat/2.1.2-intel-19.0.4
   gcc/9.2.0-gcc-4.8.5                            miniconda3/4.6.14-intel-19.0.4
   hisat2/2.1.0-intel-19.0.4                      ncbi-rmblastn/2.2.28-gcc-4.8.5

``module load/unload/list``: 完整的模块工作流程

可以一次加载或卸载多个模块。

.. code:: bash

   $ mdoule load gcc openmpi
   $ module unload gcc openmpi

您可以在工作中的任何时候检查加载的模块。

.. code:: bash

   $ module list

MODULES智能选择与Slurm


在SLURM上，我们应用了以下规则来选取最合适的模块。

1. 编译器：如果加载了\ ``gcc``\ 或\ ``icc``\ ，请根据相应的编译器加载已编译的模块。或在必要时加载默认的编译器\ ``gcc``\ 。
2. MPI库：如果已加载其中一个库（\ ``openmpi``\ ，\ ``impi``\ ，\ ``mvapich2``\ ，\ ``mpich``\ ），加载针对相应MPI编译的模块。在必要的时候,默认MPI
   lib中的\ ``openmpi``\ 将被装载。
3. Module版本：每个模块均有默认版本，如果未指定版本号，则将加载该默认版本。

在SLURM上，以下子句与上面的子句具有相同的作用。

.. code:: bash

   $ module load gcc/9.2.0-gcc-4.8.5 openmpi/3.1

或者，如果您喜欢最新的稳定版本，则可以忽略版本号。

.. code:: bash

   $ module load gcc openmpi

参考资料

- `Lmod: A New Environment Module System <https://lmod.readthedocs.io/en/latest/>`__   
- `Environment Modules Project <http://modules.sourceforge.net/>`__
- `Modules Software Environment on NERSC <https://www.nersc.gov/users/software/nersc-user-environment/modules/>`__

conda 安装软件
-------------------

下面介绍使用 Conda 在个人目录中安装生物信息类应用软件。

用 Conda 安装软件的流程

0. 安装之前，先申请计算节点资源（登陆节点禁止大规模编译安装）

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash

1. 加载 Miniconda3

.. code:: bash

   $ module load miniconda3

2. 创建 conda 环境来安装所需 Python 包（可指定 Python 版本，也可以不指定）

.. code:: bash

   $ conda create --name mypy

3. 激活 python 环境

.. code:: bash

   $ source activate mypy

4. 通过 conda 安装软件包（有些软件也可以用 pip 安装。软件官网一般给出推荐，用 conda 还是 pip）

许多生信软件可以在 anaconda 的 `bioconda package <https://anaconda.org/bioconda/>`__ 里找到：

.. code:: bash

   $ conda install -c bioconda openslide-python （以 openslide-python 为例）

conda 安装的软件详细列表见 `生信软件安装 <bio.html>`_

下面以 numpy 为例，展示 conda 安装方法：

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install numpy

软件安装完毕。

下次调用 numpy，仅需使用如下语句：

.. code:: bash

   module load miniconda3
   source activate mypy


以下示例 slurm 提交作业：

slurm 脚本示例：申请 small 队列的 2 个核，通过 python 打印
``hello world``

.. code:: bash

   #!/bin/bash
   #SBATCH -J py_test
   #SBATCH -p small
   #SBATCH -n 2
   #SBATCH --ntasks-per-node=2
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load miniconda3
   source activate mypy

   python -c "print('hello world')"

我们假定以上脚本内容被写到了 ``hello_python.slurm`` 中，使用 ``sbatch``
指令提交作业

.. code:: bash

   $ sbatch hello_python.slurm



具体软件页面
============

-  `abinit <abinit.html>`__
-  `cesm <cesm.html>`__
-  `cp2k <cp2k.html>`__
-  `cuda <cuda.html>`__
-  `gnu <gnu.html>`__
-  `gnuplot <gnuplot.html>`__
-  `gromacs <gromacs.html>`__
-  `hpcsdk <hpcsdk.html>`__
-  `intel <intel.html>`__
-  `keras <keras.html>`__
-  `lammps <lammps.html>`__
-  `mrbayes <mrbayes.html>`__
-  `nektar <nektar.html>`__
-  `nwchem <nwchem.html>`__
-  `openfoam <openfoam.html>`__
-  `ovito <ovito.html>`__
-  `octave <octave.html>`__
-  `pytorch <pytorch.html>`__
-  `perl <perl.html>`__
-  `python <python.html>`__
-  `quantum-espresso <quantum-espresso.html>`__
-  `relion <relion.html>`__
-  `siesta <siesta.html>`__
-  `star-ccm <star-ccm.html>`__
-  `strique <strique.html>`__
-  `r <r.html>`__
-  `tensorflow <tensorflow.html>`__
-  `vmd <vmd.html>`__


.. toctree::
   :maxdepth: 1
   :hidden:
   
   bio
   abinit
   cesm
   cp2k
   cuda
   deepvariant
   gnu
   gnuplot
   gromacs
   hpcsdk
   intel
   keras
   lammps
   mrbayes
   nektar
   nwchem
   openfoam
   ovito
   octave
   pytorch
   python
   perl
   quantum-espresso
   relion
   r
   siesta
   star-ccm
   strique
   tensorflow
   vmd

   
