.. _module:

Environment Modules软件模块
============================

ENVIRONMENT MODULES可以帮助您在Pi上使用预构建的软件包。每个ENVIRONMENT
MODULES都是可以实时应用和不应用的一组环境设置。用户可以编写自己的模块。

本文档将向您介绍ENVIRONMENT MODULES的基本用法以及Pi上可用的软件模块。

MODULES指令
-----------

======================= ================================
命令                    功能
======================= ================================
module use [PATH]       将[PATH]下的文件添加到模块列表中
module avail            列出所有模块
module load [MODULE]    加载[MODULE]
module unload [MODULE]  卸载[MODULE]
module purge            卸载所有模块
module whatis [MODULE]  显示有关[MODULE]的基本信息
module info [MODULE]    显示有关[MODULE]的详细信息
module display [MODULE] 显示有关[MODULE]的信息
module help             输出帮助信息
======================= ================================

``module use [PATH]``: 导入模块文件
-----------------------------------

``module use``\ 将在[PATH]下导入所有可用的模块文件，这些文件可在模块命令中使用。最终，\ ``[PATH]``\ 被添加到\ ``MODULEPATH``\ 环境变量的开头。

``module avail``: 列出所有模块
------------------------------

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

``module purge/load/unload/list``: 完整的模块工作流程
-----------------------------------------------------

在开始新作业之前，先卸载所有已加载的模块是一个好习惯。

.. code:: bash

   $ mdoule purge

可以一次加载或卸载多个模块。

.. code:: bash

   $ mdoule load gcc openmpi
   $ module unload gcc openmpi

您可以在工作中的任何时候检查加载的模块。

.. code:: bash

   $ module list

MODULES智能选择与Slurm
----------------------

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
--------

- Lmod: A New Environment Module System https://lmod.readthedocs.io/en/latest/   
- Environment Modules Project http://modules.sourceforge.net
- Modules Software Environment on NERSC https://www.nersc.gov/users/software/nersc-user-environment/modules/
