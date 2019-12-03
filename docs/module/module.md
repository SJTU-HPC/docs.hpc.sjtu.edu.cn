# <center>ENVIRONMENT MODULES</center>

-----------

ENVIRONMENT MODULES可以帮助您在Pi上使用预构建的软件包。每个ENVIRONMENT MODULES都是可以实时应用和不应用的一组环境设置。用户可以编写自己的模块。

本文档将向您介绍ENVIRONMENT MODULES的基本用法以及Pi上可用的软件模块。

## MODULES指令

| 命令 | 功能 |
| ---- | ---- |
| module use [PATH] | 将[PATH]下的文件添加到模块列表中 |
| module avail | 列出所有模块 |
| module load [MODULE] | 加载[MODULE] | 
| module unload [MODULE] | 卸载[MODULE] |
| module purge | 卸载所有模块 |
| module whatis  [MODULE] | 显示有关[MODULE]的基本信息 |
| module info [MODULE] | 显示有关[MODULE]的详细信息 |
| module display [MODULE] | 显示有关[MODULE]的信息 |
| module help | 输出帮助信息 |

## `module use [PATH]`: 导入模块文件

`module use`将在[PATH]下导入所有可用的模块文件，这些文件可在模块命令中使用。最终，`[PATH]`被添加到`MODULEPATH`环境变量的开头。

## `module avail`: 列出所有模块

```
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
```

## `module purge/load/unload/list`: 完整的模块工作流程

在开始新作业之前，先卸载所有已加载的模块是一个好习惯。

```bash
$ mdoule purge
```

可以一次加载或卸载多个模块。

```bash
$ mdoule load gcc openmpi
$ module unload gcc openmpi
```

您可以在工作中的任何时候检查加载的模块。

```bash
$ module list
```

## MODULES智能选择与Slurm

在SLURM上，我们应用了以下规则来选取最合适的模块。

1. 编译器：如果加载了`gcc`或`icc`，请根据相应的编译器加载已编译的模块。或在必要时加载默认的编译器`gcc`。
2. MPI库：如果已加载其中一个库（`openmpi`，`impi`，`mvapich2`，`mpich`），加载针对相应MPI编译的模块。在必要的时候,默认MPI lib中的`openmpi`将被装载。
3. Module版本：每个模块均有默认版本，如果未指定版本号，则将加载该默认版本。

在SLURM上，以下子句与上面的子句具有相同的作用。

```bash
$ module load gcc/9.2.0-gcc-4.8.5 openmpi/3.1
```

或者，如果您喜欢最新的稳定版本，则可以忽略版本号。

```bash
$ module load gcc openmpi
```

## PI上的软件模块

Pi有许多预建的软件模块，并且数量还在不断增长。欢迎您告诉我们您研究领域中流行的软件。由于收费很少甚至为零，因此开源软件的安装优先级更高。

Pi上的软件可以分类为编译器和平台，MPI库，Math库，FD工具，生物信息学工具等。

## 编译器和平台

| 模块名字 | 描述 | 提供版本 | 默认版本 | 
| ---- | ---- | ---- | ---- |
| gcc | GNU编译器集合 | 5.5 8.3 9.2 | 9.2 |
| intel | Intel编译器套件 | 19.0.4 | 19.0.4 | 
| pgi | PGI编译器 | 19.4 | 19.4 |
| cuda | NVIDIA CUDA SDK | 10.0 9.0 | 10.0 |
| jdk | Java开发套件 | 11.0 | 11.0 | 

## MPI库

| 模块名字 | 描述 | 提供版本 | 默认版本 | 
| ---- | ---- | ---- | ---- |
| openmpi | OpenMPI | 3.1.4 | 3.1.4 |
| intel-mpi | Intel MPI | 2019.4 | 2019.4 |

## 数学库

| 模块名字 | 描述 | 提供版本 | 默认版本 | 备注 |
| ---- | ---- | ---- | ---- | ---- |
| intel-mkl | Intel数学核心函数库 | 19.3 | 19.3 | 包含FFTW，BLAS，LAPACK实现 |

## 计算机视觉与深度学习

| 模块名字 | 描述 | 提供版本 | 默认版本 | GPU支持 |
| ---- | ---- | ---- | ---- | ---- |
| cudnn | NVIDIA深度学习GPU加速原语库 | 7.3 7.4 7.5 | 7.5 | Yes |

<!-- ## 用于构建和调整软件的工具

| 模块名字 | 描述 | 提供版本 | 默认版本 | 
| ---- | ---- | ---- | ---- |
| maven | 软件项目管理工具 | 3.3 | 3.3 |
| bazel | 软件构建工具 | 0.1 | 0.1 |
| vtune | Intel vtune | 5.1 | 5.1 | -->

## 参考链接

 - [Environment Modules Project](http://modules.sourceforge.net/)
 - [Modules Software Environment on NERSC](https://www.nersc.gov/users/software/nersc-user-environment/modules/)
