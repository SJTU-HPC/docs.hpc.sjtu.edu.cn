# <center>GNU</center> 

-----

GNU通用公共许可协议（英语：GNU General Public License，缩写GNU GPL 或 GPL），是被广泛使用的自由软件许可证，给予了终端用户运行、学习、共享和修改软件的自由。许可证最初由自由软件基金会的理查德·斯托曼为GNU项目所撰写，并授予计算机程序的用户自由软件定义（The Free Software Definition）的权利。在本节中，GNU将代指那些基于GNU许可的开源软件，特别是GCC和OpenMPI。

## 加载预安装的GNU

Pi2.0 系统中已经预装不同版本的gcc，可以用以下命令加载: 

| 版本 | 加载方式 |
| ---- | ------ |
| gcc-5.5.0   | module load gcc/5.5.0-gcc-4.8.5 |
| gcc-7.4.0 | module load gcc/7.4.0-gcc-4.8.5 |
| gcc-8.3.0 | module load gcc/8.3.0-gcc-4.8.5 |
| gcc-9.2.0 | module load gcc/9.2.0-gcc-4.8.5 | 

不同版本的openmpi，可以用以下命令加载：

| 版本 | 加载方式 |
| ---- | ------ |
| openmpi-3.1.5/gcc-4.8.5   | module load openmpi/3.1.5-gcc-4.8.5 |
| openmpi-3.1.5/gcc-7.4.0 | module load openmpi/3.1.5-gcc-7.4.0 | 
| openmpi-3.1.5/gcc-8.3.0 | module load openmpi/3.1.5-gcc-8.3.0 |
| openmpi-3.1.5/gcc-9.2.0 | module load openmpi/3.1.5-gcc-9.2.0 |

!!! tip
    在同时使用openmpi和gcc的时候，请尽量保持gcc版本与openmpi后缀中的编译器版本一致，如gcc-8.3.0和openmpi-3.1.5/gcc-8.3.0

## 使用GCC+OpenMPI编译应用

这里，我们演示如何使用系统中的OpenMPI和GCC编译MPI代码，所使用的MPI代码可以在`/lustre/share/samples/MPI/mpihello.c`中找到。

加载和编译：

```bash
$ module purge; module load gcc/8.3.0-gcc-4.8.5 openmpi-3.1.5/gcc-8.3.0
$ mpicc mpihello.c -o mpihello
```

## 提交GCC+OpenMPI应用

准备一个名为job_openmpi.slurm的作业脚本

```bash
#!/bin/bash

#SBATCH --job-name=mpihello
#SBATCH --partition=cpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -n 80
#SBATCH --ntasks-per-node=40

ulimit -s unlimited
ulimit -l unlimited

source /usr/share/Modules/init/bash
module purge
module load gcc/8.3.0-gcc-4.8.5 openmpi-3.1.5/gcc-8.3.0

srun --mpi=pmi2 ./mpihello
```

最后，将作业提交到SLURM

```bash
$ sbatch job_openmpi.slurm
```

## 参考文献

- [Top 20 licenses](https://web.archive.org/web/20160719043600/https://www.blackducksoftware.com/top-open-source-licenses)