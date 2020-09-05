# <center>GNU</center> 

-----

GNU 通用公共许可协议（英语：GNU General Public License，缩写GNU GPL 或 GPL），是被广泛使用的自由软件许可证，给予了终端用户运行、学习、共享和修改软件的自由。许可证最初由自由软件基金会的理查德·斯托曼为 GNU 项目所撰写，并授予计算机程序的用户自由软件定义（The Free Software Definition）的权利。在本节中，GNU将代指那些基于 GNU 许可的开源软件，特别是 GCC 和 OpenMPI。

## 加载预安装的GNU

Pi2.0 系统中已经预装不同版本的 gcc，可以用 `module av gcc` 命令来查看当前所有可用的 gcc，并用以下命令加载: 

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
    在同时使用 openmpi 和 gcc 的时候，请尽量保持 gcc 版本与 openmpi 后缀中的编译器版本一致，如 gcc-8.3.0 和 openmpi-3.1.5/gcc-8.3.0

## 使用 GCC + OpenMPI 编译应用

这里，我们演示如何使用系统中的 OpenMPI 和 GCC 编译 MPI 代码，所使用的 MPI 代码可以在 `/lustre/share/samples/MPI/mpihello.c` 中找到。

加载和编译：

```bash
$ module purge; module load gcc/8.3.0-gcc-4.8.5 openmpi-3.1.5/gcc-8.3.0
$ mpicc mpihello.c -o mpihello
```

## 提交GCC+OpenMPI应用

准备一个名为 job_openmpi.slurm 的作业脚本

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

module purge
module load gcc/8.3.0-gcc-4.8.5 openmpi-3.1.5/gcc-8.3.0

srun --mpi=pmi2 ./mpihello
```

最后，将作业提交到 SLURM

```bash
$ sbatch job_openmpi.slurm
```

## 参考文献

- [Top 20 licenses](https://web.archive.org/web/20160719043600/https://www.blackducksoftware.com/top-open-source-licenses)
