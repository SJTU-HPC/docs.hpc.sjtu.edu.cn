# <center>GROMACS</center>

---------

## 简介

GROMACS 是一种分子动力学应用程序，可以模拟具有数百至数百万个粒子的系统的牛顿运动方程。GROMACS旨在模拟具有许多复杂键合相互作用的生化分子，例如蛋白质，脂质和核酸。

## Pi 上的 GROMACS

Pi 上有多种版本的 GROMACS:

- ![cpu](https://img.shields.io/badge/-cpu-blue)  [cpu](#cpu-gromacs)

- ![gpu](https://img.shields.io/badge/-gpu-green) [gpu](#gpu-gromacs)

- ![arm](https://img.shields.io/badge/-arm-yellow) [arm](#arm-gromacs)

## ![cpu](https://img.shields.io/badge/-cpu-blue) (CPU) GROMACS 模块调用

查看 Pi 上已编译的软件模块:
```bash
$ module avail gromacs
```

调用该模块:
```bash
$ module load gromacs/2019.4-gcc-9.2.0-openmpi
```

## ![cpu](https://img.shields.io/badge/-cpu-blue) (CPU) GROMACS 的 Slurm 脚本
在 cpu 队列上，总共使用 80 核 (n = 80)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 2 个节点：
```bash
#!/bin/bash

#SBATCH -J gromacs_test
#SBATCH -p cpu
#SBATCH -n 80
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load gromacs/2019.4-gcc-9.2.0-openmpi
module load gcc openmpi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 gmx_mpi mdrun -s ./ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 10000
```

## ![cpu](https://img.shields.io/badge/-cpu-blue) (CPU) GROMACS 提交作业
```bash
$ sbatch slurm.test
```


## ![gpu](https://img.shields.io/badge/-gpu-green) (GPU) GROMACS 使用

Pi 集群已预置 NVIDIA GPU CLOUD 提供的优化镜像，调用该镜像即可运行 GROMACS，无需单独安装，目前版本为 2018.2。该容器文件位于 /lustre/share/img/gromacs-2018.2.simg

以下 slurm 脚本，在 dgx2 队列上使用 1 块 gpu，并配比 6 cpu 核心，调用 singularity 容器中的 GROMACS：

```bash
#!/bin/bash
#SBATCH -J gromacs_gpu_test
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 6
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH -N 1

IMAGE_PATH=/lustre/share/img/gromacs-2018.2.simg

ulimit -s unlimited
ulimit -l unlimited

singularity run --nv $IMAGE_PATH gmx mdrun -deffnm benchmark -ntmpi 6 -ntomp 1
```

使用如下指令提交：

```bash
$ sbatch gromacs_gpu.slurm
```

## ![cpu](https://img.shields.io/badge/-cpu-blue) ![gpu](https://img.shields.io/badge/-gpu-green) 性能评测

测试使用了 GROMACS 提供的 Benchmark 算例进行了 CPU 和 GPU 的性能进行对比。其中 cpu 测试使用单节点40核心，dgx2 测试分配 1 块 gpu 并配比 6 核心。

| Settings | Performance(ns/day) |
| --- | --- |
| CPU (2019.2-gcc/8.3) | 43.718 |
| CPU (2019.2-gcc/9.2) | 43.362 |
| CPU (2019.4-gcc/8.3) | 43.783 |
| CPU (2019.4-gcc/9.2) | 43.057 |
| CPU (2019.4-intel/19.0.4) | 43.296 |
| DGX2 (Singularity) | 19.425 |

本测试中使用到的测试算例均可在 `/lustre/share/benchmarks/gromacs`找到，用户可自行取用测试。测试时，需将上述目录复制到家目录下。

## 参考文献

- [gromacs官方网站](http://www.gromacs.org/)
- [NVIDIA GPU CLOUD](ngc.nvidia.com)
- [Singularity文档](https://sylabs.io/guides/3.5/user-guide/)
