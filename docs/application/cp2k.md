# <center>CP2k</center> 

-----

## 简介

CP2K is a quantum chemistry and solid state physics software package that can perform atomistic simulations of solid state, liquid, molecular, periodic, material, crystal, and biological systems.

## Pi 上的 CP2K

Pi2.0 系统中已经预装 CP2K (GNU+cpu 版本)，可用以下命令加载: 

```bash
$ module load cp2k/7.1-gcc-9.2.0-openblas-openmpi
```

## Pi 上的 Slurm 脚本 slurm.test

在 cpu 队列上，总共使用 40 核 (n = 40)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 1 个节点：

```bash
#!/bin/bash

#SBATCH -J cp2k_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load cp2k/7.1-gcc-9.2.0-openblas-openmpi
module load openmpi/3.1.5-gcc-9.2.0
module load gcc/9.2.0-gcc-4.8.5

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 cp2k.popt -i example.inp
```

并使用如下指令提交：

```bash
$ sbatch slurm.test
```

## 参考链接
- [CP2K 官网](https://manual.cp2k.org/#gsc.tab=0)