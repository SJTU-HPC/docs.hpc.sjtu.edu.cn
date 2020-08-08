# <center>SIESTA</center> 

-----

## 简介

SIESTA is both a method and its computer program implementation, to perform efficient electronic structure calculations and ab initio molecular dynamics simulations of molecules and solids. SIESTA's efficiency stems from the use of a basis set of strictly-localized atomic orbitals. A very important feature of the code is that its accuracy and cost can be tuned in a wide range, from quick exploratory calculations to highly accurate simulations matching the quality of other approaches, such as plane-wave methods.

## Pi 上的 SIESTA

Pi2.0 系统中已经预装 SIESTA (GNU+cpu 版本)，可用以下命令加载: 

```bash
$ module load siesta
```

## Pi 上的 Slurm 脚本 slurm.test

在 cpu 队列上，总共使用 40 核 (n = 40)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 1 个节点：

```bash
#!/bin/bash

#SBATCH -J nechem_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load siesta

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 siesta < input.in
```

并使用如下指令提交：

```bash
$ sbatch slurm.test
```

## 参考链接
- [SIESTA 官网](http://departments.icmab.es/leem/siesta/)