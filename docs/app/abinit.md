# <center>ABINIT</center> 

-----

## 简介

ABINIT is a DFT code based on pseudopotentials and a planewave basis, which calculates the total energy, charge density and electronic structure for molecules and periodic solids. In addition to many other features, it provides the time dependent DFT, or many-body perturbation theory (GW approximation) to compute the excited states.

## Pi 上的 ABINIT
查看 Pi 上已编译的软件模块:
```bash
$ module avail abinit
```

调用该模块:
```bash
$ module load abinit/8.10.3-gcc-9.2.0-openblas-openmpi
```

## Pi 上的 Slurm 脚本 slurm.test

在 cpu 队列上，总共使用 80 核 (n = 80)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 2 个节点：
```bash
#!/bin/bash

#SBATCH -J abinit_test
#SBATCH -p cpu
#SBATCH -n 80
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load abinit

srun --mpi=pmi2 < example.in
```

## Pi 上提交作业
```bash
$ sbatch slurm.test
```

## 参考资料
- [ABINIT 官网](http://www.abinit.org)

