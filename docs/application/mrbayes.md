# <center>MrBayes</center> 

-----

## 简介

MrBayes is a program for Bayesian inference and model choice across a wide range of phylogenetic and evolutionary models. MrBayes uses Markov chain Monte Carlo (MCMC) methods to estimate the posterior distribution of model parameters.

## Pi 上的 MrBayes
查看 Pi 上已编译的软件模块:
```bash
$ module spider mrbayes
```

调用该模块:
```bash
$ module load mrbayes/3.2.7a-gcc-8.3.0-openmpi
```

## Pi 上的 Slurm 脚本 slurm.test
在 cpu 队列上，总共使用 16 核 (n = 16)<br>
cpu 队列每个节点配有 40 核，这里使用了 1 个节点：
```bash
#!/bin/bash

#SBATCH --job-name=mrbayes
#SBATCH --partition=cpu
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --output=%j.out
#SBATCH --error=%j.err

ulimit -s unlimited
ulimit -l unlimited

module purge
module load mrbayes/3.2.7a-gcc-8.3.0-openmpi

srun --mpi=pmi2 mb your_input_file
```

!!! tips
    根据我们的测试，mrbayes最多只能使用16进程/节点的配置，请根据具体需要调整`-n`和`--ntasks-per-node`参数
    
## Pi 上提交作业
```bash
$ sbatch mrbayes_cpu_gnu.slurm
```

## 参考链接
- [MrBayes 官网](http://nbisweden.github.io/MrBayes/)
