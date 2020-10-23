# <center>GAUSSIAN</center> 

-----

## 简介

Gaussian is a computer program used by chemists, chemical engineers, biochemists, physicists and other scientists. It utilizes fundamental laws of quantum mechanics to predict energies, molecular structures, spectroscopic data (NMR, IR, UV, etc) and much more advanced calculations.

## Gaussian 需用户自行购买核安装

Pi 上无预装的 Gaussian 软件。用户需自行购买获取版权并安装。

## Pi 上的 Slurm 脚本 slurm.test

Gaussian 有不同版本，有的适合跨节点，有的仅支持单节点。请根据使用版本，选择是否多节点并行。

示例：单节点运行 Gaussian

在 cpu 队列上，总共使用 40 核 (1 个 cpu 节点)：
```bash
#!/bin/bash

#SBATCH -J test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load abinit

g09 < example.in
```

## Pi 上提交作业
```bash
$ sbatch slurm.test
```

## 参考链接
- [GAUSSIAN 官网](https://gaussian.com/)

