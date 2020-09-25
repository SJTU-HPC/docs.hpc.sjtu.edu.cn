# <center>Keras</center> 

-----

## 简介

Keras is a minimalist, highly modular neural networks library written in Python and capable on running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

## Pi 上的 Keras 安装方法

```bash
module purge
module load miniconda3
conda create -n mypy
source activate mypy
conda install -c anaconda keras tensorflow-gpu
```

## Pi 上的 Slurm 脚本 slurm.test
在 dgx2 队列上，使用 1 张卡（gres=gpu:1），配合 6 核芯 (n = 6)<br>
```bash
#!/bin/bash

#SBATCH -J keras
#SBATCH --partition=dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem=MaxMemPerNode

ulimit -l unlimited
ulimit -s unlimited

module purge
source activate mypy

python ...
```
  
## Pi 上提交作业
```bash
$ sbatch slurm.test
```

## 参考链接
- [Keras 官网](https://keras.io/)
