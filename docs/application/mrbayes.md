# <center>MrBayes</center> 

-----

## 加载预安装的MrBayes

Pi2.0 系统中已经预装 mrbayes/3.2.7a (GNU+cpu 版本)，可以用以下命令加载: 

```
$ module load mrbayes/3.2.7a-gcc-8.3.0-openmpi
```

## 提交GNU+CPU版本MrBayes任务

使用GNU编译的CPU版本MrBayes运行单节点作业脚本示例mrbayes_cpu_gnu.slurm如下：


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

并使用如下指令提交：

```bash
$ sbatch mrbayes_cpu_gnu.slurm
```