# <center>Nektar++</center> 

-----

## 加载预安装的Nektar++

Pi 2.0 系统中已经预装 nektar-5.0.0 (intel 版本)，可以用以下命令加载: 

```
$ module load nektar/5.0.0-intel-19.0.4-impi
```

## 提交 GNU+Intel 版本 Nektar 任务

使用 intel 编译的 CPU 版本 Nektar 运行单节点作业脚本示例 nektar_cpu_intel.slurm 如下：


```bash
#!/bin/bash

#SBATCH -J Nektar_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load nektar/5.0.0-intel-19.0.4-impi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 IncNavierStokesSolver-rg
```

并使用如下指令提交：

```bash
$ sbatch nektar_cpu_intel.slurm
```
