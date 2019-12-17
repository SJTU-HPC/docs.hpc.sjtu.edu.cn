# <center>Nektar++</center> 

-----

## 加载预安装的Nektar++

Pi2.0 系统中已经预装nektar-4.4.1(GNU+cpu 版本)，可以用以下命令加载: 

```
$ module load nektar/4.4.1-gcc-8.3.0-openblas-openmpi
```

## 提交GNU+CPU版本Nektar任务

使用GNU编译的CPU版本Nektar运行单节点作业脚本示例nektar_cpu_gnu.slurm如下：


```bash
#!/bin/bash

#SBATCH -J Nektar_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load nektar/4.4.1-gcc-8.3.0-openblas-openmpi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 IncNavierStokesSolver-rg
```

并使用如下指令提交：

```bash
$ sbatch nektar_cpu_gnu.slurm
```