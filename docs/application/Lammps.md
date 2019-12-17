# <center>Lammps</center> 

-----

## 加载预安装的Lammps

Pi2.0 系统中已经预装 lammps/20190807 (GNU+cpu 版本)，可以用以下命令加载: 

```
$ module load lammps/20190807-gcc-8.3.0-openblas-openmpi
```

## 提交GNU+CPU版本Lammps任务

使用GNU编译的CPU版本Lammps运行单节点作业脚本示例lammps_cpu_gnu.slurm如下：


```bash
#!/bin/bash

#SBATCH -J lammps_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load lammps/20190807-gcc-8.3.0-openblas-openmpi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 您的LAMMPS命令
```

并使用如下指令提交：

```bash
$ sbatch lammps_cpu_gnu.slurm
```