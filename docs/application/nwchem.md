# <center>nwchem</center> 

-----

## 加载预安装的nwchem

Pi2.0 系统中已经预装nwchem-6.8.1(GNU+cpu 版本)，可以用以下命令加载: 

```bash
$ module load nwchem/6.8.1-gcc-8.3.0-openblas-openmpi
```

## 提交GNU+CPU版本nwchem任务

使用GNU编译的CPU版本nwchem运行单节点作业脚本示例nwchem_cpu_gnu.slurm如下：

```bash
#!/bin/bash

#SBATCH -J nechem_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load nwchem/6.8.1-gcc-8.3.0-openblas-openmpi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 nwchem
```

并使用如下指令提交：

```bash
$ sbatch nwchem_cpu_gnu.slurm
```