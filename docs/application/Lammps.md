# <center>Lammps</center> 

-----

## 加载预安装的 Lammps

Pi 2.0 系统中已经预装 lammps/20190807 (Intel+cpu 版本)，可以用以下命令加载: 

```
$ module load lammps/20190807-intel-19.0.5-impi
```

## 提交 Intel+CPU 版本 Lammps 任务

使用 Intel 2019 编译的 CPU 版本 Lammps 运行单节点作业脚本示例 lammps_cpu_intel.slurm 如下：


```bash
#!/bin/bash

#SBATCH -J lammps_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
module load lammps/20190807-intel-19.0.5-impi

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 lmp -i YOUR_INPUT_FILE
```

并使用如下指令提交：

```bash
$ sbatch lammps_cpu_intel.slurm
```
