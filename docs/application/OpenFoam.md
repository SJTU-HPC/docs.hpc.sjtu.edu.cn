# <center>OPENFOAM<center>

---------

## 加载预编译的Openfoam

Pi2.0 系统中已经预装 openfoam/1712&1906 (GNU+cpu版本)，可以用以下方式加载: 

版本|加载指令
---|:--:
1712| module load openfoam/1712-gcc-8.3.0-openmpi
1906| module load openfoam/1906-gcc-8.3.0-openmpi


## 作业脚本示例

使用GNU编译的CPU版本openfoam/1906运行单节点作业脚本test_openfoam.slurm示例如下：

```
#!/bin/bash
#SBATCH -J gromacs_cpu_test
#SBATCH -p cpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive

ulimit -s unlimited
ulimit -l unlimited

module purge
module load openfoam/1906-gcc-8.3.0-openmpi

srun --mpi=pmi2 icoFoam -parallel
```
 
并使用如下指令提交：

```
$ sbatch lammps_cpu_gnu.slurm
```

## 参考文献

- [openfoam官方网站](https://openfoam.org/)
