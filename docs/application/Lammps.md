# <center>Lammps</center> 

-----

Pi2.0 系统中已经预装 lammps/20190807，可以用以下命令加载: 

```
$ module load lammps/20190807-gcc-8.3.0-openblas-openmpi
```

加载后即可使用 `lmp` 命令。计算时可用以下示例脚本提交任务，

```
#SBATCH -J lammps_test
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module load lammps/20190807-gcc-8.3.0-openblas-openmpi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 您的LAMMPS命令
```
