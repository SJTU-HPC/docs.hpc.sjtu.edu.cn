# <center>OPENFOAM<center>

---------

## 加载预编译的Openfoam

Pi2.0 系统中已经预装 openfoam/1712&1912 (GNU+cpu版本)，可以用以下方式加载: 

版本|加载指令
---|:--:
1712| module load openfoam/1712-gcc-7.4.0-openmpi 
1912| module load openfoam/1912-gcc-7.4.0-openmpi

可以用下方命令查看系统中已装的全部 module:
```bash
module av
```

## 作业脚本示例

使用GNU编译的CPU版本openfoam/1912运行单节点作业脚本test_openfoam.slurm示例如下：

```bash
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
module load openfoam/1912-gcc-7.4.0-openmpi

srun --mpi=pmi2 icoFoam -parallel
```

并使用如下指令提交：

```bash
$ sbatch lammps_cpu_gnu.slurm
```

## 使用singularity容器中的OPENFOAM

集群中已经预置了OPENFOAM的镜像，通过调用该镜像即可运行OPENFOAM作业，无需单独安装，目前版本为`OPENFOAM 6`。该容器文件位于/lustre/share/img/openfoam-6.simg和/lustre/share/img/openfoam-6-it.simg，分别用于提交脚本作业和交互式作业。

## 使用singularity容器提交OPENFOAM作业

使用singularity容器中的openfoam运行单节点作业脚本示例openfoam_singularity.slurm如下：

```bash
#!/bin/bash
#SBATCH -J openfoam_singularity_test
#SBATCH -p cpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

IMAGE_PATH=/lustre/share/img/openfoam-6.simg

ulimit -s unlimited
ulimit -l unlimited

singularity run $IMAGE_PATH "simpleFoam --help"
```

并使用如下指令提交：

```bash
$ sbatch openfoam_singularity.slurm
```

使用singularity容器中的openfoam运行多节点作业脚本示例openfoam_singularity_multi_node.slurm如下：

```bash
#!/bin/bash

#SBATCH -J openfoam_singularity_multi_node_test
#SBATCH -p cpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 128
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive

IMAGE_PATH=/lustre/share/img/openfoam-6.simg

ulimit -s unlimited
ulimit -l unlimited

module load openmpi/2.1.1-gcc-4.8.5

mpirun -n 128 singularity run $IMAGE_PATH "sprayFlameletFoamOutput -parallel"
```

并使用如下指令提交：

```bash
$ sbatch openfoam_singularity_multi_node.slurm
```

## 使用singularity容器提交交互式OPENFOAM作业

要提交交互式作业：

```bash
srun -p cpu -N 1 --exclusive --pty singularity run /lustre/share/img/openfoam-6-it.simg
```

## 参考文献

- [openfoam官方网站](https://openfoam.org/)
- [Singularity文档](https://sylabs.io/guides/3.5/user-guide/)
