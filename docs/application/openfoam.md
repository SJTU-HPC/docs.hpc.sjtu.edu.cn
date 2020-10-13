# <center>OpenFOAM<center>

---------

## Pi 上的 LAMMPS

查看 Pi 上已编译的软件模块:
```bash
module av openfoam
```

调用该模块:
```bash
module load openfoam/8
```

在 cpu 队列上，总共使用 80 核 (n = 80)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 2 个节点。脚本名称可设为 slurm.test

!!! example "cpu 队列 slurm 脚本示例 OpenFoam"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 80                     # 总核数 80
    #SBATCH --ntasks-per-node=40      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load openfoam/8

    ulimit -s unlimited
    ulimit -l unlimited

    srun --mpi=pmi2 icoFoam -parallel
    ```

用下方语句提交作业
```bash
sbatch slurm.test
```


## 参考文献

- [openfoam官方网站](https://openfoam.org/)
- [Singularity文档](https://sylabs.io/guides/3.5/user-guide/)
