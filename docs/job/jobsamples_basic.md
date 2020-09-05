# <center>作业示例<center/>

-------
根据 π 集群的不同队列、不同应用软件，示例 slurm 作业脚本。


## <center>作业提交流程<center/>


- 编写作业脚本
```
  vi test.slurm  # 根据需求，选择计算资源：CPU 或 GPU、所需核数、是否需要大内存
```

- 提交作业
```
  sbatch test.slurm
```

- 查看作业和资源
```
  squeue       # 查看正在排队或运行的作业
```
或
```
  sacct        # 查看过去 24 小时内已完成的作业
```
集群资源实时状态查询
```
  sinfo        # 若有 idle 或 mix 状态的节点，排队会比较快
```


## <center>π 集群队列介绍<center/>

π 集群上现有 small, cpu, huge, 192c6t 和 dgx2 队列。

`scontrol show partition` 查看集群队列介绍

`sinfo` 查看集群资源实时状态

| 队列名 | 说明 |
| ---- | ---- |
| small | 允许使用 CPU 核数为 1~39，每核配比 4G 内存，节点可共享使用；单节点配置为 40 核，192G 内存 |
| cpu | 允许使用 CPU 核数为 40~24000，每核配比 4G 内存，节点需独占使用；单节点配置为 40核，192G 内存。用户需独占节点，用满 40 核，或使用部分核心的时候加上 `--exclusive` 命令 |
| huge | 允许使用 CPU 核数为 1~80，每核配比 35G 内存，节点可共享使用；单节点配置为 80 核，3T 内存 |
| 192c6t | 允许使用 CPU 核数为 96~192，每核配比 31G 内存，节点可共享使用；单节点配置为 192 核，6T 内存 |
| dgx2 | 允许使用 GPU 卡数为 1~128，推荐每卡配比 CPU 为 6，每 CPU 配比 15G 内存；单节点配置为 96 核，1.45T 内存，16>块 32G 显存的 V100卡 |

small, cpu, dgx2 队列允许的作业运行最长时间为 7 天。huge 和 192c6t 为 2 天

若预计超出 7 天，需提前 2 天发邮件告知用户名和 jobID 以便延长时限

## <center>各队列作业示例<center/>
  下面根据不同队列，示例 slurm 作业脚本

### small

!!! example "small 队列 slurm 脚本示例"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=small         # small 队列
    #SBATCH -n 20                     # 总核数需 <=39
    #SBATCH --ntasks-per-node=20      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err 
    ```
    small 队列不支持独占使用，也即不可添加 --exclusive 语句

### cpu

!!! example "cpu 队列 slurm 脚本示例：多节点（160 核）"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 160                    # 使用核数 160
    #SBATCH --ntasks-per-node=40      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err 
    ```

!!! example "cpu 队列 slurm 脚本示例：单节点（40 核）"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 40                     # 使用核数 40
    #SBATCH --ntasks-per-node=40      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err 
    ```

!!! example "cpu 队列 slurm 脚本示例：单节点（20 核），比如为了独占整个节点的大内存"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 20                     # 使用核数 20
    #SBATCH --ntasks-per-node=20      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err 
    #SBATCH --exclusive               # 独占节点（核数小于 40，cpu 队列必须加上此命令）           
    ```

### huge

!!! example "huge 队列 slurm 脚本示例：单节点（20 核，最高可用 80 核）"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=huge          # huge 队列
    #SBATCH -n 20                     # 使用核数 20
    #SBATCH --ntasks-per-node=20      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err 
    ```

### 192c6t

!!! example "192c6t 队列 slurm 脚本示例：单节点（20 核，最高可用 192 核）"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=192c6         # 192c6t 队列
    #SBATCH -n 96                     # 使用半个节点的资源，对应 3T 内存
    #SBATCH --ntasks-per-node=20      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err 
    ```

### dgx2

!!! example "dgx2 队列 slurm 脚本示例：单节点，分配 2 块 GPU，GPU:CPU 配比 1:6"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=dgx2          # dgx2 队列
    #SBATCH -N 1                      # 单节点
    #SBATCH --ntasks-per-node=1       
    #SBATCH --cpus-per-task=12        # 1:6 的 GPU:CPU 配比
    #SBATCH --mem=MaxMemPerNode
    #SBATCH --gres=gpu:2              # 2 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err 
    ```

## <center>常用软件作业示例<center/>
  下面根据不同应用软件，示例 slurm 作业脚本

### LAMMPS

!!! example "cpu 队列 slurm 脚本示例 LAMMPS"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 80                     # 使用核数 80
    #SBATCH --ntasks-per-node=40      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load intel-parallel-studio/cluster.2019.4-intel-19.0.4
    module load lammps/20200505-intel-19.0.4-impi

    export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
    export I_MPI_FABRICS=shm:ofi

    ulimit -s unlimited
    ulimit -l unlimited

    srun lmp -i YOUR_INPUT_FILE
    ```

### VASP

!!! example "cpu 队列 slurm 脚本示例 VASP"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 80                     # 使用核数 80
    #SBATCH --ntasks-per-node=40      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load intel-parallel-studio/cluster.2018.4-intel-18.0.4

    export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
    export I_MPI_FABRICS=shm:ofi

    ulimit -s unlimited
    ulimit -l unlimited

    srun /path/to/your_vasp_dir/bin/vasp_std
    ```

### GROMACS

!!! example "cpu 队列 slurm 脚本示例 GROMACS"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 80                     # 使用核数 80
    #SBATCH --ntasks-per-node=40      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load gromacs/2019.4-gcc-9.2.0-openmpi

    ulimit -s unlimited
    ulimit -l unlimited

    srun --mpi=pmi2 gmx_mpi mdrun -deffnm -s test.tpr -ntomp 1
    ```

### Quantum ESPRESSO

!!! example "cpu 队列 slurm 脚本示例 Quantum ESPRESSO"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 80                     # 总核数 80
    #SBATCH --ntasks-per-node=40      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load quantum-espresso/6.5-intel-19.0.5-impi

    export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
    export I_MPI_FABRICS=shm:ofi

    ulimit -s unlimited
    ulimit -l unlimited

    srun pw.x -i test.in
    ```

### OpenFoam

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
    module load openfoam/1912-gcc-7.4.0-openmpi

    ulimit -s unlimited
    ulimit -l unlimited

    srun --mpi=pmi2 icoFoam -parallel
    ```

### TensorFlow

!!! example "cpu 队列 slurm 脚本示例 TensorFlow"
    ```
    #!/bin/bash
    #SBATCH -J test
    #SBATCH -p dgx2
    #SBATCH -o %j.out
    #SBATCH -e %j.err
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=12
    #SBATCH --mem=MaxMemPerNode
    #SBATCH --gres=gpu:2

    module load miniconda3
    source activate tf-env

    python -c 'import tensorflow as tf; \
           print(tf.__version__);   \
           print(tf.test.is_gpu_available());'
    ```

## <center>其它示例<center/>

### singularity 容器

Pi 上已部署的 singularity 容器位于 `/lustre/share/img`

其中，gromacs/lammps/relion/pytorch/tensorflow/chroma 为 GPU 版本的 singularity

!!! example "cpu 队列 slurm 脚本示例 OpenFoam singularity 版"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=cpu           # cpu 队列
    #SBATCH -n 80                     # 使用核数 80
    #SBATCH --ntasks-per-node=40      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load openmpi/2.1.1-gcc-4.8.5

    ulimit -s unlimited
    ulimit -l unlimited

    IMAGE_PATH=/lustre/share/img/openfoam-6.simg
    mpirun -n 80 singularity run $IMAGE_PATH "sprayFlameletFoamOutput -parallel"
    ```

!!! example "gpu 队列 slurm 脚本示例 GPU lammps"
    ```
    #!/bin/bash
    #SBATCH -J lmp
    #SBATCH -p dgx2
    #SBATCH -o %j.out
    #SBATCH -e %j.err
    #SBATCH -n 1
    #SBATCH --ntasks-per-node=6
    #SBATCH --gres=gpu:1
    #SBATCH -N 1

    module use /lustre/share/img/modules
    module load lammps/2020-ngc

    ulimit -s unlimited
    ulimit -l unlimited

    srun --mpi=pmi2 lmp -k on g 1 t 6  -sf kk -pk kokkos comm device \
    -i in.eam -var x 4 -var y 2 -var z 4
    ```
    
### Job Array 阵列作业

一批作业，若所需资源和内容相似，可借助 Job Array 批量提交。Job Array 中的每一个作业在调度时视为独立的作业。

!!! example "cpu 队列 slurm 脚本示例 array"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           # 作业名
    #SBATCH --partition=small         # small 队列
    #SBATCH -n 1                      # 总核数 1
    #SBATCH --ntasks-per-node=1       # 每节点核数
    #SBATCH --output=array_%A_%a.out
    #SBATCH --output=array_%A_%a.err
    #SBATCH --array=1-20%10           # 总共 20 个子任务，每次最多同时运行 10 个

    echo $SLURM_ARRAY_TASK_ID
    ```

### 作业状态邮件提醒

--mail-type=<type> 指定状态发生时，发送邮件通知: ALL, BEGIN, END, FAIL

!!! example "small 队列 slurm 脚本示例：邮件提醒"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           
    #SBATCH --partition=small         
    #SBATCH -n 20                     
    #SBATCH --ntasks-per-node=20
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH --mail-type=end           # 作业结束时，邮件提醒
    #SBATCH --mail-user=XX@sjtu.edu.cn
    ```



