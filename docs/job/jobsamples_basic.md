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

说明：

1. cpu 队列为单个用户独占，无法共享。无论是否添加 --exclusive

2. cpu 队列使用核数小于 40 时，需要添加 --exclusive，否则报错

3. 独占节点时，--ntask-per-node 无论给多少，都分配 40 核。若是 mpi 程序，这个值就是指定的进程数

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
    module load lammps/2020-cpu

    ulimit -s unlimited
    ulimit -l unlimited

    srun --mpi=pmi2 lmp -i YOUR_INPUT_FILE
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
    module load gromacs/2020-cpu

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
    module load quantum-espresso/6.6

    ulimit -s unlimited
    ulimit -l unlimited

    srun --mpi=pmi2 pw.x -i test.in
    ```

### OpenFOAM

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
    #SBATCH --gres=gpu:2

    module purge
    module load tensorflow/2.2.0

    python -c 'import tensorflow as tf; \
           print(tf.__version__);   \
           print(tf.test.is_gpu_available());'
    ```

## <center>其它示例<center/>

    
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

!!! example "small 队列 slurm 脚本示例：作业结束时提醒"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           
    #SBATCH --partition=small         
    #SBATCH -n 20                     
    #SBATCH --ntasks-per-node=20
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH --mail-type=end           # 仅在作业结束时邮件提醒
    #SBATCH --mail-user=XX@sjtu.edu.cn
    ```

!!! example "small 队列 slurm 脚本示例：作业开始运行、异常终止、完结时均提醒"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test           
    #SBATCH --partition=small         
    #SBATCH -n 20                     
    #SBATCH --ntasks-per-node=20
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH --mail-type=all          # 作业开始运行、异常终止、完结时均提醒
    #SBATCH --mail-user=XX@sjtu.edu.cn
    ```




