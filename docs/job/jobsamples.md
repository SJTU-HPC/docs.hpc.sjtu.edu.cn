# <center>π集群上作业示例<center/>

-------
本文档中使用的作业样本可以在/lustre/share/samples中找到。 在继续之前，请阅读有关预置软件环境的文档。

## <center>OpenMP 示例<center/>
以OpenMP为例，名为omp_hello.c代码如下：
```c
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
int nthreads, tid;

  /* Fork a team of threads giving them their own copies of variables */
  #pragma omp parallel private(nthreads, tid)
    {

     /* Obtain thread number */
     tid = omp_get_thread_num();
     printf("Hello World from thread = %d\n", tid);

     /* Only master thread does this */
     if (tid == 0)
       {
        nthreads = omp_get_num_threads();
        printf("Number of threads = %d\n", nthreads);
       }

     }  /* All threads join master thread and disband */
}
```
### 使用GCC 9.2.0编译
```bash
$ module purge; module load gcc 
$ gcc -fopenmp omp_hello.c -o omphello
```
在本地运行4线程应用程序
```bash
$ export OMP_NUM_THREADS=4 && ./omphello
```

准备一个名为ompgcc.slurm的作业脚本

```bash
#!/bin/bash

#SBATCH --job-name=Hello_OpenMP
#SBATCH --partition=small
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -n 8
#SBATCH --ntasks-per-node=8

ulimit -l unlimited
ulimit -s unlimited

source /usr/share/Modules/init/bash
module purge
module load gcc

export OMP_NUM_THREADS=8
./omphello
```

提交到SLURM
```bash
$ sbatch ompgcc.slurm
```

### 使用Intel编译器

```bash
$ module purge; module load intel 
$ icc -fopenmp omp_hello.c -o omphello
```
在本地运行4线程应用程序
```bash
$ export OMP_NUM_THREADS=4 && ./omphello
```

准备一个名为ompicc.slurm的作业脚本
```bash
#!/bin/bash

#SBATCH --job-name=Hello_OpenMP
#SBATCH --partition=small
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -n 8
#SBATCH –-ntasks-per-node=8
ulimit -l unlimited
ulimit -s unlimited

source /usr/share/Modules/init/bash
module purge
module load intel

export OMP_NUM_THREADS=8
./omphello
```

提交到SLURM
```bash
$ sbatch ompicc.slurm
```

## <center>MPI示例</center>
以mpihello.c为例，代码如下：
```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <netdb.h>

#define MAX_HOSTNAME_LENGTH 256

int main(int argc, char *argv[])
{
    int pid;
    char hostname[MAX_HOSTNAME_LENGTH];

    int numprocs;
    int rank;

    int rc;

    /* Initialize MPI. Pass reference to the command line to
     * allow MPI to take any arguments it needs
     */
    rc = MPI_Init(&argc, &argv);

    /* It's always good to check the return values on MPI calls */
    if (rc != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }

    /* Get the number of processes and the rank of this process */
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* let's see who we are to the "outside world" - what host and what PID */
    gethostname(hostname, MAX_HOSTNAME_LENGTH);
    pid = getpid();

    /* say who we are */
    printf("Rank %d of %d has pid %5d on %s\n", rank, numprocs, pid, hostname);
    fflush(stdout);

    /* allow MPI to clean up after itself */
    MPI_Finalize();
    return 0;
}
```

### 使用OpenMPI+GCC编译
```bash
$ module purge; module load gcc/8.3.0-gcc-4.8.5 openmpi/3.1.4-gcc-8.3.0
$ mpicc mpihello.c -o mpihello
```
准备一个名为job_openmpi.slurm的作业脚本
```bash
#!/bin/bash

#SBATCH --job-name=mpihello
#SBATCH --partition=cpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -n 80
#SBATCH --ntasks-per-node=40

ulimit -s unlimited
ulimit -l unlimited

source /usr/share/Modules/init/bash
module purge
module load gcc/8.3.0-gcc-4.8.5 openmpi/3.1.4-gcc-8.3.0

srun --mpi=pmi2 ./mpihello
```

最后，将作业提交到SLURM
```bash
$ sbatch job_openmpi.slurm
```

### 使用Intel编译器
```bash
$ module purge; module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
$ mpiicc mpihello.c -o mpihello
```

准备一个名为job_impi.slurm的作业脚本
```bash
#!/bin/bash

#SBATCH --job-name=mpihello
#SBATCH --partition=cpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -n 80
#SBATCH --ntasks-per-node=40

ulimit -s unlimited
ulimit -l unlimited

source /usr/share/Modules/init/bash
module purge
module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

srun ./mpihello
```

最后，将作业提交到SLURM
```bash
$ sbatch -p cpu job_impi.slurm
```

## <center>MPI + OpenMP混合示例</center>

以hybridmpi.c为例，代码如下：
```c
#include <stdio.h>
#include "mpi.h"
#include <omp.h>

int main(int argc, char *argv[]) {
  int numprocs, rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int iam = 0, np = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);

  #pragma omp parallel default(shared) private(iam, np)
  {
    np = omp_get_num_threads();
    iam = omp_get_thread_num();
    printf("Hello from thread %d out of %d from process %d out of %d on %s\n",
           iam, np, rank, numprocs, processor_name);
  }

  MPI_Finalize();
}
```

### 使用GCC编译如下：
```bash
$ module purge && module load gcc/8.3.0-gcc-4.8.5 openmpi/3.1.4-gcc-8.3.0
$ mpicc -O3 -fopenmp hybridmpi.c -o hybridmpi
```
准备一个名为hybridmpi.slurm的作业脚本
```bash
#!/bin/bash

#SBATCH --job-name=HybridMPI
#SBATCH --partition=cpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBAkCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:01:00 

ulimit -s unlimited
ulimit -l unlimited

source /usr/share/Modules/init/bash
module purge
module load gcc/8.3.0-gcc-4.8.5 openmpi/3.1.4-gcc-8.3.0

export OMP_NUM_THREADS=40
srun --mpi=pmi2 ./hybridmpi
```

### 使用ICC编译
```bash
$ module purge; module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
$ mpiicc -O3 -fopenmp hybridmpi.c -o hybridmpi
```
准备一个名为hybridmpi.slurm的作业脚本

```bash
#!/bin/bash

#SBATCH --job-name=HybridMPI
#SBATCH --partition=cpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:01:00 

ulimit -s unlimited
ulimit -l unlimited

source /usr/share/Modules/init/bash
module purge; module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

export I_MPI_DEBUG=5
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

export OMP_NUM_THREADS=40
srun ./hybridmpi
```
### 将作业提交到4个计算节点上
```bash
$ sbatch -N 4 hybridmpi.slurm
```

## <center>CUDA 示例</center> 
以cublashello.cu为例，代码如下：
```c
//Example 2. Application Using C and CUBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

int main (void){
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* devPtrA;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}
```

### 使用CUDA编译
```bash
$ module purge; module load gcc/8.3.0-gcc-4.8.5 cuda/10.1.243-gcc-8.3.0
$ nvcc cublashello.cu -o cublashello -lcublas
```

作业脚本cublashello.slurm如下：
```bash
#!/bin/bash

#SBATCH --job-name=cublas
#SBATCH --partition=dgx2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1

ulimit -s unlimited
ulimit -l unlimited

source /usr/share/Modules/init/bash
module purge
module load gcc/8.3.0-gcc-4.8.5 cuda/10.1.243-gcc-8.3.0

./cublashello
```

### 将作业提交到SLURM上的dgx2分区：
```bash
$ sbatch cublashello.slurm
```
## <center>通过sbatch运行Intel LINPACK</center>

假如在多节点运行MPI作业，首先准备执行文件并输入数据：

```bash
$ cd ~/tmp
$ cp /lustre/usr/samples/LINPACK/64/xhpl_intel64 .
$ cp /lustre/usr/samples/LINPACK/64/HPL.dat .
```

然后，准备一个的作业脚本linpack.sh。 在此脚本中，我们请求cpu分区上的64个内核，每个节点16个内核。 请注意，MPI作业是通过srun（不是mpirun）启动的。

```bash
#!/bin/bash

#SBATCH --job-name=Intel_MPLINPACK
#SBATCH --partition=cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@EMAIL.COM
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -n 80
#SBATCH --ntasks-per-node=40

ulimit -s unlimited
ulimit -l unlimited

source /usr/share/Modules/init/bash
module purge
module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi
export I_MPI_DEBUG=100

srun ./xhpl_intel64
```

最后，将作业提交到SLURM.
```bash
$ sbatch linpack.sh
Submitted batch job 358
```

我们可以附加到正在运行的任务，并观察其STDOUT和STDERR：

```bash
$ sattach 358.0
$ CTRL-C
```

我们可以查看作业输出文件：
```bash
$ tail -f /lustre/home/hpc-jianwen/tmp/358.out
```

停止工作：
```bash
$ scancel 358
```

<!-- ## <center>提交具有Spack依赖关系的作业</center>

以下是一个运行Space的应用程序(gromacs)的作业脚本示例。 
```bash
#!/bin/bash

#SBATCH -J your_app
#SBATCH -p cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@EMAIL.COM
#SBATCH -o %j.out
#SBATCH -e %j.err

unlimit -s unlimited
unlimit -l unlimited

source /usr/share/Modules/init/bash

 /* Spack */
if [ -d "$HOME/spack" ]; then
    export SPACK_ROOT=$HOME/spack
    source $SPACK_ROOT/share/spack/setup-env.sh
fi

/* Add your Spack apps here */
source <(spack module tcl loads gromacs+cuda %gcc@5.4.0 )

LAUNCH_YOUR_APP
``` -->
## <center>提交作业到small分区</center>
串行作业需要提交到名为串行的队列中，作业脚本和qos中的队列名称需要相应调整。
​```bash
#!/bin/bash

#SBATCH -J Hello
#SBATCH -p small
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1

hostname
```
<!-- ## <center>提交作业到调试分区</center>
应用程序可免费调试，提交给分区的作业最多可以运行6个小时。
```bash
#!/bin/bash

#SBATCH -J Hello
#SBATCH -p debug
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@EMAIL.COM
#SBATCH --time=01:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 16
#SBATCH --ntasks-per-node=16

hostname
``` -->

