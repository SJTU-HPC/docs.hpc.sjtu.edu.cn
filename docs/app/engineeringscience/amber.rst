.. _amber:

amber
======

简介
----

Amber 是分子动力学软件，用于蛋白质、核酸、糖等生物大分子的计算模拟。Amber 为商业软件，需购买授权使用。

如需使用集群上的 Amber，推荐自行编译，方法见下文。

π 集群和 ARM 上也有全局部署的 Amber，如需使用，请发邮件至 hpc 邮箱，附上课题组购买 Amber 的证明，并抄送超算帐号负责人。

Amber 编译方法
-----------------------


.. code:: bash

   srun -p 64c512g -N 1 -n 8 --pty /bin/bash     # 申请计算节点编译

   tar xvfj AmberTools20.tar.bz2                 # 解压缩
   tar xvfj Amber20.tar.bz2

   module load miniconda3
   conda create -n amber
   source activate amber
   pip3 install numpy==1.17.2 scipy Cython
   pip3 install matplotlib
   pip3 install tk

   cd amber20_src/build/

将 `amber20_src/build/run_cmake` 第 42 行  `-DCUDA=FALSE` 改为 `-DCUDA=TRUE`

然后编译：

.. code:: bash

   module load cuda
   ./run_cmake -j 8
   make install
   
编译完成后，激活即可运行

.. code:: bash

   source ../../amber20/amber.sh


作业脚本示例：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=amber
   #SBATCH --partition=a100
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=16
   #SBATCH --gres=gpu:1          # use 1 GPU
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module purge
   module load miniconda3
   module load cuda
   source activate amber
   source $YOUR_AMBER_PATH/amber20/amber.sh

   pmemd.cuda...


ARM 版 AMBER
-------------

ARM平台上运行脚本如下(amber.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 2          
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   source /lustre/share/singularity/commercial-app/amber/activate arm

   mpirun -n $SLURM_NTASKS pmemd.MPI ...

使用如下指令提交：

.. code:: bash

   $ sbatch amber.slurm


思源平台Amber
---------------

思源平台上运行脚本如下(amber.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=64c512g    
   #SBATCH -N 2          
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH --exclusive

   source /dssg/share/imgs/commercial-app/amber/activate 18cpu

   mpirun -n $SLURM_NTASKS pmemd.MPI ...

使用如下指令提交：

.. code:: bash

   $ sbatch amber.slurm
