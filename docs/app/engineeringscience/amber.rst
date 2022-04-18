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

   module load miniconda3
   conda create -n amber
   source activate amber
   pip3 install numpy==1.17.2 scipy Cython
   pip3 install matplotlib
   pip3 install tk

   tar xvf amber20.tar.gz
   cd amber20
   make veryclean

将 `amber20_src/build/run_cmake` 按下方修改

.. code:: bash

   cmake $AMBER_PREFIX/amber20_src \
    -DCMAKE_INSTALL_PREFIX=$AMBER_PREFIX/amber20 \
    -DCOMPILER=GNU \
    -DMPI=FALSE -DCUDA=TRUE -DINSTALL_TESTS=TRUE \
    -DDOWNLOAD_MINICONDA=FALSE -DMINICONDA_USE_PY3=TRUE \
    2>&1 | tee cmake.log

.. code:: bash

   module load cuda
   ./run_cmake -j 8

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
