.. _openseespy:

Openseespy  
============

简介
-------
OpenSeesPy is free for research, education, and internal use. Commercial redistribution of OpenSeesPy, such as, but not limited to, an application or cloud-based service that uses import openseespy, requires a license similar to that required for commercial redistribution of OpenSees.exe. 

使用conda在集群上安装openseespy
--------------------------------------

以在Pi2.0集群安装为例

.. code:: console
    
    $ srun -p cpu -n 8 --pty /bin/bash
    $ module load miniconda3
    $ conda create -n openseespy python=3.6.8
    $ source activate openseespy
    $ pip install openseespy==3.2.2.6


提交openseespy多核并行计算的脚本：
------------------------------------

.. code:: console

    #!/bin/bash
    #SBATCH -J hello-python
    #SBATCH -p cpu
    #SBATCH -o %j.out
    #SBATCH -e %j.err
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=40

    module load miniconda3
    source activate openseespy
    module load mpich

mpiexec -np 40 python trial.py

参考资料
--------

-  `Openseespy docs <https://openseespydoc.readthedocs.io/en/latest/index.html>`__
-  `Openseespy github <https://github.com/zhuminjie/OpenSeesPy>`__