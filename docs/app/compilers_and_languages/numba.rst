.. _numba:

Numba
==========

简介
----

Numba是一款可以将python函数编译为机器代码的JIT编译器，经过Numba编译的python代码(仅限数组运算)，其运行速度可以接近C或FORTRAN语言。





Numba安装以及使用说明
-----------------------------

思源一号上的Numba
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录Numbatest并进入该目录：

.. code::

    mkdir Numbatest
    cd Numbatest

2. 申请计算资源并通过conda安装Numba

.. code::

    srun -p 64c512g -n 10 --pty /bin/bash
    module load miniconda3
    conda create -n numbatest
    source activate numbatest
    conda install -c conda-forge numba

3. 在该目录下创建如下测试文件test.py：

.. code::

    import numba as nb
    import numpy as np
    from numba.typed import List
    import time


    @nb.jit('List(f4)(f4[:], f4[:], i4)', nopython=True, cache=True, parallel=False)
    def fun1(a, b, len):
        res = []
        for i in range(len):
            res.append(a[i]+b[i])
        return res
    @nb.jit('ListType(f4)(f4[:], f4[:], i4)', nopython=True, cache=True, parallel=False)
    def fun2(a, b, len):
        res = List()
        for i in range(len):
            res.append(a[i]+b[i])
        return res

    def fun3(a, b, len):
        res = []
        for i in range(len):
            res.append(a[i]+b[i])
        return res

    if __name__ == '__main__':
        len = 100000000
        a = np.random.randn(len).astype(np.float32)
        b = np.random.randn(len).astype(np.float32)
        t1 = time.time()
        c1 = fun1(a, b, len)
        t2 = time.time()
        c2 = fun2(a, b, len)
        t3 = time.time()
        c3 = fun3(a, b, len)
        t4 = time.time()

        print(f'fun1 cost: {t2-t1}s, \nfun2 cost: {t3-t2}s, \nfun3 cost: {t4-t3}s.')



4. 在该目录下创建如下作业提交脚本numbatest.slurm:

.. code::

    #!/bin/bash

    #BATCH --job-name=numbatest      # 作业名
    #SBATCH --partition=64c512g      # 64c512g队列
    #SBATCH --ntasks-per-node=10     # 每节点核数
    #SBATCH -n 10                     # 作业核心数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    ulimit -s unlimited
    ulimit -l unlimited

    module load miniconda3
    source activate numbatest

    python3 test.py

5. 使用如下命令提交作业：

.. code::

  sbatch numbatest.slurm

6. 作业完成后在.out文件中可看到如下结果：

.. code::

    fun1 cost: 2.0397536754608154s,
    fun2 cost: 1.9905965328216553s,
    fun3 cost: 17.56288480758667s.




pi2.0上的Numba
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 申请计算资源并通过conda安装numba

.. code::

    srun -p cpu -N 1 --ntasks-per-node 40    --pty /bin/bash
    module load miniconda3
    conda create -n numbatest
    source activate numbatest
    conda install -c conda-forge numba



3. 此步骤和上文完全相同；

4. 在该目录下创建如下作业提交脚本mpi4pytest.slurm:

.. code::

    #!/bin/bash

    #BATCH --job-name=numbatest      # 作业名
    #SBATCH --partition=small        # small队列
    #SBATCH --ntasks-per-node=10     # 每节点核数
    #SBATCH -n 10                     # 作业核心数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    ulimit -s unlimited
    ulimit -l unlimited

    module load miniconda3
    source activate numbatest

    python3 test.py


5. 此步骤和上文完全相同；
6. 此步骤和上文完全相同；









参考资料
-----------

-  `Numba 官网 <http://numba.pydata.org/>`__
-  `Numba github <https://github.com/numba/numba>`__
-  `Numba 知乎 <https://www.zhihu.com/tardis/bd/art/78882641?source_id=1001>`__

