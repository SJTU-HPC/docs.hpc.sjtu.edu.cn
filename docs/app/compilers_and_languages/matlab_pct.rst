.. _matlab_pct:

MATLAB Parallel Computing Toolbox
---------------------------------------

利用 Parallel Computing Toolbox™，可以使用多核处理器、GPU 和计算机集群来解决计算问题和数据密集型问题。利用并行 for 循环、特殊数组类型和并行化数值算法等高级别构造，无需进行 CUDA 或 MPI 编程即可对 MATLAB® 应用程序进行并行化。 通过该工具箱可以使用 MATLAB 和其他工具箱中支持并行的函数。你可以将该工具箱与 Simulink 配合使用，并行运行一个模型的多个仿真。程序和模型可以在交互模式和批处理模式下运行。

集群上部署的 MATLAB 镜像均已安装  Parallel  Computing Toolbox 并获取相关授权，打开 MATLAB 即可使用相应功能。

目前集群上配置的 MATLAB 镜像仅支持单节点并行。因此，在 π 超算上最多可40核并行，在思源超算上最多可64核并行.


MATLAB Parallel Computing Toolbox支持四种模式：

1. `运行后台进程`_

2. `分布式计算`_

3. `并行计算`_

4. `GPU计算`_ 

本文档将给出各种计算模式的示例。示例均以交互式命令行形式展现，需要先申请计算资源：

.. code:: console

    $ srun -p 64c512g -n 10 --pty /bin/bash

在计算节点上运行 MATLAB 镜像：

.. code:: console

    $ singularity run /dssg/share/imgs/matlab/matlab_latest.sif matlab
    MATLAB is selecting SOFTWARE OPENGL rendering.

                                        < M A T L A B (R) >
                                Copyright 1984-2022 The MathWorks, Inc.
                            R2022a Update 2 (9.12.0.1956245) 64-bit (glnxa64)
                                            May 11, 2022

    
    To get started, type doc.
    For product information, visit www.mathworks.com.
    
    >> 

更多的 MATLAB 使用方式，请参考文档 :ref:`matlab`.


.. _运行后台进程:

运行后台进程
--------------------------

.. _gabor_patch_avi: https://hpc.nih.gov/examples/gabor_patch_avi.html

在使用 MATLAB 交互式命令行的时候，可以让一个进程在后台运行，从而不影响继续使用交互式命令行。

请至 `gabor_patch_avi`_  将该段代码拷贝至本地并命名为 `gabor_patch_avi.m`。这段代码耗时10-60秒，并将在本地生成一个 `.avi` 文件。

函数 `gabor_patch_avi` 可在后台被运行：

.. code:: console

    >> jid = batch('gabor_patch_avi');
    >> ls
    gabor_patch_avi.m 
    >> a=5;b=10;
    >> a+b

    ans =

        15

    >> ls
    098824838.avi  gabor_patch_avi.m  

可以看到， `gabor_patch_avi` 函数和命令行命令并不是顺序执行的。



.. _分布式计算:

分布式计算
-------------------------

如果进程之间无需通信，那么可以使用分布式计算模式。

以下代码同时在后台运行4个 `gabor_patch_avi` 函数。

.. code:: console

    >> job_num=4;
    >> clust_obj = parcluster;
    >> clust_obj.NumWorkers = job_num;
    >> job_obj = clust_obj.createJob;
    >> for ii = 1:job_num
            job_obj.createTask(@gabor_patch_avi, 0);
       end
    >> job_obj.submit;


.. _并行计算: 

并行计算
-----------------------

MATLAB进程可以多核心并行、共享内存并且在线程间通信。


parfor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最简单的启用并行计算的方法是使用 MATLAB 的 `parfor` 关键字。

以下示例使用10个cpu核并行计算了 `y` 数列的值。

.. code:: console

    >>  pc = parcluster('local');
    >> parpool(pc, 10);
    Starting parallel pool (parpool) using the 'local' profile ...
    Connected to the parallel pool (number of workers: 10).
    >> n = 2000;
    >> y = zeros(n,1);
    >> parfor i = 1:n
          y(i) = max(svd(randn(i)));
       end
    >>



.. _GPU计算:


GPU计算
-------------------------

GPU计算需要先申请 GPU 计算资源：

.. code:: console

    $ srun -p a100 -N 1 --gres gpu:1 -n 12 --pty /bin/bash

使用如下命令启动GPU版本MATLAB:

.. code:: console

    $ singularity run --nv /dssg/share/imgs/matlab/matlab_latest.sif matlab

使用 `gpuArray` 将数据存入GPU，即可在GPU上进行运算。使用 `gather` 可将数据从 GPU 重新传回 CPU:

.. code:: console

    >> X = [1,2,3];
    >> G = gpuArray(X);
    >> isgpuarray(G) 

    ans =

    logical

    1

    >> GSq = G.^2;
    >> XSq = gather(GSq)

    XSq =

        1     4     9

    >> isgpuarray(XSq)

    ans =

    logical

    0

    >> 












