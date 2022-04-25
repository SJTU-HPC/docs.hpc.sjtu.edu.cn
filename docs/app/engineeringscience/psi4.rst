.. _psi4:

PSI4  
=========

简介
-------
PSI4 provides a wide variety of quantum chemical methods using state-of-the-art numerical methods and algorithms. 
Several parts of the code feature shared-memory parallelization to run efficiently on multi-core machines (see Sec. Threading).
An advanced parser written in Python allows the user input to have a very simple style for routine computations, 
but it can also automate very complex tasks with ease.

使用conda在集群上安装PSI4
-------------------------

可以使用以下命令在思源超算和闵行超算上安装psi4。

.. code:: console
    
    $ module load miniconda3
    $ conda create -n p4env psi4 -c psi4
    $ source activate p4env
    $ pip install pytest==7.0.1


测试conda安装的PSI4
--------------------

.. code:: console

    $ module load miniconda3
    $ source activate p4env
    $ psi4 --test


结果如下：

.. code:: console

    = 30 passed, 24 skipped, 3347 deselected, 2 xfailed, 31 warnings in 152.14s (0:02:32) =
