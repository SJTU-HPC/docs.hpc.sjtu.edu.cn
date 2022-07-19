.. _pysph:

PySPH
==========

简介
----

PySPH是一个用于光滑粒子流体力学(SPH)的开源框架。它允许用户用纯python编写高级代码，而这些Python代码将自动转换为高性能cython或opencl编译并执行。



PySPH使用说明
-----------------------------

在思源一号上自行安装并使用PySPH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. 使用conda创建虚拟环境并安装PySPH：

.. code::
        
  srun -p 64c512g -n 4 --pty /bin/bash
  module load miniconda3/4.10.3
  conda create --name pysph_test
  conda activate pysph_test

  conda install pip cython numpy
  pip install PySPH

2. 安装成功后执行以下命令

.. code::
        
   pysph run elliptical_drop

3. 然后可在终端得到如下结果：

.. code::
        
  Running example pysph.examples.elliptical_drop.

  Information for example: pysph.examples.elliptical_drop
  Evolution of a circular patch of incompressible fluid. (60 seconds)

  See J. J. Monaghan "Simulating Free Surface Flows with SPH", JCP, 1994, 100, pp
  399 - 406

  An initially circular patch of fluid is subjected to a velocity profile that
  causes it to deform into an ellipse. Incompressibility causes the initially
  circular patch to deform into an ellipse such that the area is conserved. An
  analytical solution for the locus of the patch is available (exact_solution)

  This is a standard test for the formulations for the incompressible SPH
  equations.
  Elliptical drop :: 5025 particles
  Effective viscosity: rho*alpha*h*c/8 = 0.5687500000000001
  Generating output in /dssg/home/acct-hpc/hpcpzz/pysphtest/elliptical_drop_output
  ----------------------------------------------------------------------
  No of particles:
  fluid: 5025
  ----------------------------------------------------------------------
  Setup took: 6.09816 secs
  100%|████████████████████| 1.1kit | 7.6e-03s [00:39.3<00:0.0 | 0.035s/it]
  Run took: 39.31432 secs
  Post processing requires matplotlib.




在pi2.0上自行安装并使用PySPH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 使用conda创建虚拟环境并安装PySPH：

.. code::
        
  srun -p small -n 4 --pty /bin/bash
  module load miniconda3/4.8.2
  conda create --name pysph_test
  conda activate pysph_test

  conda install pip==21.3.1 cython numpy

  pip install PySPH



2. 此步骤和上文完全相同；



3. 此步骤和上文完全相同；





参考资料
-----------

-  `PySPH 官网 <https://pysph.readthedocs.io/en/latest/installation.html>`__
-  `PySPH github地址 <https://github.com/pypr/pysph>`__

