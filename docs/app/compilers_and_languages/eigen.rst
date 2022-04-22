.. _eigen:

Eigen
==========

简介
----

Eigen库是一个开源的矩阵运算库，其利用C++模板编程的思想，构造所有矩阵通过传递模板参数形式完成。由于模板类不支持库链接方式编译，而且模板类要求全部写在头文件中，从而导致导致Eigen库只能通过开源的方式供大家使用，并且只需要包含Eigen头文件就能直接使用。



eigen使用说明
-----------------------------

思源一号上的Eigen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录eigentest并进入该目录：

.. code::
        
    mkdir eigentest
    cd eigentest

2. 在该目录下创建如下测试文件myeigen.cpp：

.. code::
        
  #include <iostream>
  #include <Eigen/Dense>
  using Eigen::MatrixXd;
  int main()
  {
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
  return 0;
  }

3. 在该目录下创建如下作业提交脚本eigentest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=eigentest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load eigen/3.4.0-gcc-8.3.1

  g++  myeigen.cpp -o myeigen

  ./myeigen

4. 使用如下命令提交作业：

.. code::

  sbatch eigentest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

   3  -1
   2.5 1.5

pi2.0上的Eigen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录eigentest并进入该目录：

.. code::
        
    mkdir eigentest
    cd eigentest

2. 在该目录下创建如下测试文件myeigen.cpp：

.. code::
        
  #include <iostream>
  #include <Eigen/Dense>
  using Eigen::MatrixXd;
  int main()
  {
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
  return 0;
  }

3. 在该目录下创建如下作业提交脚本eigentest.slurm:

.. code::

  #!/bin/bash

  #SBATCH --job-name=eigentest    
  #SBATCH --partition=small     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load eigen/3.3.7-gcc-8.3.0

  g++  myeigen.cpp -o myeigen

  ./myeigen

4. 使用如下命令提交作业：

.. code::

  sbatch eigentest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

   3  -1
   2.5 1.5


  



参考资料
========

-  `Eigen 官网 <https://eigen.tuxfamily.org/index.php?title=Main_Page>`__
