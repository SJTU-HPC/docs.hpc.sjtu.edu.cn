.. _JDK:

JDK
======

简介
----
JDK是 Java 语言的软件开发工具包，主要用于移动设备、嵌入式设备上的java应用程序。
JDK是整个java开发的核心，它包含了JAVA的运行环境（JVM+Java系统类库）和JAVA工具。



JDK使用说明
-----------------------------

思源一号上的JDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录jdktest并进入该目录：

.. code::
        
    mkdir jdktest
    cd jdktest

2. 在该目录下创建如下测试文件jdktest.java：

.. code::
        
 public class jdktest{
   public static void main(String[] args) {
      double[] myList = {1.9, 2.9, 3.4, 3.5};
      // 打印所有数组元素
      for (int i = 0; i < myList.length; i++) {
         System.out.println(myList[i] + " ");
      }
      // 计算所有元素的总和
      double total = 0;
      for (int i = 0; i < myList.length; i++) {
         total += myList[i];
      }
      System.out.println("Total is " + total);
      // 查找最大元素
      double max = myList[0];
      for (int i = 1; i < myList.length; i++) {
         if (myList[i] > max) max = myList[i];
      }
      System.out.println("Max is " + max);
   }
 }

3. 在该目录下创建如下作业提交脚本jdktest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=jdktest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load jdk/12.0.2_10-gcc-11.2.0
  javac -cp . -d . jdktest.java 
  java jdktest

4. 使用如下命令提交作业：

.. code::

  sbatch jdktest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

  1.9 
  2.9 
  3.4 
  3.5 
  Total is 11.7
  Max is 3.5

pi2.0上的JDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 此步骤和上文完全相同；



3. 在该目录下创建如下作业提交脚本jdktest.slurm:

.. code::

  #!/bin/bash

  #SBATCH --job-name=jdktest    
  #SBATCH --partition=small     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load jdk/12.0.2_10-gcc-9.2.0
  javac -cp . -d . jdktest.java 
  java jdktest

4. 使用如下命令提交作业：

.. code::

  sbatch jdktest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

  1.9 
  2.9 
  3.4 
  3.5 
  Total is 11.7
  Max is 3.5


  



参考资料
---------

-  `Linux下编译运行java文件 <https://www.jianshu.com/p/033dcc32e8cd>`__
-  `菜鸟教程 java <https://www.runoob.com/java/java-array.html>`__

