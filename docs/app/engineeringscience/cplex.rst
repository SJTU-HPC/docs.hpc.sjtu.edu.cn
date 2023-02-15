.. _cplex:

cplex
======

cplex是IBM公司开发的一款商业版的优化引擎,该引擎专门用于求解大规模的线性规划（LP）、二次规划（QP）、带约束的二次规划（QCQP）、二阶锥规划（SOCP）等四类基本问题，以及相应的混合整数规划（MIP）问题。

cplex安装说明
-----------------------------

1. 如下图所示，登录 `IBM cplex 官网 <https://www.ibm.com/cn-zh/products/ilog-cplex-optimization-studio?utm_content=SRCWW&p1=Search&p4=43700074800244544&p5=2&gclid=CPb79Jzplv0CFb1DwgUdCBAAEw&gclsrc=ds>`__ 并注册账号，然后下载linux平台下的cplex安装包(试用免费版)：

|image1|

|image2|

2. 申请计算资源：

.. code::

    srun -p 64c512g -n 4 --pty /bin/bash (思源一号)
    或者
    srun -p cpu -N 1 --ntasks-per-node 40  --exclusive  --pty /bin/bash  (pi2.0)


3. 在自己的家目录下新建一个目录cplex作为安装目录，进入该目录，并将上一步得到的安装包上传到当前目录：

.. code::

    mkdir cplex
    cd  cplex


4. 执行以下命令使安装文件具有执行权限，然后执行该文件：

.. code::

    chmod 777 cos_installer_preview-22.1.1.0.R0-M08SWML-linux-x86-64.bin
    ./cos_installer_preview-22.1.1.0.R0-M08SWML-linux-x86-64.bin


5. 根据终端输出的安装提示一步一步往下执行即可。需要注意的是一定要将默认的安装路径 /opt/ibm/ILOG/CPLEX_Studio221 改为自己家目录下的某个目录，并且使用绝对路径(比如说/dssg/home/acct-hpc/hpcpzz/cplex)；








参考资料
--------

-  `IBM cplex 官网 <https://www.ibm.com/cn-zh/products/ilog-cplex-optimization-studio?utm_content=SRCWW&p1=Search&p4=43700074800244544&p5=2&gclid=CPb79Jzplv0CFb1DwgUdCBAAEw&gclsrc=ds>`__


.. |image1| image:: ../../img/cplex1.png
.. |image2| image:: ../../img/cplex2.png

