.. _gamess:

GAMESS
======

简介
----

GAMESS（General Atomic and Molecular Electronic Structure System）是一款由Ames研究中心的Mark Gordon开发的开源量子化学软件。GAMESS的功能十分强大，能完成多种类型的计算任务。它支持使用RHF、UHF、ROHF或GVB波函数计算半经验MNDO、AM1或PM3模型；可以进行QM/MM计算，可以处理溶剂效应，很大程度上它可以成为Gaussian的替代品。
更多信息请访问https://www.msg.chem.iastate.edu/gamess

软件下载
--------

软件源代码需在Mark Gordon课题组主页上申请，同意协议、填写邮箱等信息后一两天内会收到下载链接、账户名和下载密码。（https://www.msg.chem.iastate.edu/gamess/download.html）

自行编译
---------

1. 申请计算节点并加载模块
~~~~~~~~~~~~~~~~~~~~~~~~~~


以选用gfortran为编译器，MKL为数学库，MPI并行编译为例

.. code:: bash
   
   srun -p 64c512g -n 4 --pty /bin/bash
   module purge
   module load intel-oneapi-compilers/2022.1.0 intel-oneapi-mkl/2022.1.0 intel-oneapi-mpi/2021.6.0

2. 检查.bashrc 和 .bash_profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   检查bash shell配置（参考https://docs.hpc.sjtu.edu.cn/faq/index.html#bashrc）

3. 解压并进入文件
~~~~~~~~~~~~~~~~~

.. code:: bash

   tar -xvf gamess-current.tar.gz
   cd ./gamess   

4. 生成编译文件
~~~~~~~~~~~~~~~

方式一
在当前路径运行“./config”命令，并根据屏幕提示逐条输出选项
方式二
直接生成Makefile文件和install.info文件

生成Makefile文件（其中默认当前路径为./gamess）

.. code:: bash

   touch Makefile
   cat >> Makefile <<EOF
   GMS_PATH = ${PWD}
   GMS_VERSION = 00
   include ${PWD}/Makefile.in
   EOF

生成install.info文件（其中默认当前路径为./gamess，如需更改gamess编译路径，请修改GMS_BUILD_DIR值

.. code:: bash

   touch install.info
   cat >> install.info <<EOF
   #!/bin/csh -f
   
   #                 GAMESS Paths                     #
   setenv GMS_PATH              ${PWD}
   setenv GMS_BUILD_DIR         ${PWD}
   
   #                  Machine Type                    #
   setenv GMS_TARGET            linux64
   setenv GMS_HPC_SYSTEM_TARGET generic
   
   #              FORTRAN Compiler Setup              #
   setenv GMS_FORTRAN           ifort
   setenv GMS_IFORT_VERNO       22
   
   #         Mathematical Library Setup               #
   setenv GMS_MATHLIB           mkl
   setenv GMS_MATHLIB_PATH      ${MKLROOT}/lib/intel64
   setenv GMS_MKL_VERNO         12
   setenv GMS_LAPACK_LINK_LINE  ""
   #         parallel message passing model setup
   setenv GMS_DDI_COMM          sockets
   
   #     Michigan State University Coupled Cluster    #
   setenv GMS_MSUCC             false
   
   #         LIBCCHEM CPU/GPU Code Interface          #
   setenv GMS_LIBCCHEM          false
   
   #      Intel Xeon Phi Build: none/knc/knl          #
   setenv GMS_PHI               none
   
   #         Shared Memory Type: sysv/posix           #
   setenv GMS_SHMTYPE           sysv
   
   #      GAMESS OpenMP support: true/false           #
   setenv GMS_OPENMP            false
   
   #      GAMESS LibXC library: true/false            #
   setenv GMS_LIBXC             false
   
   #      GAMESS MDI library: true/false              #
   setenv GMS_MDI               false
   
   #       VM2 library: true/false                    #
   setenv  GMS_VM2              false
   
   #       Tinker: true/false                         #
   setenv  TINKER               false
   
   #       VB2000: true/false                         #
   setenv  VB2000               false
   
   #       XMVB: true/false                           #
   setenv  XMVB                 false
   
   #       NEO: true/false                            #
   setenv  NEO                  false
   
   #       NBO: true/false                            #
   setenv  NBO                  false
   
   ####################################################
   # Added any additional environmental variables or  #
   # module loads below if needed.                    #
   ####################################################
   # Capture floating-point exceptions                #
   setenv GMS_FPE_FLAGS        ''
   EOF 

5. 编译
~~~~~~~

编译成功后将在目录中生成gamess.00.x的可执行文件

.. code:: bash
  
   make ddi
   make modules
   make -j gamess

6. 配置运行环境
~~~~~~~~~~~~~~~

Gamess程序需要依靠rungms文件来调用可执行文件gamess.00.x，在编译后需调整rungms中的参数
修改当前目录下的rungms文件，其中GMSPATH为rungms文件所在地址、SCR为临时文件所在目录

.. code:: bash

   set SCR=~app/gamess/scr
   set USERSCR=~/app/gamess/scr
   set GMSPATH=~/app/gamess

7. 验证
~~~~~~~

使用软件自带的测试示例进行测试，在当前路径下运行runall文件，并执行检查命令checktst

.. code:: bash

   ./runall 00
   ./tests/standard/checktst

得到如下内容表明验证算例全部通过

.. code:: bash

   Checking the results of your sample GAMESS calculations,
   the output files (exam??.log) will be taken from .
   Only 48 out of 49 examples terminated normally.
   Please check carefully each of the following runs:
   grep: ./exam49.log: No such file or directory
   ./exam49.log
   which did not completely finish.
   exam01: Eerr=0.0e+00 Gerr=0.0e+00.                                     Passed.
   exam02: Eerr=0.0e+00 Gerr=0.0e+00 Serr=0.0e+00 Lerr=1.8e-03+6.6e-05.   Passed.
   exam03: Eerr=0.0e+00 Gerr=0.0e+00 Derr=0.0e+00.                        Passed.
   .
   .
   .
   .
   exam48: E0err=0.0e+00 E1err=0.0e+00 Gerr=0.0e+00.                      Passed.

-  `GAMESS 官网 <https://www.msg.chem.iastate.edu/index.html>`__
