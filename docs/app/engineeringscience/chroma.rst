.. _chroma:

CHROMA
=======

简介
----

Chroma是一款格点量子色动力学（LQCD）数值模拟软件包，它是一个为解决夸克和胶子理论而设计的物理应用程序，其属于美国USQCD合作组在美国SciDac经费的支持下开发的USQCD软件集中的子模块。Chroma整合了QCD基本线性代数操作、多线程/进程QCD信息传递、QCD文件IO、稀疏矩阵求解等众多模块，并通过XML交互协议提供人机接口。

可用的版本
----------

+--------+-------+----------+----------------------------------------------------------+
| 版本   | 平台  | 构建方式 | 镜像路径                                                 |
+========+=======+==========+==========================================================+
| 2021.4 | |cpu| | 容器     | /dssg/share/imgs/chroma/chroma2021.04.sif 思源一号       |
+--------+-------+----------+----------------------------------------------------------+
| 2021.4 | |cpu| | 容器     | /lustre/share/img/chroma/chroma2021.04.sif pi2.0         |
+--------+-------+----------+----------------------------------------------------------+

算例路径
---------

.. code:: bash

   思源一号
   /dssg/share/sample/chroma/szscl_bench.zip

   pi2.0
   /lustre/share/samples/chroma/szscl_bench.zip

软件下载
---------

本文档使用的是在NVIDIA GPU云（NGC）上预好的Chroma-2021.04镜像。更多信息请访问

.. code:: bash

   https://catalog.ngc.nvidia.com/orgs/hpc/containers/chroma

使用方法
----------------

- `一. 思源一号 Chroma`_

- `二. π2.0 Chroma`_

.. _一. 思源一号 Chroma:

一. 思源一号 Chroma
--------------------

运行脚本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=chroma
   #SBATCH --partition=a100
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=24
   #SBATCH --gres=gpu:2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   export QUDA_RESOURCE_PATH=$PWD
   export GPU_COUNT=2

   singularity run --nv /dssg/share/imgs/chroma/chroma2021.04.sif mpirun --allow-run-as-root -x ${QUDA_RESOURCE_PATH} -n ${GPU_COUNT} chroma -i ./test.ini.xml -geom 1 1 1 ${GPU_COUNT} -ptxdb ./qdpdb -gpudirect

.. _π2.0 Chroma:

二. π2.0 Chroma
------------------

运行脚本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=chroma
   #SBATCH --partition=dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --gres=gpu:2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   export QUDA_RESOURCE_PATH=$PWD
   export GPU_COUNT=2

   singularity run --nv /lustre/share/img/chroma/chroma2021.04.sif mpirun --allow-run-as-root -x ${QUDA_RESOURCE_PATH} -n ${GPU_COUNT} chroma -i ./test.ini.xml -geom 1 1 1 ${GPU_COUNT} -ptxdb ./qdpdb -gpudirect


容器编译Chroma
--------------------

1.申请计算节点

.. code:: bash

   srun -p 64c512g -n 4 --pty /bin/bash

2.拉取远端镜像

参考文档：
``https://docs.hpc.sjtu.edu.cn/container/index.html``

.. code:: bash

   singularity pull chroma2021.04.sif docker://nvcr.io/hpc/chroma:2021.04


自行编译Chroma
--------------------

本文档编译的Chroma是基于QUDA和QDPJIT，以在思源1号上编译为例

1.申请计算节点

.. code:: bash

   srun -p 64c512g -n 4 --pty /bin/bash

2.加载模块。尝试使用mpich进行编译时会报错，建议使用openmpi

.. code:: bash

   module load gcc/11.2.0 openmpi/4.1.1-gcc-11.2.0-cuda cuda/11.5.0

3.设置环境变量

本文档使用cmake安装所有chroma的依赖项，并新建目录src以容纳所有依赖项的源程序，新建目录build用于存放编译文件，新建目录install用于存放库文件等

.. code:: bash

   cd ~/(path_to_your_installation) #自定义安装路径
   export CMAKE_MAKE_OPTS="-- -j$(nproc)"
   export SM=sm_80 # 如果在π2.0平台上编译，注意修改架构号
   export QUDA_NVSHMEM=OFF
   export QDPJIT_HOST_ARCH="X86;NVPTX"
   export ARCHFLAGS="-march=native"
   export DEBUGFLAGS=" "
   export BASEDIR=$(pwd)
   export SRCDIR=${BASEDIR}/src
   export BUILDDIR=${BASEDIR}/build
   export INSTALLDIR=${BASEDIR}/install
   mkdir -p ${SRCDIR}
   mkdir -p ${BUILDDIR}

4.下载依赖项。下载过程中容易因为网络连接问题导致拉取空项目，此处建议打包上传拉取的文件

.. code:: bash

   cd ${SRCDIR}
   git clone --depth=1 --branch  llvmorg-14.0.6 https://github.com/llvm/llvm-project.git
   git clone --branch qmp2-5-4 https://github.com/usqcd-software/qmp.git
   git clone --recursive --branch devel https://github.com/JeffersonLab/qdp-jit.git
   git clone --branch develop https://github.com/lattice/quda.git # c04150e
   git clone --branch devel --recursive https://github.com/JeffersonLab/chroma.git
   cd ${BASEDIR}

5.编译llvm-project

.. code:: bash

   cmake -S ${SRCDIR}/llvm-project/llvm -B ${BUILDDIR}/build_llvm \
    -DLLVM_ENABLE_TERMINFO="OFF" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} \
    -DLLVM_TARGETS_TO_BUILD="${QDPJIT_HOST_ARCH}" \
    -DLLVM_ENABLE_ZLIB="OFF" \
    -DBUILD_SHARED_LIBS="OFF" \
    -DLLVM_ENABLE_RTTI="ON"

   cmake --build ${BUILDDIR}/build_llvm ${CMAKE_MAKE_OPTS}
   cmake --install ${BUILDDIR}/build_llvm

6.编译QMP

.. code:: bash

   cmake -S ${SRCDIR}/qmp -B ${BUILDDIR}/build_qmp \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} \
    -DQMP_MPI=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DQMP_TESTING=OFF

   cmake --build ${BUILDDIR}/build_qmp ${CMAKE_MAKE_OPTS}
   cmake --install ${BUILDDIR}/build_qmp

7. 编译QDP-JIT

.. code:: bash

   cmake -S ${SRCDIR}/qdp-jit -B ${BUILDDIR}/build_qdp-jit \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} \
    -DCMAKE_PREFIX_PATH=${INSTALLDIR} \
    -DBUILD_SHARED_LIBS=ON \
    -DQDP_ENABLE_BACKEND=CUDA \
    -DQDP_ENABLE_COMM_SPLIT_DEVICEINIT=ON \
    -DQDP_ENABLE_LLVM14=ON \
    -DQDP_PROP_OPT=OFF \
    -DQDP_BUILD_EXAMPLES=OFF \
    -DCMAKE_CXX_FLAGS=${ARCHFLAGS}

   cmake --build ${BUILDDIR}/build_qdp-jit ${CMAKE_MAKE_OPTS}
   cmake --install ${BUILDDIR}/build_qdp-jit

8. 编译QUDA

.. code:: bash

   cmake -S ${SRCDIR}/quda -B ${BUILDDIR}/build_quda \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} \
    -DCMAKE_PREFIX_PATH=${INSTALLDIR} \
    -DQUDA_GPU_ARCH=${SM} \
    -DQUDA_NVSHMEM=${QUDA_NVSHMEM} \
    -DQUDA_DIRAC_DEFAULT_OFF=ON \
    -DQUDA_DIRAC_CLOVER=ON \
    -DQUDA_DIRAC_WILSON=ON \
    -DQUDA_INTERFACE_QDPJIT=ON \
    -DQUDA_QDPJIT=ON \
    -DQUDA_INTERFACE_MILC=OFF \
    -DQUDA_INTERFACE_CPS=OFF \
    -DQUDA_INTERFACE_QDP=ON \
    -DQUDA_INTERFACE_TIFR=OFF \
    -DQUDA_QMP=ON \
    -DQUDA_QIO=OFF \
    -DQUDA_MULTIGRID=ON \
    -DQUDA_MAX_MULTI_BLAS_N=9 \
    -DQUDA_BUILD_SHAREDLIB=ON \
    -DQUDA_BUILD_ALL_TESTS=OFF \
    -DCMAKE_CXX_FLAGS=${ARCHFLAGS}

   cmake --build ${BUILDDIR}/build_quda ${CMAKE_MAKE_OPTS}
   cmake --install ${BUILDDIR}/build_quda

9.编译Chroma

.. code:: bash

   cmake -S ${SRCDIR}/chroma -B ${BUILDDIR}/build_chroma \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/ \
    -DCMAKE_PREFIX_PATH=${INSTALLDIR}/ \
    -DBUILD_SHARED_LIBS=ON \
    -DChroma_ENABLE_JIT_CLOVER=ON \
    -DChroma_ENABLE_QUDA=ON \
    -DChroma_ENABLE_OPENMP=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_FLAGS=${ARCHFLAGS}

   cmake --build ${BUILDDIR}/build_chroma ${CMAKE_MAKE_OPTS}
   cmake --install ${BUILDDIR}/build_chroma

10.编译结果。编译完成后，目录结构如下所示

.. code:: bash

   ./(path_to_your_installation)
   ├── build      # 存放编译缓存文件
   ├── install    # 存放安装的库文件和可执行文件
   └── src        # 存放git拉取的源文件

其中，运行chroma所需的文件位于install目录中，其目录结构如下所示

.. code:: bash

   ./install
   ├── bin
   ├── examples
   ├── include
   ├── lib
   └── share

11.运行脚本。在脚本中需要设置库文件路径（/(path_to_your_installation)/install/lib）的系统变量，以及chroma可执行文件的路径（/(path_to_your_installation)/install/bin/chroma）

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=chroma
   #SBATCH --partition=a100
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --cpus-per-task=8
   #SBATCH --gres=gpu:2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc/11.2.0 openmpi/4.1.1-gcc-11.2.0-cuda cuda/11.5.0
   export LD_LIBRARY_PATH=/(path_to_your_installation)/install/lib:$LD_LIBRARY_PATH
   export QUDA_RESOURCE_PATH=$PWD
   export GPU_COUNT=2

   mpirun -x ${QUDA_RESOURCE_PATH} -np ${GPU_COUNT} /(path_to_your_installation)/install/bin/chroma -i ../test.ini.xml -geom 1 1 1 ${GPU_COUNT} -ptxdb ./ptxdb


运行结果对比
------------------

1.Chroma 思源一号
------------------

对比不同配置参数下计算所需时间，其中由于NGC镜像版本的chroma已预设好参数，因此在运行脚本中无需调整ntasks-per-node参数（保持为1即可），但对于自编译的chroma则需要根据调用的卡数调整ntasks-per-node参数（调用多少个卡就有多少个任务数）。对于不同的卡数，都需要调整运行脚本中对应的--gres=gpu参数和GPU_COUNT参数。


+----------------+------------+------------+
| 卡数-core/task | NGC镜像    | 自编译     |
+================+============+============+
| 1A100-12core   | 47         | 80         |
+----------------+------------+------------+
| 2A100s-8core   | 24         | 35         |
+----------------+------------+------------+
| 4A100s-12core  | 13         | null       |
+----------------+------------+------------+

