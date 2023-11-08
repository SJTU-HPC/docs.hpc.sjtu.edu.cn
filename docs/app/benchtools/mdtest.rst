mdtest
========

简介
~~~~~
mdtest是一个用于测试文件系统性能的基准测试工具，特别用于测量文件系统的元数据性能，如创建、查找、读取和删除文件和目录。这个工具通常用于评估并行文件系统、分布式文件系统和对象存储系统的性能，以及它们在高性能计算（HPC）和大规模数据存储环境中的表现。

主要功能
~~~~~~~~~~
mdtest 的主要功能包括：

1. 创建和删除文件/目录：mdtest可以模拟并记录文件和目录的创建和删除操作，以便评估文件系统在处理这些操作时的性能。
2. 统计文件/目录：工具可以执行文件和目录的统计操作，以评估文件系统执行元数据查询的性能。
3. 读取文件：mdtest可以模拟从文件中读取数据的操作，以便评估文件系统的读取性能。
4. 随机操作：mdtest可以配置为执行随机操作，以评估文件系统的随机访问性能。
5. 可定制性：工具提供了多个命令行选项，可以用于配置测试的参数，如目录结构的深度、分支因子、迭代次数、并发性等。
6. 并行性：mdtest可以在多个并行任务之间执行测试，以模拟多用户或多任务环境下的文件系统性能。
7. 输出和报告：mdtest生成性能数据的输出，允许用户分析文件系统性能，并生成报告以帮助优化和调整文件系统配置。

mdtest是一个有用的工具，可帮助管理员和研究人员评估和比较不同文件系统的性能，特别是在高性能计算环境中，其中文件系统的性能至关重要。

集群上部署的mdtest
~~~~~~~~~~~~~~~~~~~~~
目前mdtest工具已经被收纳到ior测试工具中，调用ior工具即可进行mdtest测试。

======== ======== ============ =================================
**版本** **平台** **构建方式** **模块名**
======== ======== ============ =================================
3.3.0    思源一号 spack        ior/3.3.0-gcc-11.2.0-openmpi
3.3.0    Pi2.0    spack        ior/3.3.0-gcc-9.2.0-openmpi-4.0.5
======== ======== ============ =================================

mdtest 用法
~~~~~~~~~~~~~

选项含义
---------

mdtest 选项含义

::

       -a：用于S3目标的用户ID
       -A：S3目标的主机名或IP地址
       -b：分层目录结构的分支因子
       -B：在阶段之间没有障碍（创建/统计/删除）
       -c：集体创建：任务0执行所有创建和删除操作
       -C：仅创建文件/目录
       -d：测试运行的目录
       -D：仅对目录执行测试（不包括文件）
       -e：从每个文件中读取的字节数
       -E：仅读取文件
       -f：测试将在其上运行的任务的第一个编号
       -F：仅对文件执行测试（不包括目录）
       -g：添加到唯一性桶名称的整数标识符
       -h：打印帮助信息
       -i：测试运行的迭代次数
       -I：每个树节点的项目数    
       -l：测试将在其上运行的任务的最后一个编号
       -L：仅在叶子级别创建文件/目录
       -M：每个任务的唯一工作目录，并跨LUSTRE MDTS执行分布
       -n：每个任务将创建/统计/删除＃个文件/目录
       -N：文件/目录统计之间相邻任务的步幅＃（本地=0）
       -p：迭代前延迟（以秒为单位）
       -r：仅删除文件/目录
       -R：随机统计文件/目录（可提供可选种子）    
       -s：每个测试的任务数量之间的步幅（需要4个节点以避免缓存效应）
       -S：共享文件访问（仅限文件，不包括目录）
       -t：唯一工作目录开销时间
       -T：仅统计文件/目录        
       -u：每个任务的唯一工作目录
       -v：详细度（每个选项的每个实例递增1）
       -V：详细度值
       -w：要写入每个文件的字节数
       -y：在写入完成后同步文件
       -z：分层目录结构的深度

分层目录结构（树）的示例
----------------------------------

::

                   分层目录结构（树）

                                  =======
                                 |       |  （树节点）
                                  =======
                                 /   |   \
                           ------    |    ------
                          /          |          \
                      =======     =======     =======
                     |       |   |       |   |       |  （叶级）
                      =======     =======     =======

在这个示例中，树的深度为1（z=1），分支因子为3（b=3）。树的顶部节点是根节点。离根节点最远的节点级别是叶子级别。所有由mdtest创建的树都是平衡的。

简单示例
--------------
下面已思源一号集群为例，执行如下简单的运行示例，进一步了解mdtest的运行方式，

1. 这个命令将创建一个类似上述示例的树，然后每个任务将在每个树节点创建10个文件/目录。将创建三个这样的树（每个迭代一个）。

::

   #!/bin/bash
   #SBATCH --job-name="mdtest"
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH -p 64c512g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc openmpi ior
   export UCX_NET_DEVICES=mlx5_0:1
   export OMPI_MCA_btl=^openib

   mdtest -z 1 -b 3 -I 10 -C -i 3

测试结果

::

   -- started at 11/07/2023 14:02:14 --

   mdtest-3.3.0 was launched with 1 total task(s) on 1 node(s)
   Command line used: mdtest '-z' '1' '-b' '3' '-I' '10' '-C' '-i' '3'
   Path: /dssg/home/acct-hpc/hpcxdy/test/mdtest
   FS: 7487.9 TiB   Used FS: 77.4%   Inodes: 3814.7 Mi   Used Inodes: 67.0%

   Nodemap: 1
   1 tasks, 40 files/directories

   SUMMARY rate: (of 3 iterations)
      Operation                      Max            Min           Mean        Std Dev
      ---------                      ---            ---           ----        -------
      Directory creation        :       7737.830       1898.994       5539.577       2592.713
      Directory stat            :          0.000          0.000          0.000          0.000
      Directory removal         :          0.000          0.000          0.000          0.000
      File creation             :       4826.385       4314.873       4610.705        216.379
      File stat                 :          0.000          0.000          0.000          0.000
      File read                 :          0.000          0.000          0.000          0.000
      File removal              :          0.000          0.000          0.000          0.000
      Tree creation             :       7340.325        234.797       4841.319       3261.222
      Tree removal              :          0.000          0.000          0.000          0.000
   -- finished at 11/07/2023 14:02:14 --

2. 在当前工作目录中创建一个深度为5、分支因子为2的目录树。每个任务在每个树节点上操作10个文件/目录。

::

   #!/bin/bash
   #SBATCH --job-name="mdtest"
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH -p 64c512g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc openmpi ior
   export UCX_NET_DEVICES=mlx5_0:1
   export OMPI_MCA_btl=^openib

   mdtest -I 10 -z 5 -b 2

测试结果

::

   -- started at 11/07/2023 14:04:55 --

   mdtest-3.3.0 was launched with 1 total task(s) on 1 node(s)
   Command line used: mdtest '-I' '10' '-z' '5' '-b' '2'
   Path: /dssg/home/acct-hpc/hpcxdy/test/mdtest
   FS: 7487.9 TiB   Used FS: 77.4%   Inodes: 3814.7 Mi   Used Inodes: 67.0%

   Nodemap: 1
   1 tasks, 630 files/directories

   SUMMARY rate: (of 1 iterations)
      Operation                      Max            Min           Mean        Std Dev
      ---------                      ---            ---           ----        -------
      Directory creation        :       8063.150       8063.150       8063.150          0.000
      Directory stat            :     285877.774     285877.774     285877.774          0.000
      Directory removal         :      21821.806      21821.806      21821.806          0.000
      File creation             :       8053.735       8053.735       8053.735          0.000
      File stat                 :     285288.229     285288.229     285288.229          0.000
      File read                 :     132316.999     132316.999     132316.999          0.000
      File removal              :       8476.868       8476.868       8476.868          0.000
      Tree creation             :      22422.530      22422.530      22422.530          0.000
      Tree removal              :      21058.275      21058.275      21058.275          0.000
   -- finished at 11/07/2023 14:04:55 --

3. 每个任务在当前路径中的根节点上创建100个文件/目录（根节点没有分支）。这将执行三次，迭代中计算聚合值。

::

   #!/bin/bash
   #SBATCH --job-name="mdtest"
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH -p 64c512g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc openmpi ior
   export UCX_NET_DEVICES=mlx5_0:1
   export OMPI_MCA_btl=^openib

   mdtest -n 100 -i 3

测试结果

::

   -- started at 11/07/2023 14:09:22 --

   mdtest-3.3.0 was launched with 1 total task(s) on 1 node(s)
   Command line used: mdtest '-n' '100' '-i' '3'
   Path: /dssg/home/acct-hpc/hpcxdy/test/mdtest
   FS: 7487.9 TiB   Used FS: 77.4%   Inodes: 3814.7 Mi   Used Inodes: 67.0%

   Nodemap: 1
   1 tasks, 100 files/directories

   SUMMARY rate: (of 3 iterations)
      Operation                      Max            Min           Mean        Std Dev
      ---------                      ---            ---           ----        -------
      Directory creation        :      29598.602      24151.649      26791.562       2226.847
      Directory stat            :     528368.082     520410.500     524339.827       3249.423
      Directory removal         :      23142.737      22403.642      22895.651        347.904
      File creation             :      27976.199      19203.375      24963.815       4074.675
      File stat                 :     531245.186     524915.095     528648.255       2706.261
      File read                 :     178095.024     175329.708     176263.574       1295.116
      File removal              :      23606.118      15639.633      20943.016       3750.070
      Tree creation             :      27059.938       4670.584      12952.803      10025.568
      Tree removal              :      14532.983      13990.906      14203.328        236.318
   -- finished at 11/07/2023 14:09:22 --

4. 每个任务在当前目录中创建一个目录树。每棵树的深度为3，分支因子为5。在每棵树的每个节点上都操作了五个文件/目录。

::

   #!/bin/bash
   #SBATCH --job-name="mdtest"
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH -p 64c512g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc openmpi ior
   export UCX_NET_DEVICES=mlx5_0:1
   export OMPI_MCA_btl=^openib

   mdtest -I 5 -z 3 -b 5 -u

测试结果

::

   -- started at 11/07/2023 14:09:59 --

   mdtest-3.3.0 was launched with 1 total task(s) on 1 node(s)
   Command line used: mdtest '-I' '5' '-z' '3' '-b' '5' '-u'
   Path: /dssg/home/acct-hpc/hpcxdy/test/mdtest
   FS: 7487.9 TiB   Used FS: 77.4%   Inodes: 3814.7 Mi   Used Inodes: 67.0%

   Nodemap: 1
   1 tasks, 780 files/directories

   SUMMARY rate: (of 1 iterations)
      Operation                      Max            Min           Mean        Std Dev
      ---------                      ---            ---           ----        -------
      Directory creation        :      13937.588      13937.588      13937.588          0.000
      Directory stat            :     322948.871     322948.871     322948.871          0.000
      Directory removal         :      23411.636      23411.636      23411.636          0.000
      File creation             :      16614.880      16614.880      16614.880          0.000
      File stat                 :     327131.260     327131.260     327131.260          0.000
      File read                 :     141012.437     141012.437     141012.437          0.000
      File removal              :      17020.530      17020.530      17020.530          0.000
      Tree creation             :      20008.421      20008.421      20008.421          0.000
      Tree removal              :      23610.532      23610.532      23610.532          0.000
   -- finished at 11/07/2023 14:09:59 --

大规模mdtest测试
~~~~~~~~~~~~~~~~

如果要对集群进行大规模小文件读写测试，可参考下面脚本，一共使用64个节点测试，每个节点启动4个MPI进程做文件操作，总共会产生约1600万文件，测试迭代100次。

在思源一号上运行
--------------------

::

   #!/bin/bash
   #SBATCH --job-name="mdtest"
   #SBATCH -N 64
   #SBATCH --ntasks-per-node=4
   #SBATCH -p 64c512g
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc openmpi ior
   export UCX_NET_DEVICES=mlx5_0:1
   export OMPI_MCA_btl=^openib

   mpirun mdtest -F -L -z 4 -b 2 -I 4096 -i 100 -u

在Pi2.0上运行
-----------------

::

   #!/bin/bash
   #SBATCH --job-name="mdtest"
   #SBATCH -N 64
   #SBATCH --ntasks-per-node=4
   #SBATCH -p cpu
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc openmpi ior

   mpirun mdtest -F -L -z 4 -b 2 -I 4096 -i 100 -u
