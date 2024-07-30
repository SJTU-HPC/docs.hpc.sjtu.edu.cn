****
容器
****

容器是一种Linux上广为采用的应用封装技术，它将可执行程序与依赖库打包成一个镜像文件，启动时与宿主节点共享操作系统内核。
由于镜像文件同时携带可执行文件和依赖库，避免了两者不匹配造成的兼容性问题，还能在一个宿主Linux操作系统上支持多种不同的Linux发行版，譬如在CentOS发行版上运行Ubuntu的 ``apt-get`` 命令。

π 超算集群采用基于 `Singularity <https://sylabs.io/singularity/>`__  的高性能计算容器技术，相比Docker等在云计算环境中使用的容器技术，Singularity 同时支持root用户和非root用户启动，且容器启动前后，用户上下文保持不变，这使得用户权限在容器内部和外部都是相同的。
此外，Singularity 强调容器服务的便捷性、可移植性和可扩展性，而弱化了容器进程的高度隔离性，因此量级更轻，内核namespace更少，性能损失更小。

您可以在专门的容器构建节点定制Singularity镜像。


通过交互式Shell构建Singularity镜像
==================================

.. tip:: 构建Singularity容器镜像通常需要root特权，通常超算集群不支持这样的操作。π超算集群的“容器化的Singularity”允许用户编写、构建和传回自定义容器镜像。

在π超算集群上，我们采用“容器化的Singularity”，允许用户在一个受限的环境内以普通用户身份“模拟”root特权，保存成Singularity镜像，并将镜像传回集群使用。

首先从登录节点使用用户名 ``build`` 跳转到专门用于构建容器镜像的节点。
需要注意的是，X86节点(用于 ``cpu`` ``small`` ``huge`` 等队列)和国产ARM节点(用于 ``arm128c256g`` 队列)的处理器指令集是不兼容的，需使用对应的镜像构建节点。

.. tip:: 请选择与目标主机(x86或arm)相匹配的容器构建节点。

从登录节点跳转X86容器构建节点：

.. code:: console

   $ ssh build@container-x86
   $ hostname
   container-x86.pi.sjtu.edu.cn

从登录节点跳转ARM容器构建节点：

.. code:: console

   $ ssh build@container-arm
   $ hostname
   container-arm.pi.sjtu.edu.cn

.. caution:: 出于安全考虑， ``container-x86`` 和 ``container-arm`` 节点每天 **23:59** 重启节点并清空数据，请及时转移容器构建节点上的数据。``build`` 为共享用户，请勿修改自己工作目录外的数据，以免影响其他用户的使用。

由于所有用户共享使用 ``build`` 用户，需要创建专属工作目录，在工作目录中构建镜像。
我们使用 ``mktemp -d`` 命令在 ``/tmp`` 目录下创建名字带有随机字符的工作目录。

.. code:: console

   $ cd $(mktemp -d)
   $ pwd
   /tmp/tmp.sr7C5813M9


使用 ``docker images`` 查看本地可用的基础镜像列表。

.. code:: console

  $ docker images
  REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
  centos       8         5d0da3dc9764   4 weeks ago   231MB

使用 ``docker run -it IMAGE_ID`` 从基础镜像创建容器(container)实例，并以 ``root`` 身份进入容器内。

.. code:: console

  $ docker run -it --name=MY_USERNAME_DATE 5d0da3dc9764 /bin/bash

因为centos停止维护，初次进入镜像需要修改yum源，才可以正常使用yum命令。

.. code:: console

   $ sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS*.repo
   $ sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS*.repo
   $ yum makecache

然后以 ``root`` 特权修改容器内容，例如安装软件等。

.. code:: console

  [root@68bdb5af0da9 /]# whoami
  root
  [root@68bdb5af0da9 /]# yum check-update
  ...
  [root@68bdb5af0da9 /]# yum install tree
  ...
  [root@68bdb5af0da9 /]# tree --version
  tree v1.7.0 (c) 1996 - 2014 by Steve Baker, Thomas Moore, Francesc Rocher, Florian Sesser, Kyosuke Tokoro

操作结束后退出容器，回到 ``build`` 用户身份下。

.. code:: console

  [root@68bdb5af0da9 /]# exit
  [build@container-x86 ~]$ whoami
  build

使用 ``docker ps -a`` 查看与先前定义名字对应的container ID，在这个示例中是 ``MY_USERNAME_DATE`` 。

.. code:: console

  [build@container-x86 ~]$ docker ps -a
  CONTAINER ID   IMAGE          COMMAND        CREATED         STATUS                     PORTS     NAMES
  515e913f12cb   5d0da3dc9764   "/bin/bash"    4 seconds ago   Exited (0) 2 seconds ago             MY_USERNAME_DATE

使用 ``docker commit CONTAINER_ID IMG_NAME`` 提交容器变更。

.. code:: console

  $ docker commit 515e913f12cb my_username_app_img

此时使用 ``docker images`` 可以在容器镜像列表中看到刚刚提交的容器变更。

.. code:: console

  $ docker images
  REPOSITORY            TAG       IMAGE ID       CREATED              SIZE
  my_username_app_img   latest    c26c43a0cc9b   About a minute ago   279MB

将Docker容器保存为可在超算平台上使用的Singularity镜像。

.. code:: console

  $ SINGULARITY_NOHTTPS=1 singularity build my_username_app_img.sif docker-daemon://my_username_app_img:latest
  INFO:    Starting build...
  INFO:    Creating SIF file...
  INFO:    Build complete: my_username_app_img.sif

使用 ``scp my_username_app_img.sif YOUR_USERNAME@pilogin1:~/`` 将Singularity镜像文件复制到超算集群家目录后，可以使用 ``singularity`` 命令测试镜像文件，从 ``/etc/redhat-release`` 内容和 ``tree`` 命令版本看，确实进入了与宿主操作系统不一样的运行环境。

.. code:: console

  $ singularity exec my_username_app_img.sif cat /etc/redhat-release
  CentOS Linux release 8.4.2105
  $ singularity exec my_username_app_img.sif tree --version
  tree v1.7.0 (c) 1996 - 2014 by Steve Baker, Thomas Moore, Francesc Rocher, Florian Sesser, Kyosuke Tokoro 


AI平台容器编译
===========================
与x86平台容器编译方式类似，在AI平台也可以 按需定制Singularity镜像。


通过交互式Shell构建AI应用镜像
--------------------------------------

.. tip:: 构建Singularity容器镜像通常需要root特权，通常超算集群不支持这样的操作。π超算集群的“容器化的Singularity”允许用户编写、构建和传回自定义容器镜像。

在π超算集群上，我们采用“容器化的Singularity”，允许用户在一个受限的环境内以普通用户身份“模拟”root特权，保存成Singularity镜像，并将镜像传回集群使用。

从登录节点跳转X86容器构建节点：

.. code:: console

   $ ssh build@container-x86
   $ hostname
   container-x86.pi.sjtu.edu.cn

.. caution:: 出于安全考虑， ``container-x86`` 和 ``container-arm`` 节点每天 **23:59** 重启节点并清空数据，请及时转移容器构建节点上的数据。``build`` 为共享用户，请勿修改自己工作目录外的数据，以免影响其他用户的使用。

由于所有用户共享使用 ``build`` 用户，需要创建专属工作目录，在工作目录中构建镜像。
我们使用 ``mktemp -d`` 命令在 ``/tmp`` 目录下创建名字带有随机字符的工作目录。

.. code:: console

   $ cd $(mktemp -d)
   $ pwd
   /tmp/tmp.sr7C5813M9


使用 ``docker images`` 查看本地可用的基础镜像列表。

.. code:: console

  $ docker images
  REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
  centos       8         5d0da3dc9764   4 weeks ago   231MB

使用 ``docker run -it IMAGE_ID`` 从基础镜像创建容器(container)实例，并以 ``root`` 身份进入容器内。

.. code:: console

  $ docker run -it --name=MY_USERNAME_DATE 5d0da3dc9764 /bin/bash

然后以 ``root`` 特权修改容器内容，例如安装软件等。

.. code:: console

  [root@68bdb5af0da9 /]# whoami
  root
  [root@68bdb5af0da9 /]# yum check-update
  ...
  [root@68bdb5af0da9 /]# yum install tree
  ...
  [root@68bdb5af0da9 /]# tree --version
  tree v1.7.0 (c) 1996 - 2014 by Steve Baker, Thomas Moore, Francesc Rocher, Florian Sesser, Kyosuke Tokoro

操作结束后退出容器，回到 ``build`` 用户身份下。

.. code:: console

  [root@68bdb5af0da9 /]# exit
  [build@container-x86 ~]$ whoami
  build

使用 ``docker ps -a`` 查看与先前定义名字对应的container ID，在这个示例中是 ``MY_USERNAME_DATE`` 。

.. code:: console

  [build@container-x86 ~]$ docker ps -a
  CONTAINER ID   IMAGE          COMMAND        CREATED         STATUS                     PORTS     NAMES
  515e913f12cb   5d0da3dc9764   "/bin/bash"    4 seconds ago   Exited (0) 2 seconds ago             MY_USERNAME_DATE

使用 ``docker commit CONTAINER_ID IMG_NAME`` 提交容器变更。

.. code:: console

  $ docker commit 515e913f12cb my_username_app_img

此时使用 ``docker images`` 可以在容器镜像列表中看到刚刚提交的容器变更。

.. code:: console

  $ docker images
  REPOSITORY            TAG       IMAGE ID       CREATED              SIZE
  my_username_app_img   latest    c26c43a0cc9b   About a minute ago   279MB

将Docker容器保存为可在超算平台上使用的Singularity镜像。

.. code:: console

  $ SINGULARITY_NOHTTPS=1 singularity build my_username_app_img.sif docker-daemon://my_username_app_img:latest
  INFO:    Starting build...
  INFO:    Creating SIF file...
  INFO:    Build complete: my_username_app_img.sif

使用 ``scp sample-x86.sif YOUR_USERNAME@login1:~/`` 将Singularity镜像文件复制到超算集群家目录后，可以使用 ``singularity`` 命令测试镜像文件，从 ``/etc/redhat-release`` 内容和 ``tree`` 命令版本看，确实进入了与宿主操作系统不一样的运行环境。

.. code:: console

  $ singularity exec my_username_app_img.sif cat /etc/redhat-release
  CentOS Linux release 8.4.2105
  $ singularity exec my_username_app_img.sif tree --version
  tree v1.7.0 (c) 1996 - 2014 by Steve Baker, Thomas Moore, Francesc Rocher, Florian Sesser, Kyosuke Tokoro 


参考资料
========

- Singularity Quick Start https://sylabs.io/guides/3.4/user-guide/quick_start.html
- Docker Hub https://hub.docker.com/
- NVIDIA GPU CLOUD https://ngc.nvidia.com/
- 更多 Singularity Definition Files 的例子请参考 https://github.com/SJTU-HPC/hpc-base-container/tree/dev/base/
