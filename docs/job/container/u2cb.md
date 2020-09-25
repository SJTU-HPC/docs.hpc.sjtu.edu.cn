# <center>非特权用户容器构建</center>

-----------

U2CB是上海交通大学高性能计算中心自行研发的非特权用户容器构建平台。在集群上普通用户可以使用U2CB自行构建Singularity镜像。

## 容器构建流程

### 镜像创建

支持从[Docker Hub](https://hub.docker.com/)或者[NVIDIA NGC](https://ngc.nvidia.com/)提供的镜像开始构建。如下指令，从`docker://ubuntu:latest`构建名为`ubuntu-test`的镜像。从`docker://nvcr.io/nvidia/pytorch:20.02-py3`构建名为`pytorch-test`的镜像。

```shell
$ u2cb create -n ubuntu-test -b docker://ubuntu:latest
$ u2cb create -n pytorch-test -b docker://nvcr.io/nvidia/pytorch:20.02-py3
```

### 从定义文件构建镜像创建（推荐）

可以参考Singularity的[Definition Files](https://sylabs.io/guides/3.5/user-guide/definition_files.html)编写您的定义文件。

例如，在您的本地编辑定义文件`test.def`，内容为：

```
Bootstrap: docker
From: ubuntu

%post
    apt update && apt install -y gcc

%enviroment
    export TEST_ENV_VAR=SJTU
```

然后使用u2cb进行镜像构建：

```shell
$ u2cb defcreate -n ubuntu-test -d ./test.def
```

### 镜像查询

完成镜像创建后，可以使用如下指令进行镜像查询。

```shell
$ u2cb list
ubuntu-test pytorch-test
```

### 与镜像进行交互

如需要与镜像进行交互，可以使用如下指令连接至容器中，在容器中可以使用root权限进行软件安装等特权行为， 以ubuntu为例，比如`apt install`：

```shell
$ u2cb connect -n ubuntu-test
Singularity> whoami
root
Singularity> apt update && apt install -y gcc
```

!!! tip
    1. 请勿将任何应用安装在`/root`下（因容器在集群上运行时为普通用户态，`/root`不会被打包），推荐直接安装在系统目录或者`/opt`下。
    2. 运行应用所需的环境变量可以添加到`/enviroment`文件中。
        ```shell
        Singularity> echo "export TEST_ENV_VAR=SJTU" >> /environment
        Singularity> echo "export PATH=/opt/app/bin:$PATH" >> /environment
        ```

### 镜像下载

可以使用如下指令可以将镜像从构建服务器上打包并下载到本地`./ubuntu-test.simg`，然后可以在集群环境中使用该镜像，详细可见[容器](../singularity/#_2)一节。

```shell
$ u2cb download -n ubuntu-test
$ srun -p small -n 1 --pty singularity shell ubuntu-test.simg
```

### 镜像删除

使用如下指令删除在构建服务器上的镜像文件。

```shell
$ u2cb delete -n ubuntu-test
```

### U2CB Shell

U2CB还支持用户通过`u2cb shell`登录U2CB Server，进行镜像查询，镜像交互，镜像删除的功能。

```shell
$ u2cb shell
(U2CB Server) > help

Documented commands (type help <topic>):
========================================
create  delete  help  list  shell

(U2CB Server) > help list

        Use `list` to see all containers
        Use `list def` to see all define files
        Use `list img` to see all image files

(U2CB Server) > list def
```

## 参考文献
 - [Singularity Quick Start](https://sylabs.io/guides/3.4/user-guide/quick_start.html)
 - [Docker Hub](https://hub.docker.com/)
 - [NVIDIA GPU CLOUD](https://ngc.nvidia.com/)
 - [Fakeroot feature of Singularity](https://sylabs.io/guides/3.5/user-guide/fakeroot.html)
