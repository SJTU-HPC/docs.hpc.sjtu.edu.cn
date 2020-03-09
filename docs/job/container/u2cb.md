# <center>非特权用户容器构建</center>

-----------

U2BC是上海交通大学高性能计算中心自行研发的非特权用户容器构建服务了。在集群上普通用户可以使用U2BC自行构建Singularity镜像。

## 容器构建流程

镜像创建，支持从[Docker Hub](https://hub.docker.com/)或者NVIDIA NGC(https://ngc.nvidia.com/)提供的镜像开始构建。如下指令，从`docker://ubuntu:latest`构建名为`ubuntu-test`的镜像。从`docker://nvcr.io/nvidia/pytorch:20.02-py3`构建名为`pytorch-test`的镜像。

```shell
u2cb create -n ubuntu-test -b docker://ubuntu:latest
u2cb create -n pytorch-test -b docker://nvcr.io/nvidia/pytorch:20.02-py3
```

也可以根据定义文件（define file）来进行容器构建（推荐）。

根据基础镜像的大小和构建流程，创建过程需要一定的时间。完成镜像创建后，可以使用如下指令进行镜像查询。

```shell
u2cb list
ubuntu-test pytorch-test
```

如需要与镜像进行交互，可以使用如下指令连接至容器中，在容器中可以使用root权限进行软件安装等特权行为， 比如`apt install`：

```shell
$ u2cb connect -n ubuntu-test
Warning: Permanently added '[172.16.12.179]:38111' (ECDSA) to the list of known hosts.
Welcome to Ubuntu 18.04.4 LTS (GNU/Linux 3.10.0-1062.9.1.el7.x86_64 x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

This system has been minimized by removing packages and content that are
not required on a system that users do not log into.

To restore this content, you can run the 'unminimize' command.
Last login: Mon Mar  2 08:41:39 2020 from 172.16.0.163
root@centos77-300GB:~# apt install relion
```

可以使用如下指令可以将镜像从构建服务器上打包并下载到本地`./ubuntu-test.simg`，然后可以在集群环境中使用该镜像。

```shell
u2cb download -n ubuntu-test
singularity shell ubuntu-test.simg
```

使用如下指令删除在构建服务器上的镜像文件。

```shell
u2cb delete -n ubuntu-test
```
