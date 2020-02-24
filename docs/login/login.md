# 登陆节点

----------

通过SSH协议连接集群后，您的本地环境是与集群登陆节点相连的。集群有多个登陆节点（login1.hpc.sjtu.edu.cn/login2.hpc.sjtu.edu.cn/login3.hpc.sjtu.edu.cn），您可以选择任意一个登陆节点。或者您也可以使用登陆节点的自动路由域名(login.hpc.sjtu.edu.cn)，该域名会随机将您的连接分配到其中一个登陆节点。

## 使用

!!! suggetion
    不要在登录节点上运行计算或内存密集型应用程序。这些节点是共享资源。 管理员可能会终止对其他用户或系统有负面影响的进程。我们也在登陆节点配置了资源限制服务，如果您的程序被检测到影响其他用户，可能会暂时封禁您的账号。

- 编译代码（限制在make -j 2）
- 编辑文件
- 提交作业

某些工作流程需要交互式使用，如IDL，NCL，python和ROOT等应用程序。请参考slurm文档中的[`srun`指令](https://docs.hpc.sjtu.edu.cn/job/slurm/#srun-and-salloc)，交互式地提交作业。

具体的，使用如下指令启动远程主机bash终端。

```bash
srun -p cpu -n 1 --exclusive --pty /bin/bash
```

## 数据传输

如果您需要传输小批量数据，直接通过登陆节点传输即可。但对于大批量数据，请发送邮件至[hpc邮箱](mailto:hpc@sjtu.edu.cn)预约硬盘传输，并将硬盘邮寄或携带至我们的办公室，具体联系方式我们会通过邮件回复您。