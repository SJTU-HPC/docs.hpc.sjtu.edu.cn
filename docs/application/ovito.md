# <center>OVITO</center> 

-----

## 简介

OVITO is a scientific visualization and analysis software for atomistic and particle simulation data. It helps scientists gain better insights into materials phenomena and physical processes. The program is freely available for all major platforms under an open source license. It has served in a growing number of computational simulation studies as a powerful tool to analyze, understand and illustrate simulation results.


## Pi 上的 OVITO

OVITO 需要在 HPC Studio 可视化平台上使用。Pi 登陆节点不支持 gnuplot 显示。

HPC Studio 可视化平台通过浏览器访问：https://studio.hpc.sjtu.edu.cn

浏览器需为 chrome, firefox 或 edge。


## 使用 OVITO

### 在 HPC Studio 上连接远程桌面

1. 浏览器打开 https://studio.hpc.sjtu.edu.cn

2. 顶栏 Interactive Apps 下拉菜单，选择第一个 Desktop

3. Desktop 里第一个 "Desktop Instance Size" 选择 16core-desktop（根据需求选择），然后点击 Launch

4. 等待几秒，甚或更长时间，取决于 small 队列可用资源量。Studio 的远程桌面以一个正常的 small 队列作业启动

5. 启动后，右上角会显示 1 node 16 core Running. 然后点击 Launch Desktop


###  远程桌面启动 gnuplot

在远程桌面空白处右键单击，Open Terminal Here 打开终端

```bash
$ module use /lustre/share/singularity/modules
$ module load ovito
$ ovito
```

###  结束后退出远程桌面

远程桌面作业，使用完毕后需退出，否则会持续计费。两种退出方法：

1. 在 Studio 界面上点 "Delete" 删除该作业

2. 或在 Pi 上用 squeue 查看作业，并用 scancel 终止该作业

![avater](../img/ovito.gif)




## 参考链接

- [gnuplot 官网](http://www.gnuplot.info/)

