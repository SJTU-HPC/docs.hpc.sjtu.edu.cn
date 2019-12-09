# <center>VMD</center>

---------

VMD是一套分子建模与可视化软件，主要用来分析分子动力学模拟的实验数据。同时，软件也包含处理长度与提及相关数据的模块，能可视化与分析轨迹，添加任意图形，并能导出成其他软件能利用的格式例如POV-Ray，PRMan，VRML等。用户能运行Tcl和Python脚本进行批量操作，也可通过Tcl/Tk与其他程序进行交互。

## 使用HPC Studio启动VMD可视化界面

首先参照[可视化平台](../../login/HpcStudio/)开启远程桌面，并在远程桌面中启动终端，并输入以下指令：

```bash
singularity run /lustre/share/img/vmd.simg vmd
```

## 参考文献

- [VMD](https://www.ks.uiuc.edu/Research/vmd/)
- [Singularity文档](https://sylabs.io/guides/3.5/user-guide/)