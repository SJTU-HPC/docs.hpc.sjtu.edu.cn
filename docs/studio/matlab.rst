MATLAB使用说明
===============

启动远程桌面
-------------------------

使用hpc账号登录HPC studio（https://studio.hpc.sjtu.edu.cn）后，点击"Interactive Apps >> Desktop"。选择需要的核数，session时长（默认1核、1小时），点击"Launch"启动远程桌面。待选项卡显示作业在RUNNING的状态时,点击"Launch Desktop"即可进入远程桌面。

.. image:: ../img/matlab01.png
.. image:: ../img/matlab02.png

启动MATLAB
-------------------------

远程桌面中点击右键，选择Open Terminal Here打开终端，在终端中使用命令 "singularity run /lustre/share/img/matlab_latest.sif matlab"

启动后即可使用MATLAB R2021a

.. image:: ../img/matlab03.png
.. image:: ../img/matlab04.png
.. image:: ../img/matlab05.png