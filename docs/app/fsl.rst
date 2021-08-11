.. _fsl:

FSL
===

简介
----
FSL是功能磁共振成像、核磁共振成像和DTI脑成像数据的综合分析工具库。

运行FSL的方式
-------------

使用sbatch提交运行脚本(fsl.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=fsl
   #SBATCH --partition=cpu    
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load fsl/6.0-fsl-gcc-4.8.5
   fsl $PWD run.sh

脚本run.sh示例如下（fsl.slurm、run.sh和数据要在同一目录下）:
   
.. code:: bash

   #!/bin/bash
   eddy_correct DWI.nii.gz DWI_eddy.nii.gz 0

使用如下指令提交：

.. code:: bash
   
   $ sbatch fsl.slurm

可视化运行方式(在Studio里的远程桌面运行)
----------------------------------------

.. code:: bash

   module load fsl/6.0-fsl-gcc-4.8.5
   fsl_gui
