.. _fsl:

FSL
===

简介
----
RoseTTAFold 是一个 "三轨" 神经网络（"three-track" neural network），能同时考虑一维蛋白质中的氨基酸序列模式、二维蛋白质的氨基酸之间如何相互作用以及蛋白质可能出现的三维结构。

在这种网络架构中，蛋白质的一维、二维和三维信息之间能够来回流动，互通有无，使神经网络能够综合所有信息，共同推理出蛋白质的化学组成部分和其折叠结构之间的关系。

运行RoseTTAFold的方式
---------------------

构建运行目录：

.. code:: bash
      
   $ mkdir ~/run_rosettafold   
   $ cd ~/run_rosettafold                             //存放input.fasta文件的目录
   $ mkdir output                                     //程序运行结束后，数据最终的存放目录
   $ ls ~/run_rosettafold/RoseTTAFold/example/output  //临时文件生成目录


使用sbatch提交运行脚本(rosettafold.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH -J rosettafold
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=6
   #SBATCH --gres=gpu:1   

   module load rosettafold/1-python-3.8
   run_pyrosetta ~/run_rosettafold input.fasta output

使用如下指令提交：

.. code:: bash
   
   $ sbatch rosettafold.slurm
