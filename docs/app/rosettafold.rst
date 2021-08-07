RoseTTAFold
=============

简介
-----
RoseTTAFold 由华盛顿大学 David Baker 团队开发，利用深度学习技术准确、快速地预测蛋白质结构。

RoseTTAFold 版本
----------------------------------------

交大 AI 平台部署了 RoseTTAFold 的 module，最新更新日期：2021 年 7 月 31 日

.. code:: bash

    rosettafold/1-python-3.8


使用前准备
---------------------------

* 新建文件夹，如 ``rosettafold``。

* 在文件夹里放置一个 ``fasta`` 文件。例如 ``test.fasta`` 文件（内容如下）：

.. code:: bash

    >2MX4
    PTRTVAISDAAQLPHDYCTTPGGTLFSTTPGGTRIIYDRKFLLDR

* 另外，还要在文件夹里新建一个输出文件夹，如 ``output``，确保文件夹里为空。     

运行 RoseTTAFold
---------------------

作业脚本示例（假设作业脚本名为 ``rosettafold.slurm``）：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=rosettafold
    #SBATCH --partition=dgx2
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -x vol08
    
    module purge
    module load rosettafold/1-python-3.8

    run_pyrosetta $PWD test.fasta output


说明：

* 可修改 ``test.fasta`` 和 ``output``，指定输入文件和输出文件夹。

* $PWD 指当前路径，也可以用绝对路径指定 RoseTTAFold 的主文件夹，以便从其他路径运行上述命令。 


作业提交命令：

.. code:: bash

    sbatch rosettafold.slurm


注意事项
----------------------

* 上述示例运行约需 1 个小时。

* 欢迎邮件联系我们，反馈软件使用情况，或提出宝贵建议。

参考资料
----------------

- RoseTTAFold GitHub: https://github.com/RosettaCommons/RoseTTAFold
- RoseTTAFold 论文: https://www.biorxiv.org/content/10.1101/2021.06.14.448402v1

