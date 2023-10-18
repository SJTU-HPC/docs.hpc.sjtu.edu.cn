RoseTTAFold
=============

简介
-----
RoseTTAFold 由华盛顿大学 David Baker 团队开发，利用深度学习技术准确、快速地预测蛋白质结构。

RoseTTAFold 版本
----------------------------------------

交大 AI 平台部署了两个版本的 RoseTTAFold：
- 1.0版本：最新更新日期为2021 年 7 月 31 日
- 1.1版本：最新更新日期为2023 年 10 月 16 日

1. 1.0版本使用说明
----------------------

.. code:: bash

    rosettafold/1-python-3.8


1.1 使用前准备
++++++++++++++++++

* 新建文件夹，如 ``rosettafold``。

* 在文件夹里放置一个 ``fasta`` 文件。例如 ``test.fasta`` 文件（内容如下）：

.. code:: bash

    >2MX4
    PTRTVAISDAAQLPHDYCTTPGGTLFSTTPGGTRIIYDRKFLLDR

* 另外，还要在文件夹里新建一个输出文件夹，如 ``output``，确保文件夹里为空。     

1.2 运行 RoseTTAFold
+++++++++++++++++++++++++++

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
    
    module load rosettafold/1-python-3.8

    run_pyrosetta $PWD test.fasta output


说明：

* 可修改 ``test.fasta`` 和 ``output``，指定输入文件和输出文件夹。

* $PWD 指当前路径，也可以用绝对路径指定 RoseTTAFold 的主文件夹，以便从其他路径运行上述命令。 


作业提交命令：

.. code:: bash

    sbatch rosettafold.slurm


1.3 注意事项
++++++++++++++++

* 上述示例运行约需 1 个小时。

* 欢迎邮件联系我们，反馈软件使用情况，或提出宝贵建议。

2. 1.1版本使用说明
-------------------------

此版本无需module加载rosettafold，直接使用conda镜像，保持了跟官方文档一致的命令使用方式。

2.1 使用前准备
+++++++++++++++++++

如果需要经常使用RoseTTAFold,可以在~/.bashrc文件中添加如下变量：

.. code:: bash

    export RoseTTAFold=/lustre/opt/contribute/cascadelake/RoseTTAFold/data/RoseTTAFold_1.1

2.2 运行RoseTTAFold
+++++++++++++++++++++++

作业脚本示例：

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

    source activate /lustre/share/conda_env/RoseTTAFold
    export RoseTTAFold=/lustre/opt/contribute/cascadelake/RoseTTAFold/data/RoseTTAFold_1.1

    /bin/bash $RoseTTAFold/run_pyrosetta_ver.sh $RoseTTAFold/example/input.fa output


说明：

* 可修改 ``/example/input.fa`` 为指定输入文件， ``output`` 为指定输出文件夹。

* 如果已经在~/.bashrc文件中添加变量RoseTTAFold，脚本中无需重复添加。

* 所有RoseTTAFold涉及命令均与官方文档保持一致，可以自行参考使用


参考资料
----------------

- RoseTTAFold GitHub: https://github.com/RosettaCommons/RoseTTAFold
- RoseTTAFold 论文: https://www.biorxiv.org/content/10.1101/2021.06.14.448402v1

