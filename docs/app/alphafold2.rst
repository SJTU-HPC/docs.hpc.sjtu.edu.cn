AlphaFold2
=============

AlphaFold2 基于深度神经网络预测蛋白质形态，能够快速生成高精确度的蛋白质 3D 模型。以往花费几周时间预测的蛋白质结构，AlphaFold2 在几小时内就能完成。

AlphaFold2 版本
----------------------------------------

交大 AI 平台部署了 AlphaFold2 的 module，最新更新日期：2021 年 7 月 25 日

.. code:: bash

    alphafold/2-python-3.8


使用前准备
---------------------------

* 新建文件夹，如 ``alphafold``。

* 在文件夹里放置一个 ``fasta`` 文件。例如 ``histone_H3.fasta`` 文件（内容如下）：

.. code:: bash

    >sp|P68431|H31_HUMAN Histone H3.1 OS=Homo sapiens OX=9606 GN=H3C1 PE=1 SV=2
    MARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPGTVALREIRRYQKSTE
    LLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEACEAYLVGLFEDTNLCAIHAKRVTI
    MPKDIQLARRIRGERA

运行 AlphaFold2
---------------------

作业脚本示例（假设作业脚本名为 ``alpha.slurm``）：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=alphafold
    #SBATCH --partition=dgx2
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    
    module load alphafold

    run_alphafold $PWD --preset=casp14 --fasta_paths=/mnt/histone_H3.fasta --max_template_date=2020-05-14 --output_dir=/mnt/output

说明：

* 可修改 ``--fasta_paths`` 和 ``--output_dir`` 参量，指定 ``fasta`` 文件，及结果文件夹。``/mnt`` 为容器封装路径，保留即可。

* $PWD 指当前路径，也可以用绝对路径指定 AlphaFold 的主文件夹，以便从其他路径运行上述命令。 

* 由于计算完成后 ``data`` 软链接会被删除，所以一个文件夹里建议最多同时运行一个作业。


作业提交命令：

.. code:: bash

    sbatch alpha.slurm


注意事项
----------------------

* 欢迎邮件联系我们，反馈软件使用情况，或提出宝贵建议。

* 我们将紧随 AlphaFold 官方更新。

* 我们近期也会部署 RoseTTAFold，敬请关注。

参考资料
----------------

- AlphaFold GitHub: https://github.com/deepmind/alphafold
- AlphaFold 主页: https://deepmind.com/research/case-studies/alphafold
- AlphaFold Nature 论文: https://www.nature.com/articles/s41586-021-03819-2


