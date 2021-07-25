AlphaFold2
=============

AlphaFold2 基于深度神经网络预测蛋白质形态，能够快速生成高精确度的蛋白质 3D 模型。以往花费几周时间预测的蛋白质结构，AlphaFold2 在几小时内就能完成。

AlphaFold2 版本
----------------------------------------

交大 AI 平台部署了 AlphaFold2 的 module，更新日期：2021 年 7 月 25 日

.. code:: bash

    alphafold/2-python-3.8


使用前准备
---------------------------

* 新建文件夹，如 ``alphafold``。

* 在文件夹里放置一个 ``fasta`` 文件。例如 ``T1024.fasta`` 文件（内容如下）：

.. code:: bash

    >T1024 LmrP, , 408 residues|
    MKEFWNLDKNLQLRLGIVFLGAFSYGTVFSSMTIYYNQYLGSAITGILLALSAVATFVAGILAGFFADRNGRKPVMVFGTIIQLLGAALAIASNLPGHVNPWSTFIAFLLISFGYNFVITAGNAMIIDASNAENRKVVFMLDYWAQNLSVILGAALGAWLFRPAFEALLVILLLTVLVSFFLTTFVMTETFKPTVKVDEKAENIFQAYKTVLQDKTYMIFMGANIATTFIIMQFDNFLPVHLSNSFKTITFWGFEIYGQRMLTIYLILACVLVVLLMTTLNRLTKDWSHQKGFIWGSLFMAIGMIFSFLTTTFTPIFIAGIVYTLGEIVYTPSVQTLGADLMNPEKIGSYNGVAAIKMPIASILAGLLVSISPMIKAIGVSLVLALTEVLAIILVLVAVNRHQKTKLN


运行 AlphaFold2
---------------------

选用下方两种方式之一来运行 AlphaFold2。交互模式适合参数调试，sbatch 模式适合正式运行作业。

方式一：交互模式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用下方命令申请 1 张 GPU 卡（含 6 个 CPU），然后 ssh 登录到分配的 GPU 节点：

.. code:: bash

    salloc --ntasks-per-node=1 --job-name=alpha-session -p dgx2 --gres=gpu:1 -N 1
    ssh vol01    # 具体节点号以屏幕显示为准（如信息 ``salloc: Nodes vol01 are ready for job``）


接下来可在这张 GPU 卡上进行交互模式的软件测试。

调用 AlphaFold：

.. code:: bash

    module load alphafold

运行 AlphaFold:

.. code:: bash

    run_alphafold $PWD --preset=casp14 --fasta_paths=/mnt/T1024.fasta --max_template_date=2020-05-14  -output_dir=/mnt/output


根据需要，修改 ``--fasta_paths`` 和 ``--output_dir`` 参量，指定 ``fasta`` 文件，及结果文件夹。

``$PWD``指 AlphaFold 主文件夹，可用绝对路径代替，以便从其他路径运行上述命令。 


方式二：sbatch 脚本提交模式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

调试完成后，推荐使用 sbatch 方式提交作业脚本进行计算。

作业脚本示例（假设作业脚本名为 ``alpha.slurm``）：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=alphafold
    #SBATCH --partition=dgx2
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:1
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    
    module load alphafold

    run_alphafold $PWD --preset=casp14 --fasta_paths=/mnt/T1024.fasta --max_template_date=2020-05-14  -output_dir=/mnt/output


作业提交命令：

.. code:: bash

    sbatch alpha.slurm


注意事项
----------------------

* 调试时，推荐使用交互模式。调试全部结束后，请退出交互模式的计算节点，避免持续计费。可用 ``squeue`` 或 ``sacct`` 命令核查交互模式的资源使用情况。

* 欢迎邮件联系我们，反馈软件使用情况，或提出宝贵建议。

* 我们将紧随 AlphaFold 官方更新。

* 我们近期也会部署 RoseTTAFold，敬请关注。

参考资料
----------------

- AlphaFold GitHub: https://github.com/deepmind/alphafold
- AlphaFold 主页: https://deepmind.com/research/case-studies/alphafold
- AlphaFold Nature 论文: https://www.nature.com/articles/s41586-021-03819-2


