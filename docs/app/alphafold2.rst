AlphaFold2
=============

AlphaFold2 基于深度神经网络预测蛋白质形态，能够快速生成高精确度的蛋白质 3D 模型。以往花费几周时间预测的蛋白质结构，AlphaFold2 在几小时内就能完成。

AlphaFold2 在 AI 平台的部署
----------------------------------------

交大 AI 平台部署了 AlphaFold2 镜像，镜像与参考数据路径如下：

* AlphaFold2 镜像：``/scratch/share/AlphaFold/alphafold.sif``
* 参考数据库（2.2 TB）： ``/scratch/share/AlphaFold/data``


使用前准备
---------------------------

准备一：设置环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~

在自己 home 下新建一系列文件夹，并将数据和镜像文件建立软链接：

.. code:: bash

	mkdir -p $HOME/alphafold
	mkdir -p $HOME/alphafold/img
	mkdir -p $HOME/alphafold/all
	mkdir -p $HOME/alphafold/all/data
	mkdir -p $HOME/alphafold/all/output
	ln -s /scratch/share/AlphaFold/data/* $HOME/alphafold/all/data/
	ln -s /scratch/share/AlphaFold/alphafold.sif $HOME/alphafold/img/

准备二：run.sh 文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~

AlphaFold 需要 run.sh 文件。可参考下方内容，新建和调整 run.sh 文件。

注意，通过 fasta_paths 参数传入 protein 的 fasta 文件，若有多条序列，每个 fasta 文件需要存放一条序列，文件之间使用逗号分割。

.. code:: bash

    #!/bin/bash

    cd /app/alphafold
    python run_alphafold.py \
    --preset=casp14   \
    --fasta_paths=/mnt/N.fasta  \
    --max_template_date=2020-05-14   \
    --output_dir=/mnt/output_here  \
    --model_names=model_1,model_2,model_3,model_4,model_5  \
    --data_dir=/mnt/alphafold/scripts/data/ \
    --uniref90_database_path=/mnt/alphafold/scripts/data/uniref90/uniref90.fasta \
    --mgnify_database_path=/mnt/alphafold/scripts/data/mgnify/mgy_clusters.fa \
    --uniclust30_database_path=/mnt/alphafold/scripts/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --bfd_database_path=/mnt/alphafold/scripts/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --pdb70_database_path=/mnt/alphafold/scripts/data/pdb70/pdb70 \
    --template_mmcif_dir=/mnt/alphafold/scripts/data/pdb_mmcif/mmcif_files \
    --obsolete_pdbs_path=/mnt/alphafold/scripts/data/pdb_mmcif/obsolete.dat

运行 AlphaFold2
---------------------

选用下方两种方式之一来运行 AlphaFold2。交互模式适合参数调试，sbatch 模式适合正式运行作业。

方式一：交互模式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用下方命令申请 1 张 GPU 卡（含 6 个 CPU）：

.. code:: bash

    salloc --ntasks-per-node=1 --job-name=alpha-session -p dgx2 --gres=gpu:1 -N 1

再 ssh 登录到分配的 vol 开头的 GPU 节点（如信息 ``salloc: Nodes vol01 are ready for job``）：

.. code:: bash

    ssh vol01    # 具体节点号以屏幕显示为准

接下来可在这张 GPU 卡上进行交互模式的软件测试。在命令行里输入下方内容运行 AlphaFold：

.. code:: bash

    AlphaFold_PATH=$PWD/alphafold
    IMAGE_PATH=$AlphaFold_PATH/img/alphafold.sif
    singularity exec --nv -B $AlphaFold_PATH/all:/mnt $IMAGE_PATH /mnt/run.sh



方式二：sbatch 脚本提交模式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

调试完成后，推荐使用 sbatch 方式提交作业脚本进行计算。

作业脚本示例（假设作业脚本名为 alpha.slurm）：

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
    
    AlphaFold_PATH=$PWD/alphafold
    IMAGE_PATH=$AlphaFold_PATH/img/alphafold.sif
    singularity exec --nv -B $AlphaFold_PATH/all:/mnt $IMAGE_PATH /mnt/run.sh


作业提交命令：

.. code:: bash

    sbatch alpha.slurm


注意事项
----------------------

调试时，推荐使用交互模式。调试全部结束后，请退出交互模式的计算节点，避免持续计费。可用 squeue 或 sacct 命令核查交互模式的资源使用情况。

欢迎邮件联系我们，反馈软件使用情况，或提出宝贵建议。

参考资料
----------------

- AlphaFold GitHub: https://github.com/deepmind/alphafold
- AlphaFold 主页: https://deepmind.com/research/case-studies/alphafold
- AlphaFold Nature 论文: https://www.nature.com/articles/s41586-021-03819-2




