AlphaFold
=============

AlphaFold2 基于深度神经网络预测蛋白质形态，能够快速生成高精确度的蛋白质 3D 模型。以往花费几周时间预测的蛋白质结构，AlphaFold2 在几小时内就能完成。

AlphaFold2 在 AI 平台的部署
----------------------------------------

交大 AI 平台部署了 AlphaFold2 镜像，镜像与参考数据路径如下：

AlphaFold2 镜像

.. code:: bash

    /lustre/share/singularity/aarch64/alphafold/alphafold.sif


参考数据（2.2 TB）

.. code:: bash

    /lustre/opt/contribute/cascadelake/AlphaFold/alphafold/scripts/data


使用前准备
----------------

准备一：设置环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~

在自己 home 下新建一系列文件夹，并将数据和镜像文件建立软链接：

.. code:: bash

    mkdir -p alphafold
    mkdir -p alphafold/img
    mkdir -p alphafold/all
    mkdir -p alphafold/all/data
    mkdir -p alphafold/all/output

    ln -s /lustre/opt/contribute/cascadelake/AlphaFold/alphafold/scripts/data/* alphafold/all/data
    ln -s /lustre/share/singularity/aarch64/alphafold/alphafold.sif alphafold/img/

准备二：run.sh 文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~

参考下方内容，编写和修改运行所需的 run.sh 文件：

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

选用下方两种方式之一来运行 AlphaFold2。参数调试推荐方式一，正式计算推荐方式二。

方式一：交互模式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

交互模式适合调试，使用如下命令申请 1 张 GPU卡（含 6 个 CPU）：

.. code:: bash

    salloc --ntasks-per-node=1 --job-name=alpha-session -p dgx2 --gres=gpu:1 -N 1

待屏幕显示分配的 DGX-2 节点后（如信息 salloc: Nodes vol01 are ready for job ），使用 ssh 登录到该节点：

.. code:: bash

    ssh vol01    # 具体节点号以屏幕显示为准

接下来可在命令行里直接计算（资源为 1 卡 + 6 CPU）。

AlphaFold 运行命令：

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

调试时，推荐使用方式一的交互模式。调试全部结束后，请退出交互模式的计算节点，避免持续计费。可用 squeue 或 sacct 命令核查交互模式的资源使用情况。

欢迎邮件联系我们，反馈使用情况，或提出宝贵建议。

参考资料
----------------

- AlphaFold GitHub: https://github.com/deepmind/alphafold
- AlphaFold 主页: https://deepmind.com/research/case-studies/alphafold
- AlphaFold Nature 论文: https://www.nature.com/articles/s41586-021-03819-2




