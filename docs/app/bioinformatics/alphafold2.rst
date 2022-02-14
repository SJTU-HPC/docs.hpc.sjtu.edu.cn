AlphaFold2
=============

AlphaFold2 基于深度神经网络预测蛋白质形态，能够快速生成高精确度的蛋白质 3D 模型。以往花费几周时间预测的蛋白质结构，AlphaFold2 在几小时内就能完成

我们对 AlphaFold 进行持续优化，欢迎了解我们的优化工作：`ParaFold: Paralleling AlphaFold for Large-Scale Predictions <https://arxiv.org/abs/2111.06340>`__

AlphaFold2 三大版本
----------------------------------------

交大计算平台提供 AlphaFold2 三大版本：module 标准版、ParaFold、ColabFold。三个版本在思源一号和 π 集群上均可使用，且都支持复合体计算：

* module 标准版，加载即用，免除安装困难。可满足大部分计算需求

* ParaFold，支持 CPU、GPU 分离计算，适合大规模批量计算

* ColabFold，快速计算，含有多种功能，由 Sergey Ovchinnikov 等人开发


使用前准备
----------------------------------------

* 新建文件夹，如 ``alphafold``。

* 在文件夹里放置一个 ``fasta`` 文件。例如 ``test.fasta`` 文件（内容如下）：

（单体 fasta 文件示例）

.. code:: bash

    >2LHC_1|Chain A|Ga98|artificial gene (32630)
    PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK

（复合体 fasta 文件示例）

.. code:: bash

    >2MX4
    PTRTVAISDAAQLPHDYCTTPGGTLFSTTPGGTRIIYDRKFLLDR
    >2MX4
    PTRTVAISDAAQLPHDYCTTPGGTLFSTTPGGTRIIYDRKFLLDR
    >2MX4
    PTRTVAISDAAQLPHDYCTTPGGTLFSTTPGGTRIIYDRKFLLDR

版本一：module 标准版
----------------------------------------

module 标准版加载后即可使用，支持复合体计算

module 在思源一号上运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本示例（假设作业脚本名为 ``sub.slurm``）：

**单体**

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=colabfold
    #SBATCH --partition=a100
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1          # use 1 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load alphafold

    af2.1 \
    --fasta_paths=test.fasta \
    --max_template_date=2020-05-14 \
    --model_preset=monomer \
    --output_dir=output

然后使用 ``sbatch sub.slurm`` 语句提交作业。

**复合体**

.. code:: bash

    （slurm 脚本开头的 12 行内容跟上方一样，请自行补齐）

    af2.1 \
    --fasta_paths=test.fasta \
    --max_template_date=2020-05-14 \
    --model_preset=multimer \
    --is_prokaryote_list=false \
    --output_dir=output 

module 在 π 集群上运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**单体**

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=alphafold
    #SBATCH --partition=dgx2
    #SBATCH -N 1
    #SBATCH -x vol04,vol05
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1          # use 1 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load alphafold

    af2.1 \
    --fasta_paths=ha.fasta \
    --max_template_date=2020-05-14 \
    --model_preset=monomer \
    --output_dir=output

然后使用 ``sbatch sub.slurm`` 语句提交作业。

**复合体**

.. code:: bash

    （slurm 脚本开头的 13 行内容跟上方一样，请自行补齐）

    af2.1 \
    --fasta_paths=test.fasta \
    --max_template_date=2020-05-14 \
    --model_preset=multimer \
    --is_prokaryote_list=false \
    --output_dir=output 


module 使用说明
~~~~~~~~~~~~~~~~~~~~~~~~

* 单体计算可选用 monomer, monomer_ptm, 或 monomer_casp14
  
* 需严格按照推荐的参数内容和顺序运行（调换参数顺序或增删参数条目均可能导致报错）。若需使用更多模式，请换用另外三个版本的 AlphaFold

* 更多使用方法及讨论，请见水源文档 `AlphaFold & ColabFold <https://notes.sjtu.edu.cn/s/ielJnqiwX/>`__

版本二：ParaFold
----------------------------------------

ParaFold 为交大开发的适用于大规模计算的 AlphaFold 集群版，可选 CPU 与 GPU 分离计算，并支持 Amber 选择、module 选择、Recycling 次数指定等多个实用功能。ParaFold 并不改变 AlphaFold 计算内容和参数本身，所以在计算结果及精度上与 AlphaFold 完全一致

ParaFold (又名 ParallelFold) 将原本全部运行于 GPU 的计算，分拆为 CPU 和 GPU 两阶段进行。先至 CPU 节点完成 MSA 计算，再用 GPU 节点完成模型预测。这样既能节省 GPU 资源，又能加快运算速度

ParaFold GitHub：`https://github.com/Zuricho/ParallelFold <https://github.com/Zuricho/ParallelFold>`_ 

介绍网站：`https://parafold.sjtu.edu.cn <https://parafold.sjtu.edu.cn/>`__


ParaFold 在思源一号上运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 ParaFold

.. code:: bash

    git clone https://github.com/Zuricho/ParallelFold.git
    cd ParallelFold
    chmod +x run_alphafold.sh

使用下方``sub.slurm``脚本直接运行：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=parafold
    #SBATCH --partition=a100
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1          # use 1 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge

    singularity run --nv /dssg/share/imgs/ai/fold/1.0.sif \
    ./run_alphafold.sh \
    -d /dssg/share/data/alphafold \
    -o output \
    -p monomer \
    -i input/GA98.fasta \
    -t 2021-07-27 \
    -m model_1 -f



ParaFold 在 π 集群上运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 ParaFold

.. code:: bash

    git clone https://github.com/Zuricho/ParallelFold.git
    cd ParallelFold
    chmod +x run_alphafold.sh

使用下方``sub.slurm``脚本直接运行：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=parafold
    #SBATCH --partition=dgx2
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1          # use 1 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    singularity run --nv /lustre/share/img/ai/fold.sif \
    ./run_alphafold.sh \
    -d /scratch/share/AlphaFold/data \
    -o output \
    -p monomer_ptm \
    -i input/GA98.fasta \
    -t 2021-07-27 \
    -m model_1 -f


版本三：ColabFold
----------------------------------------

ColabFold 为 Sergey Ovchinnikov 等人开发的适用于 Google Colab 的 AlphaFold 版本，使用 MMseqs2 替代 Jackhmmer，且不使用模版。ColaFold 计算迅速，短序列五六分钟即可算完。

ColabFold 使用请至交大超算文档页面： :doc:`colabfold` 

构建自己的 AlphaFold 镜像
--------------------------

交大镜像平台提供了AlphaFold-2.1.1的 `docker 镜像 <https://hub.sjtu.edu.cn/repository/x86/alphafold>`_。

使用 ``singularity pull`` 命令可以下载该镜像：

.. code:: console

    singularity pull docker://sjtu.edu.cn/x86/alphafold:<tag>

镜像将被保存为 ``alphafold_<tag>.sif`` 文件。

镜像脚本示例如下：

.. code:: bash
    
    #!/bin/bash

    #SBATCH -J run_af
    #SBATCH -p a100
    #SBATCH -o %j.out
    #SBATCH -e %j.err
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1

    module purge
    
    singularity run --nv ${YOUR_IMAGE_PATH} python /app/alphafold/run_alphafold.py 
        --fasta_paths=${YOU_FASTA_FILE_DIR}  \
        --max_template_date=2020-05-14      \
        --bfd_database_path=${YOUR_DATA_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt  \
        --data_dir=${YOUR_DATA_DIR} \
        --output_dir=${YOU_OUTPUT_DIR} \
        --uniclust30_database_path=${YOUR_DATA_DIR}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
         --uniref90_database_path=${YOUR_DATA_DIR}/uniref90/uniref90.fasta \
         --mgnify_database_path=${YOUR_DATA_DIR}/mgnify/mgy_clusters.fa \
         --template_mmcif_dir=${YOUR_DATA_DIR}/pdb_mmcif/mmcif_files \
         --obsolete_pdbs_path=${YOUR_DATA_DIR}/pdb_mmcif/obsolete.dat \
         --pdb70_database_path=${YOUR_DATA_DIR}/pdb70/pdb70



参考资料
----------------
- ParaFold GitHub https://github.com/Zuricho/ParallelFold
- ParaFold 论文：https://arxiv.org/abs/2111.06340
- ParaFold 网站：https://parafold.sjtu.edu.cn
- AlphaFold GitHub: https://github.com/deepmind/alphafold
- AlphaFold 论文: https://www.nature.com/articles/s41586-021-03819-2
- ColabFold GitHub: https://github.com/sokrypton/ColabFold
- LocalColabFold GitHub: https://github.com/YoshitakaMo/localcolabfold
- 交大AlphaFold镜像：https://hub.sjtu.edu.cn/repository/x86/alphafold

