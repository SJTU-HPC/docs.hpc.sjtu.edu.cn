AlphaFold2
=============

AlphaFold2 基于深度神经网络预测蛋白质形态，能够快速生成高精确度的蛋白质 3D 模型。以往花费几周时间预测的蛋白质结构，AlphaFold2 在几小时内就能完成。

我们对 AlphaFold 持续优化，可至 ParaFold 网站了解我们的工作：`https://parafold.sjtu.edu.cn <https://parafold.sjtu.edu.cn/>`__

我们将于 9 月 15 日（周三）在闵行校区图书信息楼 9 楼举办《AlphaFold 使用与优化》专题培训，欢迎大家参加：`报名问卷 <https://wj.sjtu.edu.cn/q/KCZDA5VQ>`__ 

AlphaFold2 四大版本
----------------------------------------

交大 AI 平台提供四大 AlphaFold 版本

* module 版，最新更新日期：2021 年 9 月 12 日。加载即用，免除安装困难。可满足大部分计算需求；

* conda 版，支持自定义模型、PTM计算、数据集路径、Recycling 次数等，支持实时更新；

* ColabFold 版，快速计算，含有多种功能，由 Sergey Ovchinnikov 开发。可在交大 DGX-2 上通过 conda 安装使用；

* ParallelFold 版，支持 CPU、GPU 分离计算，适合大规模批量计算。
  

版本一：module
----------------------------------------

module 版为全局部署的 ``alphafold/2-python-3.8``，更新日期：2021 年 9 月 12 日

module 使用前准备
~~~~~~~~~~~~~~~~~~~~~~~~

* 新建文件夹，如 ``alphafold``。

* 在文件夹里放置一个 ``fasta`` 文件。例如 ``test.fasta`` 文件（内容如下）：

.. code:: bash

    >sp|P68431|H31_HUMAN Histone H3.1 OS=Homo sapiens OX=9606 GN=H3C1 PE=1 SV=2
    MARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPGTVALREIRRYQKSTE
    LLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEACEAYLVGLFEDTNLCAIHAKRVTI
    MPKDIQLARRIRGERA

module 运行
~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本示例（假设作业脚本名为 ``alpha.slurm``）：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=alphafold
    #SBATCH --partition=dgx2
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1          # use 1 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load alphafold

    run_af2  $PWD --preset=casp14  test.fasta  --max_template_date=2021-09-12

module 作业提交
~~~~~~~~~~~~~~~~~~~~~~~~

采用下方语句提交 AlphaFold 作业

.. code:: bash

    sbatch alpha.slurm    

module 说明
~~~~~~~~~~~~~~~~~~~~~~~~

* 资源建议：对于 500AA 以下的蛋白，推荐使用 1 块 GPU 卡；对于更大的序列，推荐使用 2 块 GPU 卡。对于 1400AA 以上的序列，3 块或 4 块卡也无法加快计算，强烈建议使用下方的 conda 安装方法计算。

* 2021年7月本文档的用法依然支持，主程序名为 ``run_alphafold``，路径含有 ``/mnt``。现为 ``run_af2``，路径不再含有 ``/mnt``。

版本二：conda
----------------------------------------

conda 方法更为灵活，支持自定义修改，如选取计算 5 CASP14 models 和 5 pTM models 的全部或不放、修改 Recycling 次数、选择是否 Amber 优化、设定 data 数据集位置等。

conda 版的 AlphaFold 安装较为复杂，建议对 conda 较为熟悉的用户尝试。如有问题，欢迎邮件联系我们。

conda 安装步骤
~~~~~~~~~~~~~~~~~~~~~~~~

AlphaFold 支持 cuda 10 和 11，vol01-07 为 cuda 10，所以接下来我们以 cuda 10 为例介绍安装。

1. 下载官方 AlphaFold
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone https://github.com/deepmind/alphafold.git

由于 git 访问不太稳定，推荐先将 GitHub zip 文件下载至本地，再上传至集群。

然后下载 ``stereo_chemical_props.txt`` 文件，放至 ``$ALPHAFOLD/alphafold/common`` 文件夹：

.. code:: bash

    wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
    mv stereo_chemical_props.txt $ALPHAFOLD/alphafold/common

2. 申请 GPU 计算节点
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    salloc --ntasks-per-node=1 -p dgx2 --gres=gpu:1 -N 1 --cpus-per-task=6 -x vol08
    ssh vol0X

``-x vol08`` 意思是不使用 vol08，因为 vol01-07 的 cuda 10 才是我们需要的

``ssh vol0X`` 登陆分配的 DGX-2 节点，注意用屏幕上显示的 vol 具体数字替换 ``0X`` 

3. 创建 conda 环境
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    module purge
    module load miniconda3
    module load cuda

    conda create -y -n af10 python=3.8

    source activate af10

4. 安装依赖软件
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    conda install -y cudatoolkit=10.1 cudnn==7.6.4

    conda install -y -c conda-forge openmm==7.5.1 pdbfixer
    conda install -y -c bioconda hmmer hhsuite kalign2

    pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow==2.3.0

    pip install tensorflow-gpu==2.3

    pip install --upgrade jax jaxlib==0.1.69+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html

注意，

* conda install 系列全部完成后再使用 pip install，避免在 pip install 后再使用 conda install；
  
* 各软件版本敏感，如 TensorFlow 不可用 2.5、jaxlib 必须用 0.1.69。请尽量按上方推荐安装；

* 检测是否安装成功（若 GPU 设备均找到，表明安装成功，否则无法正常使用 AlphaFold）：

.. code:: bash

    python
    >>> import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))
    >>> import jax; print(jax.devices())

5. 打一个补丁
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    cd ~/.conda/envs/af10/lib/python3.8/site-packages/
    patch -p0 < $ALPHAFOLD/alphafold/docker/openmm.patch 

至此，conda 安装结束。

conda 使用
^^^^^^^^^^^^^^^^^^^^^^^^

推荐在 ``$ALPHAFOLD`` 主文件夹下新建 ``input`` ``output`` ``task_file`` 三个文件夹。

.. code:: bash

    mkdir input output task_file

然后将 fasta 文件放至 ``input`` 文件夹。

新建一个 slurm 作业脚本，内容如下，命名为 ``sub.slurm``：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=alpha
    #SBATCH --partition=dgx2
    #SBATCH -x vol08
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1
    #SBATCH --output=task_file/%j_%x.out
    #SBATCH --error=task_file/%j_%x.err

    module purge
    module load miniconda3
    source activate af10

    ./run_alphafold.sh -d /home/share/AlphaFold/data \
    -o output -m model_1,model_2 \
    -t 2021-09-12 \
    -f input/test.fasta

然后使用 ``sbatch sub.slurm`` 语句提交作业。

版本三：ColabFold
----------------------------------------

ColabFold 为 Sergey Ovchinnikov 等人开发的适用于 Google Colab 的 AlphaFold 版本，使用 MMseqs2 替代 Jackhmmer，且不使用模版。ColaFold 计算迅速，短序列五六分钟即可算完。

ColabFold 安装步骤
~~~~~~~~~~~~~~~~~~~~~~~~

* ColabFold 使用与 AlphaFold 相同的 conda 环境，所以需要先按照上方“版本二：conda”的方法安装好 ``af10`` 环境；

* 在 ``af10`` 环境里再安装下方四个软件：

.. code:: bash

    pip install jupyter matplotlib py3Dmol tqdm

* 将所需的 ColabFold 文件夹从集群 ``scratch`` 复制到本地：

.. code:: bash

    cp -r /scratch/share/AlphaFold/colabfold $PWD

ColabFold 使用方法
~~~~~~~~~~~~~~~~~~~~~~~~

修改 ``runner.py`` 第 153 行的 fasta 序列，然后使用 ``sbatch sub.slurm`` 语句提交作业。

    
版本四：ParallelFold
----------------------------------------

ParallelFold 为我们开发的适用于大规模计算的集群版，支持 CPU 计算与 GPU 计算分离。

ParallelFold 优点是，对于成百上千个蛋白的批量计算，可以先在 cpu 或 small 节点上批量计算完成前面的 MSA 多序列比对，然后再将各蛋白所得的 feature.pkl 文件，交由 GPU 节点计算。这样既节省了 GPU 资源，又能加快计算速度。

我们的网站：`https://parafold.sjtu.edu.cn <https://parafold.sjtu.edu.cn/>`__

GitHub：`https://github.com/Zuricho/ParallelFold <https://github.com/Zuricho/ParallelFold>`_


ParallelFold 安装步骤
~~~~~~~~~~~~~~~~~~~~~~~~

* ParallelFold 使用与 AlphaFold 相同的 conda 环境和 AlphaFold 文件，所以需要先按照上方“版本二：conda”的方法安装好 ``af10`` 环境；

* 从 `ParallelFold GitHub <https://github.com/Zuricho/ParallelFold>`__ 里下载四个文件：run_alphafold.py run_alphafold.sh run_feature.py run_feature.sh，并将 sh 文件更改权限：

.. code:: bash

    chmod +x run_feature.sh
    chmod +x run_alphafold.sh

ParallelFold  使用方法
~~~~~~~~~~~~~~~~~~~~~~~~

* 若进行完整计算，与正常的 AlphaFold 计算无异：

.. code:: bash

    ./run_alphafold.sh -d /home/share/AlphaFold/data -o output -m model_1,model_2,model_3,model_4,model_5 -f input/test.fasta -t 2021-07-27

* 若只计算 CPU 部分，即批量在集群的 cpu 或 small 节点上计算 MSA：

.. code:: bash

    ./run_feature.sh -d /home/share/AlphaFold/data -o output -m model_1 -f input/test3.fasta -t 2021-07-27  
   


欢迎邮件联系我们，反馈使用情况，或提出宝贵建议。




参考资料
----------------

- AlphaFold GitHub: https://github.com/deepmind/alphafold
- AlphaFold 主页: https://deepmind.com/research/case-studies/alphafold
- AlphaFold Nature 论文: https://www.nature.com/articles/s41586-021-03819-2
- ColabFold GitHub: https://github.com/sokrypton/ColabFold
