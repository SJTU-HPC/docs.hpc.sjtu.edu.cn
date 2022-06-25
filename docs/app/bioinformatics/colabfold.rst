ColabFold
=============

ColabFold 是 Sergey Ovchinnikov 等人开发的快速蛋白结构预测软件，使用 MMseqs2 替代 MSA，能够快速精准预测包含复合体在内的蛋白结构。开源代码：`https://github.com/sokrypton/ColabFold <https://github.com/sokrypton/ColabFold>`__

ColabFold 支持本地安装使用，Yoshitaka Moriwaki 开发维护的 `LocalColabFold <https://github.com/YoshitakaMo/localcolabfold>`__ 可以很容易在交大思源一号上安装。下面将介绍以 LocalColabFold 形式在思源一号上安装和使用 ColabFold

交大计算平台同时也部署了 AlphaFold 和 ParaFold，欢迎查看：:doc:`alphafold2` 

ColabFold 安装
----------------------------------------

申请 CPU 计算节点，以交互模式安装。conda 安装都需要至计算节点：

.. code:: bash

    srun -p 64c512g -n 8 --pty /bin/bash

假设在个人 home 文件夹下建立 colab 文件夹：

.. code:: bash

    mkdir ~/colab; cd colab

下载 localcolabfold：

.. code:: bash

    git clone https://github.com/YoshitakaMo/localcolabfold.git

接下来一键安装全部软件 (这里预计半小时以上)：

.. code:: bash

    cd localcolabfold
    ./install_colabbatch_linux.sh

安装完毕，将出现 Installation of colabfold_batch finished 字样


ColabFold 使用
----------------------------------------

ColabFold 在思源一号上有两种运行方法：

* 交互模式，适用于短序列和调试
  
* slurm 作业模式，适用于长时间或正式计算

方法一：交互模式运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

申请 GPU 计算节点：

.. code:: bash

    srun -p a100 -N 1 -n 1 --cpus-per-task=16 --gres=gpu:1 --pty /bin/bash

激活 conda 环境：

.. code:: bash

    export PATH="~/colab/localcolabfold/colabfold_batch/bin:$PATH"
    module load miniconda3
    source activate ~/colab/localcolabfold/colabfold_batch/colabfold-conda

在包含 ``test.fasta`` 的文件夹里运行：

.. code:: bash

    colabfold_batch --num-recycle 1 test.fasta output

其中，``test.fasta`` 文件内容示例：

.. code:: bash

    >2LHC_1|Chain A|Ga98|artificial gene (32630)
    PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK

方法二：slurm 脚本运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本示例（假设作业脚本名为 ``sub.slurm``）：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=colabfold
    #SBATCH --partition=a100
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=16
    #SBATCH --gres=gpu:1          # use 1 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    export PATH="~/colab/localcolabfold/colabfold_batch/bin:$PATH"
    export https_proxy=http://proxy2.pi.sjtu.edu.cn:3128
    export http_proxy=http://proxy2.pi.sjtu.edu.cn:3128
    export no_proxy=puppet,proxy,172.16.0.133,pi.sjtu.edu.cn

    module load miniconda3
    source activate ~/colab/localcolabfold/colabfold_batch/colabfold-conda

    colabfold_batch --num-recycle 1 test.fasta output

然后使用 ``sbatch sub.slurm`` 语句提交作业



参考资料
----------------

- AlphaFold GitHub: https://github.com/deepmind/alphafold
- ColabFold GitHub: https://github.com/sokrypton/ColabFold
- LocalColabFold GitHub: https://github.com/YoshitakaMo/localcolabfold
- ParaFold 网站：https://parafold.sjtu.edu.cn