AlphaFold3
===============

AlphaFold3 是由谷歌 DeepMind 和 Isomorphic Labs 团队开发的人工智能程序，于 2024 年 5 月发布，11 月开源供学术用途。它能准确预测蛋白质、DNA、RNA 等多种生物分子及其复合体的结构和相互作用，预测精度较 AlphaFold2 大幅提升，在药物研发等领域应用广泛。其架构引入 Pairformer 和扩散模块，采用跨蒸馏技术训练，减少了多重序列比对处理量，提高了计算效率和泛化能力，为生命科学研究提供了更强大的工具。

可用的版本
----------

+--------------+--------------+
| A100-40GB    | 思源一号     |
+--------------+--------------+
| A800-80GB    | 思源一号     |
+--------------+--------------+
| V100-32GB    | Pi2.0(待更新)|
+--------------+--------------+

版本区别
---------------------
1. 思源一号的A800-80GB镜像为原镜像，未做修改，AlphaFold3可以在单张A800-80GB上运行，最大能够处理5120个tokens。

2. 对于A100-40GB，根据官方文档，我们对原镜像进行了以下修改：

- 开启unified memory,启用统一内存允许程序在空间不足时将 GPU 内存溢出到主机内存,这可以防止OOM,修改/app/alphafold/run_alphafold.py文件如下:

.. code-block:: python

  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
  os.environ['XLA_CLIENT_MEM_FRACTION']='3.2'
  os.environ['TF_FORCE_UNIFIED_MEMORY']='true'


- 调整/app/alphafold/run_alphafold.py/src/alphafold3/model/model_config.py中pair_transition_shard_spec参数如下：

.. code-block:: python

  pair_transition_shard_spec: Sequence[_Shape2DType] = (
      (2048, None),
      (3072, 1024),
      (None, 512),
  )

3. A100-40GB上最大能够处理4352个tokens的输入，与A800-80GB相比，计算精度一致，但吞吐量更小。

使用前准备
----------

- 新建运行文件夹，如 ``alphafold``
- 在运行文件夹中创建输入文件夹 ``input`` 和输出文件夹 ``output``
- 在 ``input`` 文件夹中放置输入 JSON 文件，例如 fold_input.json，自定义输入 JSON 文件可参考官方文档 `https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md <https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md>`_ 

单体蛋白质结构预测 JSON 文件示例
--------------------------------

.. code-block:: json

    {
      "name": "2PV7",
      "sequences": [
        {
          "protein": {
            "id": ["A", "B"],
            "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
          }
        }
      ],
      "modelSeeds": [1],
      "dialect": "alphafold3",
      "version": 1
    }

在思源一号上运行 AlphaFold3
---------------------------------

A100-40GB
###########################

.. code:: bash

  #!/bin/bash
  #SBATCH --job-name=alphafold3
  #SBATCH --partition=a100
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --cpus-per-task=16
  #SBATCH --gres=gpu:1          # use 1 GPU
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err


  singularity exec \
      --nv \
      --bind $PWD/input:/root/af_input \
      --bind $PWD/output:/root/af_output \
      --bind /dssg/share/data/alphafold3/models:/root/models \
      --bind /dssg/share/data/alphafold3/database:/root/public_databases \
      /dssg/share/imgs/ai/alphafold/alphafold3-a100.sif \  
      /alphafold3_venv/bin/python /app/alphafold/run_alphafold.py \
      --json_path=/root/af_input/fold_input.json \  #fold_input.json为输入文件名称
      --model_dir=/root/models \
      --db_dir=/root/public_databases \
      --output_dir=/root/af_output

A800-80GB
###########################

.. code:: bash

  #!/bin/bash
  #SBATCH --job-name=alphafold3
  #SBATCH --partition=a800
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --cpus-per-task=16
  #SBATCH --gres=gpu:1          # use 1 GPU
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err


  singularity exec \
      --nv \
      --bind $PWD/input:/root/af_input \
      --bind $PWD/output:/root/af_output \
      --bind /dssg/share/data/alphafold3/models:/root/models \
      --bind /dssg/share/data/alphafold3/database:/root/public_databases \
      /dssg/share/imgs/ai/alphafold/alphafold3-a800.sif \
      /alphafold3_venv/bin/python /app/alphafold/run_alphafold.py \
      --json_path=/root/af_input/fold_input.json \   #fold_input.json为输入文件名称
      --model_dir=/root/models \
      --db_dir=/root/public_databases \
      --output_dir=/root/af_output

使用 ``sbatch sub.slurm`` 语句提交作业

运行结束后，计算结果保存在 alphafold/output/ 下，具体可参考 `https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md <https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>`_ 

参考资料
----------------
- AlphaFold3 GitHub: https://github.com/google-deepmind/alphafold3