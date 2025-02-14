AlphaFold3
===============

AlphaFold3 是由谷歌 DeepMind 和 Isomorphic Labs 团队开发的人工智能程序，于 2024 年 5 月发布，11 月开源供学术用途。它能准确预测蛋白质、DNA、RNA 等多种生物分子及其复合体的结构和相互作用，预测精度较 AlphaFold2 大幅提升，在药物研发等领域应用广泛。其架构引入 Pairformer 和扩散模块，采用跨蒸馏技术训练，减少了多重序列比对处理量，提高了计算效率和泛化能力，为生命科学研究提供了更强大的工具。

可用的版本
----------

+--------------+--------------+---------+
| 硬件         | 平台         | 版本    |
+--------------+--------------+---------+
| A100-40GB    | 思源一号     | v3.0.1  |
+--------------+--------------+---------+
| A800-80GB    | 思源一号     | v3.0.1  |
+--------------+--------------+---------+
| V100-32GB    | Pi2.0(待更新)| v3.0.0  |
+--------------+--------------+---------+
| K100_AI-64GB | Pi2.0        | v3.0.0  |
+--------------+--------------+---------+


版本区别
---------------------
1. 思源一号的A800-80GB镜像为原镜像，未做修改，AlphaFold3可以在单张A800-80GB上运行，最大能够处理5120个tokens。

2. 对于A100-40GB，根据官方文档，我们对原镜像进行了以下修改：

- 开启unified memory,启用统一内存允许程序在空间不足时将 GPU 内存溢出到主机内存,这可以防止OOM,修改 `/app/alphafold/run_alphafold.py` 文件如下:

.. code-block:: python

  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
  os.environ['XLA_CLIENT_MEM_FRACTION']='3.2'
  os.environ['TF_FORCE_UNIFIED_MEMORY']='true'


- 调整 `/app/alphafold/src/alphafold3/model/model_config.py` 中 `pair_transition_shard_spec` 参数如下：

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

蛋白质结构预测 JSON 文件示例
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

作业脚本 `af3.slurm` 示例如下：

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
      --json_path=/root/af_input/fold_input.json \
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
      --json_path=/root/af_input/fold_input.json \
      --model_dir=/root/models \
      --db_dir=/root/public_databases \
      --output_dir=/root/af_output

使用 `sbatch af3.slurm` 语句提交作业。

运行结束后，计算结果保存在 alphafold/output/ 下，具体可参考 `https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md <https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>`_ 

分阶段运行AlphaFold3
###########################
AlphaFold3运行分为 `data_pipeline` 和 `inference` 两个阶段， `data_pipeline` 阶段主要利用CPU进行MSA/模板搜索， `inference` 阶段主要利用GPU进行模型推理。

1. 将 `data_pipeline` 与 `inference` 分开，减少GPU机时浪费。
2. 输出结果的json中会保存缓存 MSA / 模板搜索的结果，可以用于跨种子或其他特征变体（例如配体）的多个不同推理。

以A100为例：

- data_pipeline阶段脚本示例（64c512g队列运行）：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=alphafold3
    #SBATCH --partition=64c512g
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=32
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err


    /usr/bin/time -v singularity exec \
        --nv \
        --bind $PWD/input:/root/af_input \
        --bind $PWD/output:/root/af_output \
        --bind /dssg/share/data/alphafold3/models:/root/models \
        --bind /dssg/share/data/alphafold3/database:/root/public_databases \
        /dssg/share/imgs/ai/alphafold/alphafold3-a100.sif \
        /alphafold3_venv/bin/python /app/alphafold/run_alphafold.py \
        --norun_inference \
        --jackhmmer_n_cpu=$SLURM_NTASKS \
        --nhmmer_n_cpu=$SLURM_NTASKS \
        --input_dir=/root/af_input \
        --model_dir=/root/models \
        --db_dir=/root/public_databases \
        --output_dir=/root/af_output


`alphafold/output/` 文件夹下会保存 `data_pipeline` 阶段缓存的结果，以该目录下的json文件作为 `inference` 阶段的输入。本示例中生成的结果文件为 `2PV7_data.json`。

- inference阶段脚本示例（a100队列运行）：

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
        /dssg/share/imgs/ai/alphafold/alphafold3-a800.sif \
        /alphafold3_venv/bin/python /app/alphafold/run_alphafold.py \
        --norun_data_pipeline \
        --json_path=/root/af_input/2PV7_data.json \
        --model_dir=/root/models \
        --db_dir=/root/public_databases \
        --output_dir=/root/af_output


在国产DCU平台上运行AlphaFold3
--------------------------------
平台基于K100_AI计算卡，提供k100队列用于测试，硬件参数如下：

- CPU：2 × Hygon C86 7490 (2.2GHz, 64 cores)
- DCU：8 × K100_AI 64GB
- 架构：x86
- 系统：Rocky Linux 9.4

运行AlphaFold3示例脚本如下：

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=k100test
    #SBATCH --partition=k100
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=16
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    unset APPTAINER_BIND
    export TF_CPP_MIN_LOG_LEVEL=2
    export PYTHONPATH=/usr/local:${PYTHONPATH}
    free_dcu=$(/opt/hyhal/set_k100.sh)
    export ROCR_VISIBLE_DEVICES=$free_dcu
    echo "已设置ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"

    INPUT_DIR=./input
    MODEL_PATH=/home/aftest/alphafold3/models
    OUTPUT_PATH=./output
    DB_DIR=/home/aftest/alphafold3/public_databases
    singularity exec \
            -B /opt/hyhal \
            -B /opt/share \
            -B /home/aftest/alphafold3 \
            /opt/share/alphafold3-deepmind_jax0423-py311-dtk2404-ubuntu2204.sif \
            python /opt/share/alphafold3-code/run_alphafold.py \
                --input_dir=$INPUT_DIR \
                --model_dir=$MODEL_PATH \
                --output_dir=$OUTPUT_PATH \
                --run_data_pipeline=true \
                --db_dir=$DB_DIR \
                --flash_attention_implementation=xla

使用sbatch af3_k100.slurm命令，从π 2.0提交作业。

data_pipeline性能测试
---------------------

思源平台 A100-40GB

+--------------+--------------+
| 版本         |   用时(s)    |
+--------------+--------------+
| v3.0.0       | 2475         |
+--------------+--------------+
| v3.0.1       | 677          |
+--------------+--------------+



参考资料
----------------
- AlphaFold3 GitHub: https://github.com/google-deepmind/alphafold3