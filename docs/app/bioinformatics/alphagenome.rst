AlphaGenome
===========

AlphaGenome 是 Google DeepMind 发布的基因组基础模型，可对 DNA 序列进行调控功能预测、变异效应预测和多组学信号预测等分析。AlphaGenome 使用 JAX 运行，建议在 NVIDIA GPU 节点上执行推理任务。

本文档介绍如何在思源一号 GPU 节点上使用 miniconda3 配置 AlphaGenome 环境，并使用公共模型权重进行本地推理。

可用资源
--------

+--------------+--------------+-------------------------+
| 硬件         | 平台         | 说明                    |
+==============+==============+=========================+
| A100-40GB    | 思源一号     | 可用于小规模本地推理    |
+--------------+--------------+-------------------------+

模型权重位于公共路径：

.. code-block:: bash

   /dssg/share/data/alphagenome

.. note::

   AlphaGenome 官方推荐使用 H100 GPU。A100-40GB 可用于较短序列或较少输出类型的推理测试；较长序列、多个输出类型或批量任务可能需要更多显存和更长的首次 JIT 编译时间。

配置 Conda 环境
---------------

AlphaGenome 的 GPU 依赖需要在 GPU 环境中安装。请不要在登录节点直接安装 GPU 版 JAX，建议先使用 debuga100 队列申请交互式 GPU 作业完成环境配置。debuga100 的具体限制和用法可参考 :doc:`../../job/dgx`。

在思源一号登录节点上申请 debuga100 交互式作业：

.. code-block:: bash

   srun -p debuga100 --qos=debug -N 1 -n 1 --gres=gpu:1 --cpus-per-task=4 --pty /bin/bash

.. note::

   debuga100 是调试队列，单卡显存较小，适合安装环境和检查 GPU 依赖；完整 AlphaGenome 推理建议提交到 a100 队列。

进入 GPU 计算节点后，加载 miniconda3 并创建独立环境：

.. code-block:: bash

   module load miniconda3
   conda create -n alphagenome-env python=3.11 -y
   conda activate alphagenome-env
   python -m pip install --upgrade pip

安装 AlphaGenome 及 JAX GPU 依赖。若可访问 Python 包源，可直接安装：

.. code-block:: bash

   pip install alphagenome
   pip install "jax[cuda12]"

如需安装 AlphaGenome research 代码，可从源码目录安装：

.. code-block:: bash

   git clone https://github.com/google-deepmind/alphagenome.git alphagenome_research
   cd alphagenome_research
   pip install -e .

安装完成后可检查 GPU 后端是否可用：

.. code-block:: bash

   python -c 'import jax; print("JAX backend:", jax.default_backend()); print("JAX devices:", jax.devices())'

若输出中包含 gpu 和 CudaDevice，表示 JAX 已识别 GPU。

本地权重推理示例
----------------

以下示例直接输入 2048 bp DNA 序列，使用公共路径中的模型权重进行 DNASE 输出预测。该方式不依赖参考基因组 FASTA 文件，适合作为环境和 GPU 推理测试。

创建 alphagenome_predict_sequence.py：

.. code-block:: python

   import pathlib
   import time

   import jax
   from alphagenome_research.model import dna_model

   checkpoint = pathlib.Path("/dssg/share/data/alphagenome")

   print("checkpoint:", checkpoint)
   print("JAX backend:", jax.default_backend())
   print("JAX devices:", jax.devices())

   t0 = time.time()
   model = dna_model.create(checkpoint, device=jax.devices()[0])
   print("loaded local checkpoint in %.2fs" % (time.time() - t0))

   sequence = "ACGT" * 512
   t0 = time.time()
   output = model.predict_sequence(
       sequence,
       requested_outputs=[dna_model.OutputType.DNASE],
       ontology_terms=["UBERON:0001157"],
   )
   print("predict_sequence done in %.2fs" % (time.time() - t0))
   print("DNASE shape:", output.dnase.values.shape)
   print("DNASE dtype:", output.dnase.values.dtype)
   print("DNASE first value:", float(output.dnase.values.reshape(-1)[0]))

交互式测试
----------

环境安装和 GPU 后端检查可使用 debuga100 调试队列：

.. code-block:: bash

   srun -p debuga100 --qos=debug -N 1 -n 1 --gres=gpu:1 --cpus-per-task=4 --pty /bin/bash

完整模型推理建议使用思源一号 a100 队列。申请 A100 GPU 交互式作业：

.. code-block:: bash

   srun -p a100 -N 1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --pty /bin/bash

进入计算节点后运行：

.. code-block:: bash

   module load miniconda3
   conda activate alphagenome-env
   python alphagenome_predict_sequence.py

正常情况下可看到类似输出：

.. code-block:: text

   JAX backend: gpu
   JAX devices: [CudaDevice(id=0)]
   loaded local checkpoint in ...s
   predict_sequence done in ...s
   DNASE shape: (2048, 1)

Slurm 作业脚本
--------------

创建 alphagenome.slurm：

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=alphagenome
   #SBATCH --partition=a100
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=8
   #SBATCH --gres=gpu:1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load miniconda3
   conda activate alphagenome-env

   export TF_CPP_MIN_LOG_LEVEL=2
   export XLA_PYTHON_CLIENT_PREALLOCATE=false

   python alphagenome_predict_sequence.py

提交作业：

.. code-block:: bash

   sbatch alphagenome.slurm

使用 squeue 查看作业状态：

.. code-block:: bash

   squeue -u $USER

按基因组坐标预测
----------------

predict_interval 和 predict_variant 会按坐标从参考基因组中提取序列，因此除模型权重外，还需要为相应物种配置参考基因组 FASTA、索引文件和必要的注释文件。若仅验证 AlphaGenome 是否能在 GPU 上运行，推荐优先使用上文的 predict_sequence 示例。

常见问题
--------

1. JAX backend 显示为 cpu
^^^^^^^^^^^^^^^^^^^^^^^^^

**A:** 请确认作业运行在 GPU 计算节点上，并检查是否申请了 --gres=gpu:1。同时确认安装的是支持 CUDA 的 JAX 版本。

2. 首次推理耗时较长
^^^^^^^^^^^^^^^^^^^

**A:** AlphaGenome 使用 JAX JIT 编译，首次运行会包含编译时间。相同输入形状和输出类型的后续推理通常更快。

3. 运行时显存不足
^^^^^^^^^^^^^^^^^

**A:** 可减少序列长度、减少 requested_outputs 中的输出类型，或只预测必要的 ontology terms。也可设置 XLA_PYTHON_CLIENT_PREALLOCATE=false 避免 JAX 启动时预分配大部分显存。

4. 使用 predict_interval 报错找不到 FASTA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A:** predict_interval 需要参考基因组 FASTA extractor。若没有配置参考基因组文件，请先使用 predict_sequence 直接输入 DNA 序列。

参考资料
--------

- AlphaGenome GitHub: https://github.com/google-deepmind/alphagenome
- JAX NVIDIA GPU 安装说明: https://docs.jax.dev/en/latest/installation.html#nvidia-gpu
