AlphaFold3
===============

AlphaFold3 是由谷歌 DeepMind 和 Isomorphic Labs 团队开发的人工智能程序，于 2024 年 5 月发布，11 月开源供学术用途。它能准确预测蛋白质、DNA、RNA 等多种生物分子及其复合体的结构和相互作用，预测精度较 AlphaFold2 大幅提升，在药物研发等领域应用广泛。其架构引入 Pairformer 和扩散模块，采用跨蒸馏技术训练，减少了多重序列比对处理量，提高了计算效率和泛化能力，为生命科学研究提供了更强大的工具。

使用前准备
----------

- 新建运行文件夹，如 ``alphafold``。
- 在运行文件夹中创建输入文件夹 ``input`` 和输出文件夹 ``output``。
- 在 ``input`` 文件夹中放置输入 JSON 文件，例如 fold_input.json，自定义输入 JSON 文件可参考官方文档 `https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md <https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md>`_ 。

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

可用的版本
----------

+--------------+--------------+
| A100-40G     | 思源一号     |
+--------------+--------------+
| A800-80GB    | 思源一号     |
+--------------+--------------+
| V100-32GB    | Pi2.0(待更新)|
+--------------+--------------+

在思源一号上运行 AlphaFold3
---------------------------------

1. 从公共路径拷贝作业脚本到运行文件夹

   A100-40GB 版本：

   ``cp /dssg/share/imgs/ai/alphafold/af3-a100.slurm ./alphafold/``

   A800-80GB 版本：

   ``cp /dssg/share/imgs/ai/alphafold/af3-a800.slurm ./alphafold/``

2. 在运行文件夹中提交 AlphaFold3 计算作业

   A100-40GB 版本：

   ``sbatch af3-a100.slurm``

   A800-80GB 版本：

   ``sbatch af3-a800.slurm``

3. 运行结束后，计算结果保存在 alphafold/output/ 下，具体可参考 `https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md <https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>`_ 。

参考资料
----------------
- AlphaFold3 GitHub https://github.com/google-deepmind/alphafold3/tree/main