.. _RFdiffusion:

RFdiffusion
=======================

简介
---------------

RFdiffusion is an open source method for structure generation, with or without conditional information (a motif, target etc). It can perform a whole range of protein design challenges as we have outlined in the RFdiffusion paper.

Things Diffusion can do:

- Motif Scaffolding
- Unconditional protein generation
- Symmetric unconditional generation (cyclic, dihedral and tetrahedral symmetries currently implemented, more coming!)
- Symmetric motif scaffolding
- Binder design
- Design diversification ("partial diffusion", sampling around a design)

安装步骤
----------------

1.克隆github仓库

.. code:: bash

  git clone https://github.com/RosettaCommons/RFdiffusion.git

2.下载模型权重文件

.. code:: bash

  cd RFdiffusion
  mkdir models && cd models
  wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
  wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
  wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
  wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
  wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
  wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
  wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt

  Optional:
  wget http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt

  # original structure prediction weights
  wget http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt1

3.在conda中安装软件

.. code:: bash

  srun -n 1 -p a100 --gres=gpu:1 --pty /bin/bash
  module load miniconda3
  module load gcc

  conda env create -f env/SE3nv.yml

  conda activate SE3nv
  cd env/SE3Transformer
  pip install --no-cache-dir -r requirements.txt
  python setup.py install
  cd ../.. # change into the root directory of the repository
  pip install -e . # install the rfdiffusion module from the root of the repository

软件使用
----------------

测试软件是否能够正常使用

.. code:: bash

  srun -n 1 -p a100 --gres=gpu:1 --pty /bin/bash
  module load miniconda3
  module load gcc

  source activate SE3nv
  ./scripts/run_inference.py 'contigmap.contigs=[150-150]' inference.output_prefix=test_outputs/test inference.num_designs=10

如果能够正常在主屏幕输出且无报错，则软件可用。
更多参数和使用方式参见官网： https://github.com/RosettaCommons/RFdiffusion

