.. _Hydra-sv:

Hydra-sv
====================


简介
---------------

用于使用双端映射检测结构变异 (SV) 断点。与其他算法类似，Hydra 通过对不一致的双端对齐进行聚类来检测 SV 断点，
其“签名”证实了相同的假定断点。Hydra 可以检测由所有类别的结构变异引起的断点。它还旨在检测独特和重复基因组区域
的变异（例如，片段重复和转座子插入中的突变）；因此，它将检查具有多个不一致比对的配对末端读数。

完整步骤
---------------

.. code:: bash

  wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hydra-sv/Hydra.v0.5.3.tar.gz
  wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hydra-sv/bedpeToBed12.py
  wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hydra-sv/hydraFrequency.py
  tar -zxvf Hydra.v0.5.3.tar.gz
  cd Hydra-Version-0.5.3
  make clean 
  make all

使用方法

.. code:: bash

  cd bin
  ./hydra -h

