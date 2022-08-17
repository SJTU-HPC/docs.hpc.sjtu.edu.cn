.. _sra-tools:

sra-tools
===========================

简介
----------------

来自 NCBI 的 SRA 工具包和 SDK 是用于使用 INSDC 序列读取档案中的数据的工具和库的集合。

完整步骤
-----------------

.. code:: bash
   
   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda sra-tools

源码安装步骤
----------------

.. code:: bash

    srun -p small -n 4 --pty /bin/bash
    mkdir -p ${HOME}/01.application/13.sra-tools && cd ${HOME}/01.application/13.sra-tools
    wget --output-document sratoolkit.tar.gz https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-centos_linux64.tar.gz
    tar -vxzf sratoolkit.tar.gz
    export PATH=${HOME}/01.application/13.sra-tools/sratoolkit.3.0.0-centos_linux64/bin:$PATH
    which fasterq-dump


参考资料
------------

-  `sra-tools <https://github.com/ncbi/sra-tools/wiki/>`__
