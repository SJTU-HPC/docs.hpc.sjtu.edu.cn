.. _metawrap:


MetaWRAP
===========

简介
----

MetaWRAP这是一套强大的宏基因组分析流程，专注于宏基因组Binning，能实现原始序列的质控、物种注释和可视化、宏基因组拼接、三种主流Bin方法分析和结果筛选与可视化、Bin的重新组装、Bin的物种和功能注释等。轻松实现Bin相关分析和可视化的绝大部分需求。

可通过conda自行安装。

安装代码
-----------

.. code-block:: bash

    srun -p 64c512g -n 4 --pty /bin/bash
    mkdir ${HOME}/01.application/11.metaWRAP && cd ${HOME}/01.application/11.metaWRAP
    git clone https://github.com/bxlab/metaWRAP.git
    mkdir database
    cd metaWRAP
    sed -i 's#/scratch/gu#${HOME}/01.application/11.metaWRAP/database#g' bin/config-metawrap
    export PATH=${HOME}/01.application/11.metaWRAP/metaWRAP/bin/:$PATH
    module load miniconda3
    conda create -n env4mamba mamba -y
    source activate env4mamba
    mamba create -n env4metawrap -c conda-forge python=2.7 -y
    source activate env4metawrap
    conda config --add channels defaults
    conda config --add channels conda-forge
    conda config --add channels bioconda
    conda config --add channels ursky
    ${HOME}/.conda/envs/env4mamba/bin/mamba install --only-deps -c ursky metawrap-mg -y

   
参考资料
--------

-  `MetaWRAP <https://github.com/bxlab/metaWRAP>`__
