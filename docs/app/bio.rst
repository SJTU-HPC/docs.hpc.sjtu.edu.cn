.. _appbio:

生物信息软件
============

生物信息类软件可通过conda,pip等方法在Pi上安装。

安装之前，须先申请计算节点资源（登陆节点禁止大规模编译安装）

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash
   
+-----------------+-------------+-------------+---------------+
| MELT            | Manta       | Lumpy       | Hydra-sv      |
+-----------------+-------------+-------------+---------------+
| VariationHunter | GRIDSS      | GenomeSTRiP | FermiKit      |
+-----------------+-------------+-------------+---------------+
| ERDS            | DELLY       | CREST       | Control-FREEC |
+-----------------+-------------+-------------+---------------+
| CNVnator        | CLEVER      | BreakDancer |BICseq2        |
+-----------------+-------------+-------------+---------------+
| BatVI           | BASIL-ANISE | MetaSV      |MindTheGap     |
+-----------------+-------------+-------------+---------------+
| Mobster         | pbsv        | Pindel      |PRISM          |
+-----------------+-------------+-------------+---------------+


MELT安装
--------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda melt

Manta安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda manta

Lumpy安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda lumpy-sv

Hydra-sv安装
------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c conda-forge hydra

VariationHunter安装
-------------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda tardis

GRIDSS安装
----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda gridss

GenomeSTRiP安装
---------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda genomestrip

FermiKit安装
------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda fermikit

ERDS安装
--------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda erds

DELLY安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda delly

CREST安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda blat
   conda install -c bioconda cap3
   conda install -c bioconda samtools
   conda install -c bioconda perl-bioperl
   conda install -c bioconda perl-bio-db-sam
   conda install -c imperial-college-research-computing crest

Control-FREEC安装
-----------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda control-freec

CNVnator安装
------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda -c conda-forge cnvnator

CLEVER安装
----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda clever-toolkit

BreakDancer安装
---------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda breakdancer

BICseq2安装
-----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda bicseq2-norm

BatVI安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda batvi

BASIL-ANISE安装
---------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda anise_basil

MetaSV安装
----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda metasv

MindTheGap安装
--------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda mindthegap

Mobster安装
-----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda mobster

pbsv安装
--------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda pbsv

Pindel安装
----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda pindel

PRISM安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c conda-forge pyprism



RetroSeq安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c hcc retroseq



Sniffles安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda sniffles



SV2安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda sv2



SvABA安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda svaba



SVDetect安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c imperial-college-research-computing svdetect



Wham安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda wham



gsutil安装
---------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c conda-forge gsutil



openslide-python安装
---------------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda openslide-python
   conda install libiconv

pandas安装
-----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c anaconda pandas

cdsapi安装
-----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c conda-forge cdsapi

STRique安装
------------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n mypy
   source activate mypy
   git clone --recursive https://github.com/giesselmann/STRique
   cd STRique
   pip install -r requirements.txt
   python setup.py install 

r-rgl安装
----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c r r-rgl

sra-tools安装
--------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda sra-tools

DESeq2安装
-----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda bioconductor-deseq2

安装完成后可以在 R 中输入 ``library("DESeq2")`` 检测是否安装成功

WGCNA安装
----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda r-wgcna

MAKER安装
----------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda maker

AUGUSTUS安装
-------------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c anaconda boost
   conda install -c bioconda augustus

DeepGo安装
-----------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   git clone https://github.com/bio-ontology-research-group/deepgo.git
   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install pip
   pip install -r requirements.txt

km安装
-------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   git clone https://github.com/iric-soft/km.git
   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   chmod +x easy_install.sh 
   ./easy_install.sh

Requests安装
-------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c anaconda requests

