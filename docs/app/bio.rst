.. _appbio:

生物信息软件
============

生物信息类软件可通过conda,pip等方法在Pi上安装。

安装之前，须先申请计算节点资源（登陆节点禁止大规模编译安装）

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash

.. raw:: html

   <table>

.. raw:: html

   <tr>

.. raw:: html

   <td>

MELT

.. raw:: html

   </td>

.. raw:: html

   <td>

Manta

.. raw:: html

   </td>

.. raw:: html

   <td>

Lumpy

.. raw:: html

   </td>

.. raw:: html

   <td>

Hydra-sv

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

VariationHunter

.. raw:: html

   </td>

.. raw:: html

   <td>

GRIDSS

.. raw:: html

   </td>

.. raw:: html

   <td>

GenomeSTRiP

.. raw:: html

   </td>

.. raw:: html

   <td>

FermiKit

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

ERDS

.. raw:: html

   </td>

.. raw:: html

   <td>

DELLY

.. raw:: html

   </td>

.. raw:: html

   <td>

CREST

.. raw:: html

   </td>

.. raw:: html

   <td>

Control-FREEC

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

CNVnator

.. raw:: html

   </td>

.. raw:: html

   <td>

CLEVER

.. raw:: html

   </td>

.. raw:: html

   <td>

BreakDancer

.. raw:: html

   </td>

.. raw:: html

   <td>

BICseq2

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

BatVI

.. raw:: html

   </td>

.. raw:: html

   <td>

BASIL-ANISE

.. raw:: html

   </td>

.. raw:: html

   <td>

MetaSV

.. raw:: html

   </td>

.. raw:: html

   <td>

MindTheGap

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

Mobster

.. raw:: html

   </td>

.. raw:: html

   <td>

pbsv

.. raw:: html

   </td>

.. raw:: html

   <td>

Pindel

.. raw:: html

   </td>

.. raw:: html

   <td>

PRISM

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

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
