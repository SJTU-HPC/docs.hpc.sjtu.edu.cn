.. _MAKER:

MAKER
============================

简介
-------------

MAKER is a portable and easily configurable genome annotation pipeline. Its purpose is
to allow smaller eukaryotic and prokaryotic genome projects to independently annotate
their genomes and to create genome databases. MAKER identifies repeats, aligns ESTs and
proteins to a genome, produces ab-initio gene predictions and automatically synthesizes
these data into gene annotations having evidence-based quality values. MAKER is also
easily trainable: outputs of preliminary runs can be used to automatically retrain its
gene prediction algorithm, producing higher quality gene-models on seusequent runs.
MAKER's inputs are minimal and its ouputs can be directly loaded into a GMOD database.
They can also be viewed in the Apollo genome browser; this feature of MAKER provides an
easy means to annotate, view and edit individual contigs and BACs without the overhead of
a database. MAKER should prove especially useful for emerging model organism projects with
minimal bioinformatics expertise and computer resources.

完整步骤
--------------

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda maker
