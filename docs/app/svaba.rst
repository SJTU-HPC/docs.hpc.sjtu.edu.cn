.. _SvABA:

SvABA
=================

简介
------------

SvABA is a method for detecting structural variants in sequencing data using genome-wide local assembly.
Under the hood, SvABA uses a custom implementation of SGA (String Graph Assembler) by Jared Simpson,
and BWA-MEM by Heng Li. Contigs are assembled for every 25kb window (with some small overlap) for every
region in the genome. The default is to use only clipped, discordant, unmapped and indel reads, although
this can be customized to any set of reads at the command line using VariantBam rules. These contigs are
then immediately aligned to the reference with BWA-MEM and parsed to identify variants. Sequencing reads
are then realigned to the contigs with BWA-MEM, and variants are scored by their read support.

完整步骤
----------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda svaba
