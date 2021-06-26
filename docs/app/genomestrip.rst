.. _GenomeSTRiP:

GenomeStrip
==================


简介
----------------
Genome STRiP (Genome STRucture In Populations) is a suite of tools for discovery and
genotyping of structural variation using whole-genome sequencing data. The methods
used in Genome STRiP are designed to find shared variation using data from multiple
individuals. Genome STRiP looks both across and within a set of sequenced genomes to
detect variation.


完整步骤
----------------
.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda genomestrip
