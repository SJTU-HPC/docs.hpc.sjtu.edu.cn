Pi上的软件
==========

本文档介绍 Pi 上的软件。商业软件需用户自行获取版权并安装。 |cpu| |gpu|
|arm| 标签表明软件有 cpu, gpu 和 arm 版本。

.. _pi-上的软件-1:

Pi上的软件
----------

+-----------------+-----------------+-----------------+-----------------+
| Name            | Version         | Platform        | Introduction    |
+=================+=================+=================+=================+
| `ABINIT <https: | ![](https://img | |cpu|           | ABINIT is a     |
| //docs.hpc.sjtu | .shields.io/bad |                 | package whose   |
| .edu.cn/applica | ge/version-8.10 |                 | main program    |
| tion/abinit/>`_ | .3-yellowgreen? |                 | allows one to   |
| _               | style=flat-squa |                 | find the total  |
|                 | re)             |                 | energy, charge  |
|                 |                 |                 | density and     |
|                 |                 |                 | electronic      |
|                 |                 |                 | structure of    |
|                 |                 |                 | systems made of |
|                 |                 |                 | electrons and   |
|                 |                 |                 | nuclei          |
|                 |                 |                 | (molecules and  |
|                 |                 |                 | periodic        |
|                 |                 |                 | solids) within  |
|                 |                 |                 | Density         |
|                 |                 |                 | Functional      |
|                 |                 |                 | Theory (DFT),   |
|                 |                 |                 | using           |
|                 |                 |                 | pseudopotential |
|                 |                 |                 | s               |
|                 |                 |                 | and a planewave |
|                 |                 |                 | or wavelet      |
|                 |                 |                 | basis.          |
+-----------------+-----------------+-----------------+-----------------+
| `Amber <https:/ |                 | |cpu| |gpu|     | A package of    |
| /docs.hpc.sjtu. |                 |                 | molecular       |
| edu.cn/applicat |                 |                 | simulation      |
| ion/Amber/>`__  |                 |                 | programs and    |
|                 |                 |                 | analysis tools. |
+-----------------+-----------------+-----------------+-----------------+
| BCFtools        | ![](https://img | |cpu|           | BCFtools is a   |
|                 | .shields.io/bad |                 | program for     |
|                 | ge/version-1.9  |                 | variant calling |
|                 | .3-yellowgreen? |                 | and             |
|                 | style=flat-squa |                 | manipulating    |
|                 | re)             |                 | files in the    |
|                 |                 |                 | Variant Call    |
|                 |                 |                 | Format (VCF)    |
|                 |                 |                 | and its binary  |
|                 |                 |                 | counterpart     |
|                 |                 |                 | BCF. All        |
|                 |                 |                 | commands work   |
|                 |                 |                 | transparently   |
|                 |                 |                 | with both VCFs  |
|                 |                 |                 | and BCFs, both  |
|                 |                 |                 | uncompressed    |
|                 |                 |                 | and             |
|                 |                 |                 | BGZF-compressed |
|                 |                 |                 | .               |
+-----------------+-----------------+-----------------+-----------------+
| Bedtools2       | ![](https://img | |cpu|           | The bedtools    |
|                 | .shields.io/bad |                 | utilities are a |
|                 | ge/version-2.27 |                 | swiss-army      |
|                 | .1-yellowgreen? |                 | knife of tools  |
|                 | style=flat-squa |                 | for a           |
|                 | re)             |                 | wide-range of   |
|                 |                 |                 | genomics        |
|                 |                 |                 | analysis tasks. |
|                 |                 |                 | The most        |
|                 |                 |                 | widely-used     |
|                 |                 |                 | tools enable    |
|                 |                 |                 | genome          |
|                 |                 |                 | arithmetic:     |
|                 |                 |                 | that is, set    |
|                 |                 |                 | theory on the   |
|                 |                 |                 | genome.         |
+-----------------+-----------------+-----------------+-----------------+
| Bismark         | ![](https://img | |cpu|           | Bismark is a    |
|                 | .shields.io/bad |                 | program to map  |
|                 | ge/version-0.19 |                 | bisulfite       |
|                 | .0-yellowgreen? |                 | treated         |
|                 | style=flat-squa |                 | sequencing      |
|                 | re)             |                 | reads to a      |
|                 |                 |                 | genome of       |
|                 |                 |                 | interest and    |
|                 |                 |                 | perform         |
|                 |                 |                 | methylation     |
|                 |                 |                 | calls in a      |
|                 |                 |                 | single step.    |
|                 |                 |                 | The output can  |
|                 |                 |                 | be easily       |
|                 |                 |                 | imported into a |
|                 |                 |                 | genome viewer,  |
|                 |                 |                 | such as         |
|                 |                 |                 | SeqMonk, and    |
|                 |                 |                 | enables a       |
|                 |                 |                 | researcher to   |
|                 |                 |                 | analyse the     |
|                 |                 |                 | methylation     |
|                 |                 |                 | levels of their |
|                 |                 |                 | samples         |
|                 |                 |                 | straight away.  |
+-----------------+-----------------+-----------------+-----------------+
| Bowtie          | ![](https://img | |cpu|           | Bowtie is an    |
|                 | .shields.io/bad |                 | ultrafast,      |
|                 | ge/version-1.2  |                 | memory-efficien |
|                 | .3-yellowgreen? |                 | t               |
|                 | style=flat-squa |                 | short read      |
|                 | re)             |                 | aligner geared  |
|                 |                 |                 | toward quickly  |
|                 |                 |                 | aligning large  |
|                 |                 |                 | sets of short   |
|                 |                 |                 | DNA sequences   |
|                 |                 |                 | (reads) to      |
|                 |                 |                 | large genomes.  |
+-----------------+-----------------+-----------------+-----------------+
| BWA             | ![](https://img | |cpu|           | BWA is a        |
|                 | .shields.io/bad |                 | software        |
|                 | ge/version-0.7. |                 | package for     |
|                 | 17-yellowgreen? |                 | mapping         |
|                 | style=flat-squa |                 | low-divergent   |
|                 | re)             |                 | sequences       |
|                 |                 |                 | against a large |
|                 |                 |                 | reference       |
|                 |                 |                 | genome, such as |
|                 |                 |                 | the human       |
|                 |                 |                 | genome.         |
+-----------------+-----------------+-----------------+-----------------+
| `CESM <https:// | ![](https://img | |cpu|           | Community Earth |
| docs.hpc.sjtu.e | .shields.io/bad |                 | System Model,   |
| du.cn/applicati | ge/version-1.2  |                 | or CESM, is a   |
| on/CESM/>`__    | -yellowgreen?   |                 | fully-coupled,  |
|                 | style=flat-squa |                 | community,      |
|                 | re)             |                 | global climate  |
|                 |                 |                 | model that      |
|                 |                 |                 | provides        |
|                 |                 |                 | state-of-the-ar |
|                 |                 |                 | t               |
|                 |                 |                 | computer        |
|                 |                 |                 | simulations of  |
|                 |                 |                 | the Earth’s     |
|                 |                 |                 | past, present,  |
|                 |                 |                 | and future      |
|                 |                 |                 | climate states. |
+-----------------+-----------------+-----------------+-----------------+
| CDO             | ![](https://img | |cpu|           | CDO is a        |
|                 | .shields.io/bad |                 | collection of   |
|                 | ge/version-1.9  |                 | command line    |
|                 | .8-yellowgreen? |                 | Operators to    |
|                 | style=flat-squa |                 | manipulate and  |
|                 | re)             |                 | analyse Climate |
|                 |                 |                 | and NWP model   |
|                 |                 |                 | Data.           |
+-----------------+-----------------+-----------------+-----------------+
| CP2K            | ![](https://img | |cpu|           | A freely        |
|                 | .shields.io/bad |                 | available       |
|                 | ge/version-6.1  |                 | program to      |
|                 | -yellowgreen?   |                 | perform         |
|                 | style=flat-squa |                 | atomistic and   |
|                 | re)             |                 | molecular       |
|                 |                 |                 | simulations of  |
|                 |                 |                 | solid state,    |
|                 |                 |                 | liquid,         |
|                 |                 |                 | molecular and   |
|                 |                 |                 | biological      |
|                 |                 |                 | systems. It     |
|                 |                 |                 | provides a      |
|                 |                 |                 | general         |
|                 |                 |                 | framework for   |
|                 |                 |                 | different       |
|                 |                 |                 | methods such as |
|                 |                 |                 | e.g. density    |
|                 |                 |                 | functional      |
|                 |                 |                 | theory (DFT)    |
|                 |                 |                 | using a mixed   |
|                 |                 |                 | Gaussian and    |
|                 |                 |                 | plane waves     |
|                 |                 |                 | approach (GPW), |
|                 |                 |                 | and classical   |
|                 |                 |                 | pair and        |
|                 |                 |                 | many-body       |
|                 |                 |                 | potentials.     |
+-----------------+-----------------+-----------------+-----------------+
| CUDA            | ![](https://img | |gpu|           |                 |
|                 | .shields.io/bad |                 |                 |
|                 | ge/version-10.0 |                 |                 |
|                 | .130-yellowgree |                 |                 |
|                 | n?style=flat-sq |                 |                 |
|                 | uare)           |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| Cufflinks       | ![](https://img | |cpu|           | Cufflinks       |
|                 | .shields.io/bad |                 | assembles       |
|                 | ge/version-2.2  |                 | transcripts,    |
|                 | .1-yellowgreen? |                 | estimates their |
|                 | style=flat-squa |                 | abundances, and |
|                 | re)             |                 | tests for       |
|                 |                 |                 | differential    |
|                 |                 |                 | expression and  |
|                 |                 |                 | regulation in   |
|                 |                 |                 | RNA-Seq         |
|                 |                 |                 | samples.        |
+-----------------+-----------------+-----------------+-----------------+
| DeepVariant     | ![](https://img | |cpu| |gpu|     |                 |
|                 | .shields.io/bad |                 |                 |
|                 | ge/version-10.0 |                 |                 |
|                 | .130-yellowgree |                 |                 |
|                 | n?style=flat-sq |                 |                 |
|                 | uare)           |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| FastQC          | ![](https://img | |cpu|           | FastQC aims to  |
|                 | .shields.io/bad |                 | provide a       |
|                 | ge/version-0.11 |                 | simple way to   |
|                 | .7-yellowgreen? |                 | do some quality |
|                 | style=flat-squa |                 | control checks  |
|                 | re)             |                 | on raw sequence |
|                 |                 |                 | data coming     |
|                 |                 |                 | from high       |
|                 |                 |                 | throughput      |
|                 |                 |                 | sequencing      |
|                 |                 |                 | pipelines.      |
+-----------------+-----------------+-----------------+-----------------+
| GATK            | ![](https://img | |cpu|           | The GATK is the |
|                 | .shields.io/bad |                 | industry        |
|                 | ge/version-3.8  |                 | standard for    |
|                 | -yellowgreen?   |                 | identifying     |
|                 | style=flat-squa |                 | SNPs and indels |
|                 | re)             |                 | in germline DNA |
|                 |                 |                 | and RNAseq      |
|                 |                 |                 | data.           |
+-----------------+-----------------+-----------------+-----------------+
| Gaussian        |                 | |cpu|           | Gaussian is a   |
|                 |                 |                 | general purpose |
|                 |                 |                 | computational   |
|                 |                 |                 | chemistry       |
|                 |                 |                 | software        |
|                 |                 |                 | package         |
|                 |                 |                 | initially       |
|                 |                 |                 | released in     |
|                 |                 |                 | 1970 by John    |
|                 |                 |                 | Pople and his   |
|                 |                 |                 | research group  |
|                 |                 |                 | at Carnegie     |
|                 |                 |                 | Mellon          |
|                 |                 |                 | University as   |
|                 |                 |                 | Gaussian 70.    |
+-----------------+-----------------+-----------------+-----------------+
| Geant4          | ![](https://img | |cpu|           |                 |
|                 | .shields.io/bad |                 |                 |
|                 | ge/version-10.6 |                 |                 |
|                 | .2-yellowgreen? |                 |                 |
|                 | style=flat-squa |                 |                 |
|                 | re)             |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
|                 |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| GMAP-GSNAP      | ![](https://img | |cpu|           | GMAP is a tools |
|                 | .shields.io/bad |                 | for rapidly and |
|                 | ge/version-2019 |                 | accurately      |
|                 | -5-12-yellowgre |                 | mapping and     |
|                 | en?style=flat-s |                 | aligning cDNA   |
|                 | quare)          |                 | sequences to    |
|                 |                 |                 | genomic         |
|                 |                 |                 | sequences.      |
|                 |                 |                 | GSNAP is        |
|                 |                 |                 | designed to     |
|                 |                 |                 | align short     |
|                 |                 |                 | reads from NGS  |
|                 |                 |                 | data and allow  |
|                 |                 |                 | detection of    |
|                 |                 |                 | short and long  |
|                 |                 |                 | range splicing  |
|                 |                 |                 | de novo or with |
|                 |                 |                 | a database of   |
|                 |                 |                 | know juctions.  |
+-----------------+-----------------+-----------------+-----------------+
| Gnuplot         |                 | |studio|        | Gnuplot is a    |
|                 |                 |                 | command-driven  |
|                 |                 |                 | interactive     |
|                 |                 |                 | function        |
|                 |                 |                 | plotting        |
|                 |                 |                 | program. It can |
|                 |                 |                 | be used to plot |
|                 |                 |                 | functions and   |
|                 |                 |                 | data points in  |
|                 |                 |                 | both two- and   |
|                 |                 |                 | three-          |
|                 |                 |                 | dimensional     |
|                 |                 |                 | plots in many   |
|                 |                 |                 | different       |
|                 |                 |                 | formats. It was |
|                 |                 |                 | originally      |
|                 |                 |                 | made to allow   |
|                 |                 |                 | scientists and  |
|                 |                 |                 | students to     |
|                 |                 |                 | visualize       |
|                 |                 |                 | mathematical    |
|                 |                 |                 | functions and   |
|                 |                 |                 | data            |
|                 |                 |                 | interactively,  |
|                 |                 |                 | but has grown   |
|                 |                 |                 | to support many |
|                 |                 |                 | non-interactive |
|                 |                 |                 | uses such as    |
|                 |                 |                 | web scripting.  |
|                 |                 |                 | It is also used |
|                 |                 |                 | as a plotting   |
|                 |                 |                 | engine by       |
|                 |                 |                 | third-party     |
|                 |                 |                 | applications    |
|                 |                 |                 | like Octave.    |
+-----------------+-----------------+-----------------+-----------------+
| GraphMap        | ![](https://img | |cpu|           | A highly        |
|                 | .shields.io/bad |                 | sensitive and   |
|                 | ge/version-0.3  |                 | accurate mapper |
|                 | .0-yellowgreen? |                 | for long,       |
|                 | style=flat-squa |                 | error-prone     |
|                 | re)             |                 | reads.          |
+-----------------+-----------------+-----------------+-----------------+
| `Gromacs <https | ![](https://img | |cpu|           | GROMACS is a    |
| ://docs.hpc.sjt | .shields.io/bad | |gpu|\ |arm|    | versatile       |
| u.edu.cn/applic | ge/version-2020 |                 | package to      |
| ation/Gromacs/> | -yellowgreen?   |                 | perform         |
| `__             | style=flat-squa |                 | molecular       |
|                 | re)             |                 | dynamics,       |
|                 |                 |                 | i.e. simulate   |
|                 |                 |                 | the Newtonian   |
|                 |                 |                 | equations of    |
|                 |                 |                 | motion for      |
|                 |                 |                 | systems with    |
|                 |                 |                 | hundreds to     |
|                 |                 |                 | millions of     |
|                 |                 |                 | particles. It   |
|                 |                 |                 | is primarily    |
|                 |                 |                 | designed for    |
|                 |                 |                 | biochemical     |
|                 |                 |                 | molecules like  |
|                 |                 |                 | proteins,       |
|                 |                 |                 | lipids and      |
|                 |                 |                 | nucleic acids   |
|                 |                 |                 | that have a lot |
|                 |                 |                 | of complicated  |
|                 |                 |                 | bonded          |
|                 |                 |                 | interactions,   |
|                 |                 |                 | but since       |
|                 |                 |                 | GROMACS is      |
|                 |                 |                 | extremely fast  |
|                 |                 |                 | at calculating  |
|                 |                 |                 | the nonbonded   |
|                 |                 |                 | interactions    |
|                 |                 |                 | (that usually   |
|                 |                 |                 | dominate        |
|                 |                 |                 | simulations)    |
|                 |                 |                 | many groups are |
|                 |                 |                 | also using it   |
|                 |                 |                 | for research on |
|                 |                 |                 | non-biological  |
|                 |                 |                 | systems,        |
|                 |                 |                 | e.g. polymers.  |
+-----------------+-----------------+-----------------+-----------------+
| HISAT2          | ![](https://img | |cpu|           | HISAT2 is a     |
|                 | .shields.io/bad |                 | fast and        |
|                 | ge/version-2.1  |                 | sensitive       |
|                 | .0-yellowgreen? |                 | alignment       |
|                 | style=flat-squa |                 | program for     |
|                 | re)             |                 | mapping         |
|                 |                 |                 | next-generation |
|                 |                 |                 | sequencing      |
|                 |                 |                 | reads (both DNA |
|                 |                 |                 | and RNA) to a   |
|                 |                 |                 | population of   |
|                 |                 |                 | human genomes   |
|                 |                 |                 | as well as to a |
|                 |                 |                 | single          |
|                 |                 |                 | reference       |
|                 |                 |                 | genome.         |
+-----------------+-----------------+-----------------+-----------------+
| Keras           |                 |                 | Keras is a      |
|                 |                 |                 | minimalist,     |
|                 |                 |                 | highly modular  |
|                 |                 |                 | neural networks |
|                 |                 |                 | library written |
|                 |                 |                 | in Python and   |
|                 |                 |                 | capable on      |
|                 |                 |                 | running on top  |
|                 |                 |                 | of either       |
|                 |                 |                 | TensorFlow or   |
|                 |                 |                 | Theano. It was  |
|                 |                 |                 | developed with  |
|                 |                 |                 | a focus on      |
|                 |                 |                 | enabling fast   |
|                 |                 |                 | experimentation |
|                 |                 |                 | . Being able to |
|                 |                 |                 | go from idea to |
|                 |                 |                 | result with the |
|                 |                 |                 | least possible  |
|                 |                 |                 | delay is key to |
|                 |                 |                 | doing good      |
|                 |                 |                 | research.       |
+-----------------+-----------------+-----------------+-----------------+
| `LAMMPS <https: | ![](https://img | |cpu|           | (Large-scale    |
| //docs.hpc.sjtu | .shields.io/bad | |gpu|\ |arm|    | Atomic/Molecula |
| .edu.cn/applica | ge/version-2020 |                 | r               |
| tion/Lammps/>`_ | -yellowgreen?   |                 | Massively       |
| _               | style=flat-squa |                 | Parallel        |
|                 | re)             |                 | Simulator) a    |
|                 |                 |                 | classical       |
|                 |                 |                 | molecular       |
|                 |                 |                 | dynamics code.  |
+-----------------+-----------------+-----------------+-----------------+
| LUMPY-SV        | ![](https://img | |cpu|           | A general       |
|                 | .shields.io/bad |                 | probabilistic   |
|                 | ge/version-0.2. |                 | framework for   |
|                 | 13-yellowgreen? |                 | structural      |
|                 | style=flat-squa |                 | variant         |
|                 | re)             |                 | discovery.      |
+-----------------+-----------------+-----------------+-----------------+
| MEGAHIT         | ![](https://img | |cpu|           | MEGAHIT is an   |
|                 | .shields.io/bad |                 | ultra-fast and  |
|                 | ge/version-1.1  |                 | memory-efficien |
|                 | .4-yellowgreen? |                 | t               |
|                 | style=flat-squa |                 | NGS assembler.  |
|                 | re)             |                 | It is optimized |
|                 |                 |                 | for             |
|                 |                 |                 | metagenomes,    |
|                 |                 |                 | but also works  |
|                 |                 |                 | well on generic |
|                 |                 |                 | single genome   |
|                 |                 |                 | assembly (small |
|                 |                 |                 | or mammalian    |
|                 |                 |                 | size) and       |
|                 |                 |                 | single-cell     |
|                 |                 |                 | assembly.       |
+-----------------+-----------------+-----------------+-----------------+
| METIS           | ![](https://img | |cpu|           | METIS is a set  |
|                 | .shields.io/bad |                 | of serial       |
|                 | ge/version-5.1  |                 | programs for    |
|                 | .0-yellowgreen? |                 | partitioning    |
|                 | style=flat-squa |                 | graphs,         |
|                 | re)             |                 | partitioning    |
|                 |                 |                 | finite element  |
|                 |                 |                 | meshes, and     |
|                 |                 |                 | producing fill  |
|                 |                 |                 | reducing        |
|                 |                 |                 | orderings for   |
|                 |                 |                 | sparse          |
|                 |                 |                 | matrices.       |
+-----------------+-----------------+-----------------+-----------------+
| MrBayes         | ![](https://img | |cpu|           | MrBayes is a    |
|                 | .shields.io/bad |                 | program for     |
|                 | ge/version-3.2. |                 | Bayesian        |
|                 | 7a-yellowgreen? |                 | inference and   |
|                 | style=flat-squa |                 | model choice    |
|                 | re)             |                 | across a wide   |
|                 |                 |                 | range of        |
|                 |                 |                 | phylogenetic    |
|                 |                 |                 | and             |
|                 |                 |                 | evolutionary    |
|                 |                 |                 | models.         |
+-----------------+-----------------+-----------------+-----------------+
| NCBI-RMBlastn   | ![](https://img | |cpu|           | RMBlast is a    |
|                 | .shields.io/bad |                 | RepeatMasker    |
|                 | ge/version-2.2. |                 | compatible      |
|                 | 28-yellowgreen? |                 | version of the  |
|                 | style=flat-squa |                 | standard NCBI   |
|                 | re)             |                 | BLAST suite.    |
|                 |                 |                 | The primary     |
|                 |                 |                 | difference      |
|                 |                 |                 | between this    |
|                 |                 |                 | distribution    |
|                 |                 |                 | and the NCBI    |
|                 |                 |                 | distribution is |
|                 |                 |                 | the addition of |
|                 |                 |                 | a new program   |
|                 |                 |                 | “rmblastn” for  |
|                 |                 |                 | use with        |
|                 |                 |                 | RepeatMasker    |
|                 |                 |                 | and             |
|                 |                 |                 | RepeatModeler.  |
+-----------------+-----------------+-----------------+-----------------+
| `Nektar++ <http | ![](https://img | |cpu|           | Nektar++ is a   |
| s://docs.hpc.sj | .shields.io/bad |                 | spectral/hp     |
| tu.edu.cn/appli | ge/version-5.0  |                 | element         |
| cation/Nektar/> | .0-yellowgreen? |                 | framework       |
| `__             | style=flat-squa |                 | designed to     |
|                 | re)             |                 | support the     |
|                 |                 |                 | construction of |
|                 |                 |                 | efficient       |
|                 |                 |                 | high-performanc |
|                 |                 |                 | e               |
|                 |                 |                 | scalable        |
|                 |                 |                 | solvers for a   |
|                 |                 |                 | wide range of   |
|                 |                 |                 | partial         |
|                 |                 |                 | differential    |
|                 |                 |                 | equations       |
|                 |                 |                 | (PDE).          |
+-----------------+-----------------+-----------------+-----------------+
| `nwChem <https: | ![](https://img | |cpu|           | NWChem aims to  |
| //docs.hpc.sjtu | .shields.io/bad |                 | provide its     |
| .edu.cn/applica | ge/version-6.8  |                 | users with      |
| tion/nwchem/>`_ | .1-yellowgreen? |                 | computational   |
| _               | style=flat-squa |                 | chemistry tools |
|                 | re)             |                 | that are        |
|                 |                 |                 | scalable both   |
|                 |                 |                 | in their        |
|                 |                 |                 | ability to      |
|                 |                 |                 | treat large     |
|                 |                 |                 | scientific      |
|                 |                 |                 | computational   |
|                 |                 |                 | chemistry       |
|                 |                 |                 | problems        |
|                 |                 |                 | efficiently,    |
|                 |                 |                 | and in their    |
|                 |                 |                 | use of          |
|                 |                 |                 | available       |
|                 |                 |                 | parallel        |
|                 |                 |                 | computing       |
|                 |                 |                 | resources from  |
|                 |                 |                 | high-performanc |
|                 |                 |                 | e               |
|                 |                 |                 | parallel        |
|                 |                 |                 | supercomputers  |
|                 |                 |                 | to conventional |
|                 |                 |                 | workstation     |
|                 |                 |                 | clusters. The   |
|                 |                 |                 | NWChem software |
|                 |                 |                 | can handle:     |
|                 |                 |                 | Biomolecules,   |
|                 |                 |                 | nanostructures, |
|                 |                 |                 | and             |
|                 |                 |                 | solid-state;    |
|                 |                 |                 | From quantum to |
|                 |                 |                 | classical, and  |
|                 |                 |                 | all             |
|                 |                 |                 | combinations;   |
|                 |                 |                 | Gaussian basis  |
|                 |                 |                 | functions or    |
|                 |                 |                 | plane-waves;    |
|                 |                 |                 | Scaling from    |
|                 |                 |                 | one to          |
|                 |                 |                 | thousands of    |
|                 |                 |                 | processors;     |
|                 |                 |                 | Properties and  |
|                 |                 |                 | relativity.     |
+-----------------+-----------------+-----------------+-----------------+
| `Octave <https: | ![](https://img | |cpu|           | GNU Octave is   |
| //docs.hpc.sjtu | .shields.io/bad |                 | software        |
| .edu.cn/applica | ge/version-5.2  |                 | featuring a     |
| tion/Octave/>`_ | .0-yellowgreen? |                 | high-level      |
| _               | style=flat-squa |                 | programming     |
|                 | re)             |                 | language,       |
|                 |                 |                 | primarily       |
|                 |                 |                 | intended for    |
|                 |                 |                 | numerical       |
|                 |                 |                 | computations.   |
+-----------------+-----------------+-----------------+-----------------+
| `OpenFoam <http | 7, 1712, 1812,  | |cpu|           | OpenFOAM is an  |
| s://docs.hpc.sj | 1912            |                 | open-source     |
| tu.edu.cn/appli |                 |                 | toolbox for     |
| cation/OpenFoam |                 |                 | computational   |
| />`__           |                 |                 | fluid dynamics. |
|                 |                 |                 | OpenFOAM        |
|                 |                 |                 | consists of     |
|                 |                 |                 | generic tools   |
|                 |                 |                 | to simulate     |
|                 |                 |                 | complex physics |
|                 |                 |                 | for a variety   |
|                 |                 |                 | of fields of    |
|                 |                 |                 | interest, from  |
|                 |                 |                 | fluid flows     |
|                 |                 |                 | involving       |
|                 |                 |                 | chemical        |
|                 |                 |                 | reactions,      |
|                 |                 |                 | turbulence and  |
|                 |                 |                 | heat transfer,  |
|                 |                 |                 | to solid        |
|                 |                 |                 | dynamics,       |
|                 |                 |                 | electromagnetis |
|                 |                 |                 | m               |
|                 |                 |                 | and the pricing |
|                 |                 |                 | of financial    |
|                 |                 |                 | options.        |
+-----------------+-----------------+-----------------+-----------------+
| OVITO           |                 | |cpu|           | OVITO (Open     |
|                 |                 |                 | Visualization   |
|                 |                 |                 | Tool) is a      |
|                 |                 |                 | scientific      |
|                 |                 |                 | visualization   |
|                 |                 |                 | and analysis    |
|                 |                 |                 | package for     |
|                 |                 |                 | atomistic and   |
|                 |                 |                 | particle-based  |
|                 |                 |                 | simulation      |
|                 |                 |                 | data.           |
+-----------------+-----------------+-----------------+-----------------+
| Paraview        | ![](https://img | |cpu|           | Paraview is a   |
|                 | .shields.io/bad |                 | data            |
|                 | ge/version-0.4  |                 | visualisation   |
|                 | .1-yellowgreen? |                 | and analysis    |
|                 | style=flat-squa |                 | package. Whilst |
|                 | re)             |                 | ARCHER compute  |
|                 |                 |                 | or login nodes  |
|                 |                 |                 | do not have     |
|                 |                 |                 | graphics cards  |
|                 |                 |                 | installed in    |
|                 |                 |                 | them paraview   |
|                 |                 |                 | is installed so |
|                 |                 |                 | the             |
|                 |                 |                 | visualisation   |
|                 |                 |                 | libraries and   |
|                 |                 |                 | applications    |
|                 |                 |                 | can be used to  |
|                 |                 |                 | post-process    |
|                 |                 |                 | simulation      |
|                 |                 |                 | data. To this   |
|                 |                 |                 | end the         |
|                 |                 |                 | pvserver        |
|                 |                 |                 | application has |
|                 |                 |                 | been installed, |
|                 |                 |                 | along with the  |
|                 |                 |                 | paraview        |
|                 |                 |                 | libraries and   |
|                 |                 |                 | client          |
|                 |                 |                 | application.    |
+-----------------+-----------------+-----------------+-----------------+
| Perl            |                 | |cpu|           |                 |
+-----------------+-----------------+-----------------+-----------------+
| Picard          | ![](https://img | |cpu|           | Picard is a set |
|                 | .shields.io/bad |                 | of command line |
|                 | ge/version-2.19 |                 | tools for       |
|                 | .0-yellowgreen? |                 | manipulating    |
|                 | style=flat-squa |                 | high-throughput |
|                 | re)             |                 | sequencing      |
|                 |                 |                 | (HTS) data and  |
|                 |                 |                 | formats such as |
|                 |                 |                 | SAM/BAM/CRAM    |
|                 |                 |                 | and VCF.        |
+-----------------+-----------------+-----------------+-----------------+
| Python          | ![](https://img | |cpu| |gpu|     |                 |
|                 | .shields.io/bad |                 |                 |
|                 | ge/version-3.7  |                 |                 |
|                 | .4-yellowgreen? |                 |                 |
|                 | style=flat-squa |                 |                 |
|                 | re)             |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| `Pytorch <https | ![](https://img | |gpu|           | PyTorch is an   |
| ://docs.hpc.sjt | .shields.io/bad |                 | open source     |
| u.edu.cn/applic | ge/version-1.6  |                 | machine         |
| ation/Pytorch/> | .0-yellowgreen? |                 | learning        |
| `__             | style=flat-squa |                 | library based   |
|                 | re)             |                 | on the Torch    |
|                 |                 |                 | library, used   |
|                 |                 |                 | for             |
|                 |                 |                 | applications    |
|                 |                 |                 | such as         |
|                 |                 |                 | computer vision |
|                 |                 |                 | and natural     |
|                 |                 |                 | language        |
|                 |                 |                 | processing,     |
|                 |                 |                 | primarily       |
|                 |                 |                 | developed by    |
|                 |                 |                 | Facebook’s AI   |
|                 |                 |                 | Research lab.   |
+-----------------+-----------------+-----------------+-----------------+
| `Quantum-Espres | ![](https://img | |cpu|           | Quantum         |
| so <https://doc | .shields.io/bad |                 | Espresso is an  |
| s.hpc.sjtu.edu. | ge/version-6.6  |                 | integrated      |
| cn/application/ | -yellowgreen?   |                 | suite of        |
| Quantum-Espress | style=flat-squa |                 | Open-Source     |
| o/>`__          | re)             |                 | computer codes  |
|                 |                 |                 | for             |
|                 |                 |                 | electronic-stru |
|                 |                 |                 | cture           |
|                 |                 |                 | calculations    |
|                 |                 |                 | and materials   |
|                 |                 |                 | modeling at the |
|                 |                 |                 | nanoscale. It   |
|                 |                 |                 | is based on     |
|                 |                 |                 | density-functio |
|                 |                 |                 | nal             |
|                 |                 |                 | theory, plane   |
|                 |                 |                 | waves, and      |
|                 |                 |                 | pseudopotential |
|                 |                 |                 | s.              |
+-----------------+-----------------+-----------------+-----------------+
| `R <https://doc | 1.1.8, 3.6.2    | |cpu|           | R is a          |
| s.hpc.sjtu.edu. |                 |                 | programming     |
| cn/application/ |                 |                 | language and    |
| R/>`__          |                 |                 | free software   |
|                 |                 |                 | environment for |
|                 |                 |                 | statistical     |
|                 |                 |                 | computing and   |
|                 |                 |                 | graphics        |
|                 |                 |                 | supported by    |
|                 |                 |                 | the R           |
|                 |                 |                 | Foundation for  |
|                 |                 |                 | Statistical     |
|                 |                 |                 | Computing.      |
+-----------------+-----------------+-----------------+-----------------+
| `Relion <https: | ![](https://img | |gpu|           | REgularised     |
| //docs.hpc.sjtu | .shields.io/bad |                 | LIkelihood      |
| .edu.cn/applica | ge/version-3.0  |                 | OptimisatioN    |
| tion/Relion/>`_ | .8-yellowgreen? |                 | (RELION)        |
| _               | style=flat-squa |                 | employs an      |
|                 | re)             |                 | empirical       |
|                 |                 |                 | Bayesian        |
|                 |                 |                 | approach to     |
|                 |                 |                 | refinement of   |
|                 |                 |                 | (multiple) 3D   |
|                 |                 |                 | reconstructions |
|                 |                 |                 | or 2D class     |
|                 |                 |                 | averages in     |
|                 |                 |                 | electron        |
|                 |                 |                 | cryomicroscopy. |
+-----------------+-----------------+-----------------+-----------------+
| RNA-SeQC        | ![](https://img | |cpu|           | RNA-SeQC is a   |
|                 | .shields.io/bad |                 | java program    |
|                 | ge/version-1.1  |                 | which computes  |
|                 | .8-yellowgreen? |                 | a series of     |
|                 | style=flat-squa |                 | quality control |
|                 | re)             |                 | metrics for     |
|                 |                 |                 | RNA-seq data.   |
+-----------------+-----------------+-----------------+-----------------+
| Salmon          | ![](https://img | |cpu|           | Salmon is a     |
|                 | .shields.io/bad |                 | tool for        |
|                 | ge/version-0.14 |                 | wicked-fast     |
|                 | .1-yellowgreen? |                 | transcript      |
|                 | style=flat-squa |                 | quantification  |
|                 | re)             |                 | from RNA-seq    |
|                 |                 |                 | data.           |
+-----------------+-----------------+-----------------+-----------------+
| SAMtools        | ![](https://img | |cpu|           | SAM Tools       |
|                 | .shields.io/bad |                 | provide various |
|                 | ge/version-1.9- |                 | utilities for   |
|                 | yellowgreen?    |                 | manipulating    |
|                 | style=flat-squa |                 | alignments in   |
|                 | re)             |                 | the SAM format. |
+-----------------+-----------------+-----------------+-----------------+
| SIESTA          | ![](https://img | |cpu|           | SIESTA is both  |
|                 | .shields.io/bad |                 | a method and    |
|                 | ge/version-4.0. |                 | its computer    |
|                 | 1-yellowgreen?  |                 | program         |
|                 | style=flat-squa |                 | implementation, |
|                 | re)             |                 | to perform      |
|                 |                 |                 | efficient       |
|                 |                 |                 | electronic      |
|                 |                 |                 | structure       |
|                 |                 |                 | calculations    |
|                 |                 |                 | and ab initio   |
|                 |                 |                 | molecular       |
|                 |                 |                 | dynamics        |
|                 |                 |                 | simulations of  |
|                 |                 |                 | molecules and   |
|                 |                 |                 | solids.         |
|                 |                 |                 | SIESTA's        |
|                 |                 |                 | efficiency      |
|                 |                 |                 | stems from the  |
|                 |                 |                 | use of a basis  |
|                 |                 |                 | set of strictly |
|                 |                 |                 | -localized      |
|                 |                 |                 | atomic orbitals |
|                 |                 |                 | . A very        |
|                 |                 |                 | important       |
|                 |                 |                 | feature of the  |
|                 |                 |                 | code is that    |
|                 |                 |                 | its accuracy    |
|                 |                 |                 | and cost can be |
|                 |                 |                 | tuned in a wide |
|                 |                 |                 | range, from     |
|                 |                 |                 | quick           |
|                 |                 |                 | exploratory     |
|                 |                 |                 | calculations to |
|                 |                 |                 | highly accurate |
|                 |                 |                 | simulations     |
|                 |                 |                 | matching the    |
|                 |                 |                 | quality of      |
|                 |                 |                 | other           |
|                 |                 |                 | approaches,     |
|                 |                 |                 | such as plane-  |
|                 |                 |                 | wave methods.   |
+-----------------+-----------------+-----------------+-----------------+
| SOAPdenovo2     | 240             | |cpu|           | SOAPdenovo is a |
|                 |                 |                 | novel           |
|                 |                 |                 | short-read      |
|                 |                 |                 | assembly method |
|                 |                 |                 | that can build  |
|                 |                 |                 | a de novo draft |
|                 |                 |                 | assembly for    |
|                 |                 |                 | the human-sized |
|                 |                 |                 | genomes.        |
+-----------------+-----------------+-----------------+-----------------+
| SRAtoolkit      | ![](https://img | |cpu|           | The SRA Toolkit |
|                 | .shields.io/bad |                 | and SDK from    |
|                 | ge/version-2.9  |                 | NCBI is a       |
|                 | .6-yellowgreen? |                 | collection of   |
|                 | style=flat-squa |                 | tools and       |
|                 | re)             |                 | libraries for   |
|                 |                 |                 | using data in   |
|                 |                 |                 | the INSDC       |
|                 |                 |                 | Sequence Read   |
|                 |                 |                 | Archives.       |
+-----------------+-----------------+-----------------+-----------------+
| STAR            | ![](https://img | |cpu|           | Spliced         |
|                 | .shields.io/bad |                 | Transcripts     |
|                 | ge/version-2.7  |                 | Alignment to a  |
|                 | .0-yellowgreen? |                 | Reference       |
|                 | style=flat-squa |                 | (STAR) software |
|                 | re)             |                 | is based on a   |
|                 |                 |                 | previously      |
|                 |                 |                 | undescribed     |
|                 |                 |                 | RNA-seq         |
|                 |                 |                 | alignment       |
|                 |                 |                 | algorithm that  |
|                 |                 |                 | uses sequential |
|                 |                 |                 | maximum         |
|                 |                 |                 | mappable seed   |
|                 |                 |                 | search in       |
|                 |                 |                 | uncompressed    |
|                 |                 |                 | suffix arrays   |
|                 |                 |                 | followed by     |
|                 |                 |                 | seed clustering |
|                 |                 |                 | and stitching   |
|                 |                 |                 | procedure.      |
+-----------------+-----------------+-----------------+-----------------+
| `STAR-CCM+ <htt |                 | |cpu|           | Much more than  |
| ps://docs.hpc.s |                 |                 | just a CFD      |
| jtu.edu.cn/appl |                 |                 | solver,         |
| ication/star-cc |                 |                 | STAR-CCM+ is an |
| m/>`__          |                 |                 | entire          |
|                 |                 |                 | engineering     |
|                 |                 |                 | process for     |
|                 |                 |                 | solving         |
|                 |                 |                 | problems        |
|                 |                 |                 | involving flow  |
|                 |                 |                 | (of fluids or   |
|                 |                 |                 | solids), heat   |
|                 |                 |                 | transfer and    |
|                 |                 |                 | stress.         |
+-----------------+-----------------+-----------------+-----------------+
| StringTie       | ![](https://img | |cpu|           | StringTie is a  |
|                 | .shields.io/bad |                 | fast and highly |
|                 | ge/version-1.3. |                 | efficient       |
|                 | 4d-yellowgreen? |                 | assembler of    |
|                 | style=flat-squa |                 | RNA-Seq         |
|                 | re)             |                 | alignments into |
|                 |                 |                 | potential       |
|                 |                 |                 | transcripts.    |
+-----------------+-----------------+-----------------+-----------------+
| STRique         |                 | |cpu|           | STRique is a    |
|                 |                 |                 | python package  |
|                 |                 |                 | to analyze      |
|                 |                 |                 | repeat          |
|                 |                 |                 | expansion and   |
|                 |                 |                 | methylation     |
|                 |                 |                 | states of short |
|                 |                 |                 | tandem repeats  |
|                 |                 |                 | (STR) in Oxford |
|                 |                 |                 | Nanopore        |
|                 |                 |                 | Technology(ONT) |
|                 |                 |                 | long read       |
|                 |                 |                 | sequencing      |
|                 |                 |                 | data.           |
+-----------------+-----------------+-----------------+-----------------+
| `TensorFlow <ht | ![](https://img | |gpu|           | TensorFlow is a |
| tps://docs.hpc. | .shields.io/bad |                 | free and        |
| sjtu.edu.cn/app | ge/version-2.0  |                 | open-source     |
| lication/Tensor | .0-yellowgreen? |                 | software        |
| Flow/>`__       | style=flat-squa |                 | library for     |
|                 | re)             |                 | dataflow and    |
|                 |                 |                 | differentiable  |
|                 |                 |                 | programming     |
|                 |                 |                 | across a range  |
|                 |                 |                 | of tasks. It is |
|                 |                 |                 | a symbolic math |
|                 |                 |                 | library, and is |
|                 |                 |                 | also used for   |
|                 |                 |                 | machine         |
|                 |                 |                 | learning        |
|                 |                 |                 | applications    |
|                 |                 |                 | such as neural  |
|                 |                 |                 | networks.       |
+-----------------+-----------------+-----------------+-----------------+
| TopHat          | ![](https://img | |cpu|           | TopHat is a     |
|                 | .shields.io/bad |                 | program that    |
|                 | ge/version-2.1  |                 | aligns RNA-Seq  |
|                 | .2-yellowgreen? |                 | reads to a      |
|                 | style=flat-squa |                 | genome in order |
|                 | re)             |                 | to identify     |
|                 |                 |                 | exon-exon       |
|                 |                 |                 | splice          |
|                 |                 |                 | junctions.      |
+-----------------+-----------------+-----------------+-----------------+
| VarDictJava     | ![](https://img | |cpu|           | VarDictJava is  |
|                 | .shields.io/bad |                 | a variant       |
|                 | ge/version-1.5  |                 | discovery       |
|                 | .1-yellowgreen? |                 | program written |
|                 | style=flat-squa |                 | in Java and     |
|                 | re)             |                 | Perl.           |
+-----------------+-----------------+-----------------+-----------------+
| `VASP <https:// | ![](https://img | |cpu| |gpu|     | A package for   |
| docs.hpc.sjtu.e | .shields.io/bad |                 | ab initio,      |
| du.cn/applicati | ge/version-5.4  |                 | quantum-mechani |
| on/VASP/>`__    | .4-yellowgreen? |                 | cal,            |
|                 | style=flat-squa |                 | molecular       |
|                 | re)             |                 | dynamics        |
|                 |                 |                 | simulations.    |
+-----------------+-----------------+-----------------+-----------------+
| VSEARCH         | ![](https://img | |cpu|           | VSEARCH stands  |
|                 | .shields.io/bad |                 | for vectorized  |
|                 | ge/version-2.4  |                 | search, as the  |
|                 | .3-yellowgreen? |                 | tool takes      |
|                 | style=flat-squa |                 | advantage of    |
|                 | re)             |                 | parallelism in  |
|                 |                 |                 | the form of     |
|                 |                 |                 | SIMD            |
|                 |                 |                 | vectorization   |
|                 |                 |                 | as well as      |
|                 |                 |                 | multiple        |
|                 |                 |                 | threads to      |
|                 |                 |                 | perform         |
|                 |                 |                 | accurate        |
|                 |                 |                 | alignments at   |
|                 |                 |                 | high speed.     |
+-----------------+-----------------+-----------------+-----------------+
| `VMD <https://d | ![](https://img | |cpu|           | VMD is a        |
| ocs.hpc.sjtu.ed | .shields.io/bad |                 | molecular       |
| u.cn/applicatio | ge/version-1.9  |                 | visualization   |
| n/VMD/>`__      | .4-yellowgreen? |                 | program for     |
|                 | style=flat-squa |                 | displaying,     |
|                 | re)             |                 | animating, and  |
|                 |                 |                 | analyzing large |
|                 |                 |                 | biomolecular    |
|                 |                 |                 | systems using   |
|                 |                 |                 | 3-D graphics    |
|                 |                 |                 | and built-in    |
|                 |                 |                 | scripting.      |
+-----------------+-----------------+-----------------+-----------------+

.. |cpu| image:: https://img.shields.io/badge/CPU-blue?style=flat-square
.. |gpu| image:: https://img.shields.io/badge/DGX2-green?style=flat-square
.. |arm| image:: https://img.shields.io/badge/-arm-yellow
.. |studio| image::  https://img.shields.io/badge/Studio-inactive?style=flat-square
.. |singularity| image:: https://img.shields.io/badge/-singularity-blueviolet

