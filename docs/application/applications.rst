.. _applications:

===========
Pi 上的软件
===========

本文档介绍 Pi 上的软件。 商业软件需用户自行获取版权并安装。 |cpu| |gpu|
|arm| |singularity| 标签表明软件有 cpu, gpu, arm 和 singularity 版本

.. _pi-上的软件-1:

Pi 上的软件
-----------

+-----------------+-----------------+-----------------+-----------------+
| Name            | Version         | Distribution    | Introduction    |
+=================+=================+=================+=================+
| `               | 8.10.3          | |cpu|           | ABINIT is a     |
| ABINIT <https:/ |                 |                 | package whose   |
| /docs.hpc.sjtu. |                 |                 | main program    |
| edu.cn/applicat |                 |                 | allows one to   |
| ion/abinit/>`__ |                 |                 | find the total  |
|                 |                 |                 | energy, charge  |
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
|                 |                 |                 | p               |
|                 |                 |                 | seudopotentials |
|                 |                 |                 | and a planewave |
|                 |                 |                 | or wavelet      |
|                 |                 |                 | basis.          |
+-----------------+-----------------+-----------------+-----------------+
| `Amber <https:  |                 | |cpu| |gpu|     | A package of    |
| //docs.hpc.sjtu |                 |                 | molecular       |
| .edu.cn/applica |                 |                 | simulation      |
| tion/Amber/>`__ |                 |                 | programs and    |
|                 |                 |                 | analysis tools. |
+-----------------+-----------------+-----------------+-----------------+
| BCFtools        | 1.9             | |cpu|           | BCFtools is a   |
|                 |                 |                 | program for     |
|                 |                 |                 | variant calling |
|                 |                 |                 | and             |
|                 |                 |                 | manipulating    |
|                 |                 |                 | files in the    |
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
|                 |                 |                 | B               |
|                 |                 |                 | GZF-compressed. |
+-----------------+-----------------+-----------------+-----------------+
| Bedtools2       | 2.27.1          | |cpu|           | The bedtools    |
|                 |                 |                 | utilities are a |
|                 |                 |                 | swiss-army      |
|                 |                 |                 | knife of tools  |
|                 |                 |                 | for a           |
|                 |                 |                 | wide-range of   |
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
| Bismark         | 0.19.0          | |cpu|           | Bismark is a    |
|                 |                 |                 | program to map  |
|                 |                 |                 | bisulfite       |
|                 |                 |                 | treated         |
|                 |                 |                 | sequencing      |
|                 |                 |                 | reads to a      |
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
| Bowtie          | 1.2.3           | |cpu|           | Bowtie is an    |
|                 |                 |                 | ultrafast,      |
|                 |                 |                 | m               |
|                 |                 |                 | emory-efficient |
|                 |                 |                 | short read      |
|                 |                 |                 | aligner geared  |
|                 |                 |                 | toward quickly  |
|                 |                 |                 | aligning large  |
|                 |                 |                 | sets of short   |
|                 |                 |                 | DNA sequences   |
|                 |                 |                 | (reads) to      |
|                 |                 |                 | large genomes.  |
+-----------------+-----------------+-----------------+-----------------+
| BWA             | 0.7.17          | |cpu|           | BWA is a        |
|                 |                 |                 | software        |
|                 |                 |                 | package for     |
|                 |                 |                 | mapping         |
|                 |                 |                 | low-divergent   |
|                 |                 |                 | sequences       |
|                 |                 |                 | against a large |
|                 |                 |                 | reference       |
|                 |                 |                 | genome, such as |
|                 |                 |                 | the human       |
|                 |                 |                 | genome.         |
+-----------------+-----------------+-----------------+-----------------+
| `CESM <https    |                 | |cpu|           | Community Earth |
| ://docs.hpc.sjt |                 |                 | System Model,   |
| u.edu.cn/applic |                 |                 | or CESM, is a   |
| ation/CESM/>`__ |                 |                 | fully-coupled,  |
|                 |                 |                 | community,      |
|                 |                 |                 | global climate  |
|                 |                 |                 | model that      |
|                 |                 |                 | provides        |
|                 |                 |                 | s               |
|                 |                 |                 | tate-of-the-art |
|                 |                 |                 | computer        |
|                 |                 |                 | simulations of  |
|                 |                 |                 | the Earth’s     |
|                 |                 |                 | past, present,  |
|                 |                 |                 | and future      |
|                 |                 |                 | climate states. |
+-----------------+-----------------+-----------------+-----------------+
| CDO             | 1.9.8           | |cpu|           | CDO is a        |
|                 |                 |                 | collection of   |
|                 |                 |                 | command line    |
|                 |                 |                 | Operators to    |
|                 |                 |                 | manipulate and  |
|                 |                 |                 | analyse Climate |
|                 |                 |                 | and NWP model   |
|                 |                 |                 | Data.           |
+-----------------+-----------------+-----------------+-----------------+
| CP2K            | 6.1             | |cpu|           | A freely        |
|                 |                 |                 | available       |
|                 |                 |                 | program to      |
|                 |                 |                 | perform         |
|                 |                 |                 | atomistic and   |
|                 |                 |                 | molecular       |
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
| Cufflinks       | 2.2.1           | |cpu|           | Cufflinks       |
|                 |                 |                 | assembles       |
|                 |                 |                 | transcripts,    |
|                 |                 |                 | estimates their |
|                 |                 |                 | abundances, and |
|                 |                 |                 | tests for       |
|                 |                 |                 | differential    |
|                 |                 |                 | expression and  |
|                 |                 |                 | regulation in   |
|                 |                 |                 | RNA-Seq         |
|                 |                 |                 | samples.        |
+-----------------+-----------------+-----------------+-----------------+
| FastQC          | 0.11.7          | |cpu|           | FastQC aims to  |
|                 |                 |                 | provide a       |
|                 |                 |                 | simple way to   |
|                 |                 |                 | do some quality |
|                 |                 |                 | control checks  |
|                 |                 |                 | on raw sequence |
|                 |                 |                 | data coming     |
|                 |                 |                 | from high       |
|                 |                 |                 | throughput      |
|                 |                 |                 | sequencing      |
|                 |                 |                 | pipelines.      |
+-----------------+-----------------+-----------------+-----------------+
| GATK            | 3.8             | |cpu|           | The GATK is the |
|                 |                 |                 | industry        |
|                 |                 |                 | standard for    |
|                 |                 |                 | identifying     |
|                 |                 |                 | SNPs and indels |
|                 |                 |                 | in germline DNA |
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
| GMAP-GSNAP      | 2019-05-12      | |cpu|           | GMAP is a tools |
|                 |                 |                 | for rapidly and |
|                 |                 |                 | accurately      |
|                 |                 |                 | mapping and     |
|                 |                 |                 | aligning cDNA   |
|                 |                 |                 | sequences to    |
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
| GraphMap        | 0.3.0           | |cpu|           | A highly        |
|                 |                 |                 | sensitive and   |
|                 |                 |                 | accurate mapper |
|                 |                 |                 | for long,       |
|                 |                 |                 | error-prone     |
|                 |                 |                 | reads.          |
+-----------------+-----------------+-----------------+-----------------+
| `Gr             | 2019.2, 2019.4  | |cpu|           | GROMACS is a    |
| omacs <https:// |                 | |gpu|\ |arm|    | versatile       |
| docs.hpc.sjtu.e |                 | |singularity|   | package to      |
| du.cn/applicati |                 |                 | perform         |
| on/Gromacs/>`__ |                 |                 | molecular       |
|                 |                 |                 | dynamics,       |
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
| HISAT2          | 2.1.0           | |cpu|           | HISAT2 is a     |
|                 |                 |                 | fast and        |
|                 |                 |                 | sensitive       |
|                 |                 |                 | alignment       |
|                 |                 |                 | program for     |
|                 |                 |                 | mapping         |
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
| `               | 20190807        | |cpu|           | (Large-scale    |
| LAMMPS <https:/ |                 | |gpu|\ |arm|    | A               |
| /docs.hpc.sjtu. |                 | |singularity|   | tomic/Molecular |
| edu.cn/applicat |                 |                 | Massively       |
| ion/Lammps/>`__ |                 |                 | Parallel        |
|                 |                 |                 | Simulator) a    |
|                 |                 |                 | classical       |
|                 |                 |                 | molecular       |
|                 |                 |                 | dynamics code.  |
+-----------------+-----------------+-----------------+-----------------+
| LUMPY-SV        | 0.2.13          | |cpu|           | A general       |
|                 |                 |                 | probabilistic   |
|                 |                 |                 | framework for   |
|                 |                 |                 | structural      |
|                 |                 |                 | variant         |
|                 |                 |                 | discovery.      |
+-----------------+-----------------+-----------------+-----------------+
| MEGAHIT         | 1.1.4           | |cpu|           | MEGAHIT is an   |
|                 |                 |                 | ultra-fast and  |
|                 |                 |                 | m               |
|                 |                 |                 | emory-efficient |
|                 |                 |                 | NGS assembler.  |
|                 |                 |                 | It is optimized |
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
| METIS           | 5.1.0           | |cpu|           | METIS is a set  |
|                 |                 |                 | of serial       |
|                 |                 |                 | programs for    |
|                 |                 |                 | partitioning    |
|                 |                 |                 | graphs,         |
|                 |                 |                 | partitioning    |
|                 |                 |                 | finite element  |
|                 |                 |                 | meshes, and     |
|                 |                 |                 | producing fill  |
|                 |                 |                 | reducing        |
|                 |                 |                 | orderings for   |
|                 |                 |                 | sparse          |
|                 |                 |                 | matrices.       |
+-----------------+-----------------+-----------------+-----------------+
| MrBayes         | 3.2.7a          | |cpu|           | MrBayes is a    |
|                 |                 |                 | program for     |
|                 |                 |                 | Bayesian        |
|                 |                 |                 | inference and   |
|                 |                 |                 | model choice    |
|                 |                 |                 | across a wide   |
|                 |                 |                 | range of        |
|                 |                 |                 | phylogenetic    |
|                 |                 |                 | and             |
|                 |                 |                 | evolutionary    |
|                 |                 |                 | models.         |
+-----------------+-----------------+-----------------+-----------------+
| NCBI-RMBlastn   | 2.2.28          | |cpu|           | RMBlast is a    |
|                 |                 |                 | RepeatMasker    |
|                 |                 |                 | compatible      |
|                 |                 |                 | version of the  |
|                 |                 |                 | standard NCBI   |
|                 |                 |                 | BLAST suite.    |
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
| `Ne             | 4.4.1           | |cpu|           | Nektar++ is a   |
| ktar++ <https:/ |                 |                 | spectral/hp     |
| /docs.hpc.sjtu. |                 |                 | element         |
| edu.cn/applicat |                 |                 | framework       |
| ion/Nektar/>`__ |                 |                 | designed to     |
|                 |                 |                 | support the     |
|                 |                 |                 | construction of |
|                 |                 |                 | efficient       |
|                 |                 |                 | h               |
|                 |                 |                 | igh-performance |
|                 |                 |                 | scalable        |
|                 |                 |                 | solvers for a   |
|                 |                 |                 | wide range of   |
|                 |                 |                 | partial         |
|                 |                 |                 | differential    |
|                 |                 |                 | equations       |
|                 |                 |                 | (PDE).          |
+-----------------+-----------------+-----------------+-----------------+
| `               | 6.8.1           | |cpu|           | NWChem aims to  |
| nwChem <https:/ |                 |                 | provide its     |
| /docs.hpc.sjtu. |                 |                 | users with      |
| edu.cn/applicat |                 |                 | computational   |
| ion/nwchem/>`__ |                 |                 | chemistry tools |
|                 |                 |                 | that are        |
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
|                 |                 |                 | h               |
|                 |                 |                 | igh-performance |
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
| `               | 5.2.0           | |cpu|           | GNU Octave is   |
| Octave <https:/ |                 | |singularity|   | software        |
| /docs.hpc.sjtu. |                 |                 | featuring a     |
| edu.cn/applicat |                 |                 | high-level      |
| ion/Octave/>`__ |                 |                 | programming     |
|                 |                 |                 | language,       |
|                 |                 |                 | primarily       |
|                 |                 |                 | intended for    |
|                 |                 |                 | numerical       |
|                 |                 |                 | computations.   |
+-----------------+-----------------+-----------------+-----------------+
| `Open           | 7, 1712, 1812,  | |cpu|           | OpenFOAM is an  |
| Foam <https://d | 1912            | |singularity|   | open-source     |
| ocs.hpc.sjtu.ed |                 |                 | toolbox for     |
| u.cn/applicatio |                 |                 | computational   |
| n/OpenFoam/>`__ |                 |                 | fluid dynamics. |
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
|                 |                 |                 | e               |
|                 |                 |                 | lectromagnetism |
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
| Paraview        | 0.4.1           | |cpu|           | Paraview is a   |
|                 |                 |                 | data            |
|                 |                 |                 | visualisation   |
|                 |                 |                 | and analysis    |
|                 |                 |                 | package. Whilst |
|                 |                 |                 | ARCHER compute  |
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
| Picard          | 2.19.0          | |cpu|           | Picard is a set |
|                 |                 |                 | of command line |
|                 |                 |                 | tools for       |
|                 |                 |                 | manipulating    |
|                 |                 |                 | high-throughput |
|                 |                 |                 | sequencing      |
|                 |                 |                 | (HTS) data and  |
|                 |                 |                 | formats such as |
|                 |                 |                 | SAM/BAM/CRAM    |
|                 |                 |                 | and VCF.        |
+-----------------+-----------------+-----------------+-----------------+
| `Py             | 19.10           | |gpu|           | PyTorch is an   |
| torch <https:// |                 | |singularity|   | open source     |
| docs.hpc.sjtu.e |                 |                 | machine         |
| du.cn/applicati |                 |                 | learning        |
| on/Pytorch/>`__ |                 |                 | library based   |
|                 |                 |                 | on the Torch    |
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
| `Quant          | 6.4.1           | |cpu|           | Quantum         |
| um-Espresso <ht |                 |                 | Espresso is an  |
| tps://docs.hpc. |                 |                 | integrated      |
| sjtu.edu.cn/app |                 |                 | suite of        |
| lication/Quantu |                 |                 | Open-Source     |
| m-Espresso/>`__ |                 |                 | computer codes  |
|                 |                 |                 | for             |
|                 |                 |                 | elect           |
|                 |                 |                 | ronic-structure |
|                 |                 |                 | calculations    |
|                 |                 |                 | and materials   |
|                 |                 |                 | modeling at the |
|                 |                 |                 | nanoscale. It   |
|                 |                 |                 | is based on     |
|                 |                 |                 | den             |
|                 |                 |                 | sity-functional |
|                 |                 |                 | theory, plane   |
|                 |                 |                 | waves, and      |
|                 |                 |                 | ps              |
|                 |                 |                 | eudopotentials. |
+-----------------+-----------------+-----------------+-----------------+
| `R <ht          | 1.1.8, 3.6.2    | |cpu|           | R is a          |
| tps://docs.hpc. |                 |                 | programming     |
| sjtu.edu.cn/app |                 |                 | language and    |
| lication/R/>`__ |                 |                 | free software   |
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
| `               | 3.0.8           | |gpu|           | REgularised     |
| Relion <https:/ |                 |                 | LIkelihood      |
| /docs.hpc.sjtu. |                 |                 | OptimisatioN    |
| edu.cn/applicat |                 |                 | (RELION)        |
| ion/Relion/>`__ |                 |                 | employs an      |
|                 |                 |                 | empirical       |
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
| RNA-SeQC        | 1.1.8           | |cpu|           | RNA-SeQC is a   |
|                 |                 |                 | java program    |
|                 |                 |                 | which computes  |
|                 |                 |                 | a series of     |
|                 |                 |                 | quality control |
|                 |                 |                 | metrics for     |
|                 |                 |                 | RNA-seq data.   |
+-----------------+-----------------+-----------------+-----------------+
| Salmon          | 0.14.1          | |cpu|           | Salmon is a     |
|                 |                 |                 | tool for        |
|                 |                 |                 | wicked-fast     |
|                 |                 |                 | transcript      |
|                 |                 |                 | quantification  |
|                 |                 |                 | from RNA-seq    |
|                 |                 |                 | data.           |
+-----------------+-----------------+-----------------+-----------------+
| SAMtools        | 1.9             | |cpu|           | SAM Tools       |
|                 |                 |                 | provide various |
|                 |                 |                 | utilities for   |
|                 |                 |                 | manipulating    |
|                 |                 |                 | alignments in   |
|                 |                 |                 | the SAM format. |
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
| SRAtoolkit      | 2.9.6           | |cpu|           | The SRA Toolkit |
|                 |                 |                 | and SDK from    |
|                 |                 |                 | NCBI is a       |
|                 |                 |                 | collection of   |
|                 |                 |                 | tools and       |
|                 |                 |                 | libraries for   |
|                 |                 |                 | using data in   |
|                 |                 |                 | the INSDC       |
|                 |                 |                 | Sequence Read   |
|                 |                 |                 | Archives.       |
+-----------------+-----------------+-----------------+-----------------+
| STAR            | 2.7.0           | |cpu|           | Spliced         |
|                 |                 |                 | Transcripts     |
|                 |                 |                 | Alignment to a  |
|                 |                 |                 | Reference       |
|                 |                 |                 | (STAR) software |
|                 |                 |                 | is based on a   |
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
| `STAR-          |                 | |cpu|           | Much more than  |
| CCM+ <https://d |                 |                 | just a CFD      |
| ocs.hpc.sjtu.ed |                 |                 | solver,         |
| u.cn/applicatio |                 |                 | STAR-CCM+ is an |
| n/star-ccm/>`__ |                 |                 | entire          |
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
| StringTie       | 1.3.4d          | |cpu|           | StringTie is a  |
|                 |                 |                 | fast and highly |
|                 |                 |                 | efficient       |
|                 |                 |                 | assembler of    |
|                 |                 |                 | RNA-Seq         |
|                 |                 |                 | alignments into |
|                 |                 |                 | potential       |
|                 |                 |                 | transcripts.    |
+-----------------+-----------------+-----------------+-----------------+
| `TensorFl       | 2.0.0           | |gpu|           | TensorFlow is a |
| ow <https://doc |                 | |singularity|   | free and        |
| s.hpc.sjtu.edu. |                 |                 | open-source     |
| cn/application/ |                 |                 | software        |
| TensorFlow/>`__ |                 |                 | library for     |
|                 |                 |                 | dataflow and    |
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
| TopHat          | 2.1.2           | |cpu|           | TopHat is a     |
|                 |                 |                 | program that    |
|                 |                 |                 | aligns RNA-Seq  |
|                 |                 |                 | reads to a      |
|                 |                 |                 | genome in order |
|                 |                 |                 | to identify     |
|                 |                 |                 | exon-exon       |
|                 |                 |                 | splice          |
|                 |                 |                 | junctions.      |
+-----------------+-----------------+-----------------+-----------------+
| VarDictJava     | 1.5.1           | |cpu|           | VarDictJava is  |
|                 |                 |                 | a variant       |
|                 |                 |                 | discovery       |
|                 |                 |                 | program written |
|                 |                 |                 | in Java and     |
|                 |                 |                 | Perl.           |
+-----------------+-----------------+-----------------+-----------------+
| `VASP <https    |                 | |cpu| |gpu|     | A package for   |
| ://docs.hpc.sjt |                 |                 | ab initio,      |
| u.edu.cn/applic |                 |                 | quan            |
| ation/VASP/>`__ |                 |                 | tum-mechanical, |
|                 |                 |                 | molecular       |
|                 |                 |                 | dynamics        |
|                 |                 |                 | simulations.    |
+-----------------+-----------------+-----------------+-----------------+
| VSEARCH         | 2.4.3           | |cpu|           | VSEARCH stands  |
|                 |                 |                 | for vectorized  |
|                 |                 |                 | search, as the  |
|                 |                 |                 | tool takes      |
|                 |                 |                 | advantage of    |
|                 |                 |                 | parallelism in  |
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
| `VMD <http      | 1.9.4           | |cpu|           | VMD is a        |
| s://docs.hpc.sj |                 | |singularity|   | molecular       |
| tu.edu.cn/appli |                 |                 | visualization   |
| cation/VMD/>`__ |                 |                 | program for     |
|                 |                 |                 | displaying,     |
|                 |                 |                 | animating, and  |
|                 |                 |                 | analyzing large |
|                 |                 |                 | biomolecular    |
|                 |                 |                 | systems using   |
|                 |                 |                 | 3-D graphics    |
|                 |                 |                 | and built-in    |
|                 |                 |                 | scripting.      |
+-----------------+-----------------+-----------------+-----------------+

.. |cpu| image:: https://img.shields.io/badge/-cpu-blue
.. |gpu| image:: https://img.shields.io/badge/-gpu-green
.. |arm| image:: https://img.shields.io/badge/-arm-yellow
.. |singularity| image:: https://img.shields.io/badge/-singularity-blueviolet
