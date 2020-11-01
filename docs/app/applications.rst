Pi上的软件
==========

本文档介绍 Pi 上的软件。商业软件需用户自行获取版权并安装。 |cpu| |gpu|
|arm| |singularity| 标签表明软件有 cpu, gpu, arm 和 singularity 版本

.. _pi-上的软件-1:

Pi上的软件
----------

+-----------------+-----------------+-----------------+-----------------+
| Name            | Version         | Distribution    | Introduction    |
+=================+=================+=================+=================+
| `ABINIT <https: | 8.10.3          | |cpu|           | ABINIT is a     |
| //docs.hpc.sjtu |                 |                 | package whose   |
| .edu.cn/applica |                 |                 | main program    |
| tion/abinit/>`_ |                 |                 | allows one to   |
| _               |                 |                 | find the total  |
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
|                 |                 |                 | BGZF-compressed |
|                 |                 |                 | .               |
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
|                 |                 |                 | memory-efficien |
|                 |                 |                 | t               |
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
| `CESM <https:// |                 | |cpu|           | Community Earth |
| docs.hpc.sjtu.e |                 |                 | System Model,   |
| du.cn/applicati |                 |                 | or CESM, is a   |
| on/CESM/>`__    |                 |                 | fully-coupled,  |
|                 |                 |                 | community,      |
|                 |                 |                 | global climate  |
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
| `Gromacs <https | 2019.2, 2019.4  | |cpu|           | GROMACS is a    |
| ://docs.hpc.sjt |                 | |gpu|\ |arm|    | versatile       |
| u.edu.cn/applic |                 | |singularity|   | package to      |
| ation/Gromacs/> |                 |                 | perform         |
| `__             |                 |                 | molecular       |
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
| `LAMMPS <https: | 20190807        | |cpu|           | (Large-scale    |
| //docs.hpc.sjtu |                 | |gpu|\ |arm|    | Atomic/Molecula |
| .edu.cn/applica |                 | |singularity|   | r               |
| tion/Lammps/>`_ |                 |                 | Massively       |
| _               |                 |                 | Parallel        |
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
|                 |                 |                 | memory-efficien |
|                 |                 |                 | t               |
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
| `Nektar++ <http | 4.4.1           | |cpu|           | Nektar++ is a   |
| s://docs.hpc.sj |                 |                 | spectral/hp     |
| tu.edu.cn/appli |                 |                 | element         |
| cation/Nektar/> |                 |                 | framework       |
| `__             |                 |                 | designed to     |
|                 |                 |                 | support the     |
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
| `nwChem <https: | 6.8.1           | |cpu|           | NWChem aims to  |
| //docs.hpc.sjtu |                 |                 | provide its     |
| .edu.cn/applica |                 |                 | users with      |
| tion/nwchem/>`_ |                 |                 | computational   |
| _               |                 |                 | chemistry tools |
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
| `Octave <https: | 5.2.0           | |cpu|           | GNU Octave is   |
| //docs.hpc.sjtu |                 | |singularity|   | software        |
| .edu.cn/applica |                 |                 | featuring a     |
| tion/Octave/>`_ |                 |                 | high-level      |
| _               |                 |                 | programming     |
|                 |                 |                 | language,       |
|                 |                 |                 | primarily       |
|                 |                 |                 | intended for    |
|                 |                 |                 | numerical       |
|                 |                 |                 | computations.   |
+-----------------+-----------------+-----------------+-----------------+
| `OpenFoam <http | 7, 1712, 1812,  | |cpu|           | OpenFOAM is an  |
| s://docs.hpc.sj | 1912            | |singularity|   | open-source     |
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
| `Pytorch <https | 19.10           | |gpu|           | PyTorch is an   |
| ://docs.hpc.sjt |                 | |singularity|   | open source     |
| u.edu.cn/applic |                 |                 | machine         |
| ation/Pytorch/> |                 |                 | learning        |
| `__             |                 |                 | library based   |
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
| `Quantum-Espres | 6.4.1           | |cpu|           | Quantum         |
| so <https://doc |                 |                 | Espresso is an  |
| s.hpc.sjtu.edu. |                 |                 | integrated      |
| cn/application/ |                 |                 | suite of        |
| Quantum-Espress |                 |                 | Open-Source     |
| o/>`__          |                 |                 | computer codes  |
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
| `Relion <https: | 3.0.8           | |gpu|           | REgularised     |
| //docs.hpc.sjtu |                 |                 | LIkelihood      |
| .edu.cn/applica |                 |                 | OptimisatioN    |
| tion/Relion/>`_ |                 |                 | (RELION)        |
| _               |                 |                 | employs an      |
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
| StringTie       | 1.3.4d          | |cpu|           | StringTie is a  |
|                 |                 |                 | fast and highly |
|                 |                 |                 | efficient       |
|                 |                 |                 | assembler of    |
|                 |                 |                 | RNA-Seq         |
|                 |                 |                 | alignments into |
|                 |                 |                 | potential       |
|                 |                 |                 | transcripts.    |
+-----------------+-----------------+-----------------+-----------------+
| `TensorFlow <ht | 2.0.0           | |gpu|           | TensorFlow is a |
| tps://docs.hpc. |                 | |singularity|   | free and        |
| sjtu.edu.cn/app |                 |                 | open-source     |
| lication/Tensor |                 |                 | software        |
| Flow/>`__       |                 |                 | library for     |
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
| `VASP <https:// |                 | |cpu| |gpu|     | A package for   |
| docs.hpc.sjtu.e |                 |                 | ab initio,      |
| du.cn/applicati |                 |                 | quantum-mechani |
| on/VASP/>`__    |                 |                 | cal,            |
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
| `VMD <https://d | 1.9.4           | |cpu|           | VMD is a        |
| ocs.hpc.sjtu.ed |                 | |singularity|   | molecular       |
| u.cn/applicatio |                 |                 | visualization   |
| n/VMD/>`__      |                 |                 | program for     |
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

