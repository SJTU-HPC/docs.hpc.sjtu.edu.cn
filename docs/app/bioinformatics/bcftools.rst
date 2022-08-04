.. _Bcftools:

BCFtools
===============

简介
----------------
BCFtools主要是用来操作vcf和BCF文件的工具合集，包含有许多命令。用户可使用集群上已经部署的版本，也可自行编译。

.. _CPU版本BCFtools:

CPU 版本 BCFtools 源码安装方法
--------------------------------------------------
.. code:: bash

    srun -p 64c512g -n 4 --pty /bin/bash
    mkdir ${HOME}/01.application/09.bcftools && cd ${HOME}/01.application/09.bcftools
    wget https://github.com/samtools/bcftools/releases/download/1.15.1/bcftools-1.15.1.tar.bz2
    tar -jxvf  bcftools-1.15.1.tar.bz2
    cd bcftools-1.15.1/
    ./configure --prefix=${HOME}/01.application/09.bcftools/
    make
    make install
    export PATH=${HOME}/01.application/09.bcftools/bin/:$PATH


π2 版本BCFtools
--------------------------------------------

示例脚本如下(bcftools.slurm):    

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=test       
   #SBATCH --partition=cpu       
   #SBATCH -N 1          
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load bcftools/1.9-gcc-9.2.0
   bcftools query -f '%CHROM %ID %POS %REF %ALT [ %TGT]\n' test.vcf.gz -o test.extract.txt
   bcftools stats test.vcf > test.vcf.stats
                         

使用如下指令提交：

.. code:: bash

   $ sbatch bcftools.slurm


ARM 版本BCFtools
----------------

示例脚本如下(bcftools.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1          
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module use /lustre/share/spack/modules/kunpeng920/linux-centos7-aarch64
   module load bcftools/1.10.2-gcc-9.3.0-openblas
   bcftools query -f '%CHROM %ID %POS %REF %ALT [ %TGT]\n' test.vcf.gz -o test.extract.txt
   bcftools stats test.vcf > test.vcf.stats
   

使用如下指令提交：

.. code:: bash

   $ sbatch bcftools.slurm


