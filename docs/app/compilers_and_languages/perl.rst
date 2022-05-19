.. _perl:

PERL
====

使用Miniconda 3环境安装perl
---------------------------

加载Miniconda 3

.. code:: bash

   $ module load miniconda3

创建conda环境

.. code:: bash

   $ conda create --name PERL

激活R环境

.. code:: bash

   $ source activate PERL

如有必要，请删除现有的CPAN模块和.bashrc中perl相关设置

.. code:: bash

   $ rm -rf ~/.perl ~/.cpan

上述操作会删除现在已有模块和perl环境配置信息，请谨慎操作。

在当前环境下安装perl并设置相关环境变量

.. code:: bash

   $ conda install perl
   ...
   $ cpan
   ...
   $ Would you like to configure as much as possible automatically? [yes] yes
   ...
   $ What approach do you want?  (Choose 'local::lib', 'sudo' or 'manual')
    [local::lib] 
   ...

拓展模块cpan安装示例
--------------------

.. code:: bash

   $ module load miniconda3
   $ source activate PERL
   $ cpan
   cpan> install XML::LibXML
   ...
   cpan> install Getopt::Std
   ...
   cpan> install Encode

手动拓展模块下载示例(不推荐)
----------------------------

.. code:: bash

   $ cd /YOUR/PACKAGE/PATH
   $ tar xvzf Net-Server-0.97.tar.gz
   $ cd Net-Server-0.97
   $ perl Makefile.PL
   $ make test

拓展模块conda安装示例(推荐)
----------------------------

.. code:: bash

   $ source activate PERL # 进入创建的conda环境
   $ conda install -c bioconda perl-pdf-api2
   $ perl -MPDF::API2 -e '' # 无报错说明模块安装成功

查看已下载的perl拓展模块
------------------------

.. code:: bash

   #方法一：
   $ module load miniconda3
   $ source activate PERL
   $ instmodsh
   > l
   Installed modules are:
      ...
      Perl

   #方法二：
   $ perldoc perllocal
   ...

Perl的SLURM作业示例
-------------------

用法：sbatch job.slurm

.. code:: bash

   #!/bin/bash

   #SBATCH -J Perl
   #SBATCH -p small
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 1

   module load miniconda3
   source activate PERL

   perl hello.pl

参考资料
--------

-  `Set Install path in
   CPAN <http://www.perlmonks.org/?node_id=630026/>`__
-  `perl模块安装大全 <http://www.bio-info-trainee.com/2451.html/>`__
