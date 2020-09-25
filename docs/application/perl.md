# <center>PERL</centet>

-----------------

## 使用Miniconda 3环境安装perl

加载Miniconda 3

```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
```

创建conda环境

```bash
$ conda create --name PERL
```

激活R环境

```bash
$ source activate PERL
```

如有必要，请删除现有的CPAN模块和.bashrc中perl相关设置
```bash
$ rm -rf ~/.perl ~/.cpan
```

!!!tip
	上述操作会删除现在已有模块和perl环境配置信息，请谨慎操作。
	

在当前环境下安装perl并设置相关环境变量
```bash
$ conda install perl
...
$ cpan
...
$ Would you like to configure as much as possible automatically? [yes] yes
...
$ What approach do you want?  (Choose 'local::lib', 'sudo' or 'manual')
 [local::lib] 
...
```

## 拓展模块下载示例
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ source activate PERL
$ cpan
cpan> install XML::LibXML
...
cpan> install Getopt::Std
...
cpan> install Encode
```

手动拓展模块下载示例(不推荐)
```bash
$ cd /YOUR/PACKAGE/PATH
$ tar xvzf Net-Server-0.97.tar.gz
$ cd Net-Server-0.97
$ perl Makefile.PL
$ make test
```

## 查看已下载的perl拓展模块
```bash
#方法一：
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ source activate PERL
$ instmodsh
> l
Installed modules are:
   ...
   Perl

#方法二：
$ perldoc perllocal
...
```


## Perl的SLURM作业示例
用法：sbatch job.slurm
```bash
#!/bin/bash

#SBATCH -J Perl
#SBATCH -p small
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@EMAIL.COM
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1

module purge
module load miniconda3/4.7.12.1-gcc-4.8.5
source activate PERL

perl hello.pl
```
## 参考文献
- [Set Install path in CPAN](http://www.perlmonks.org/?node_id=630026)
- [perl模块安装大全](http://www.bio-info-trainee.com/2451.html)
