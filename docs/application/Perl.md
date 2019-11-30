# <center>PERL</centet>

-----------------

## 加载预安装的perl

Pi2.0 系统中已经预装nektar-4.4.1(GNU+cpu 版本)，可以用以下命令加载: 
```
$ cd 
$ module load perl/5.26.2-gcc-8.3.0
```
如有必要，请删除现有的CPAN模块。
```
$ rm -rf ~/.perl ~/.cpan
```
为避免重复调用，将以下命令添加到.bashrc中：
```
module load perl/5.26.2-gcc-8.3.0
```
注销后登录以使设置生效
##在CPAN中安装Perl模块
###Synosys
```
$ module load perl/5.26.2-gcc-8.3.0
$ cpan
cpan> install MODULE_NAME
```
###示例
```
$ module load perl/5.26.2-gcc-8.3.0
$ cpan
cpan> install XML::LibXML
...
cpan> install Getopt::Std
...
cpan> install Encode
```
##Perl的SLURM作业示例
用法：sbatch job.slurm
```
#!/bin/bash

#SBATCH -J Perl
#SBATCH -p cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@EMAIL.COM
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1

module load perl/5.26.2-gcc-8.3.0

perl hello.pl
```
##参考文献
*[http://www.perlmonks.org/?node_id=630026]
*[https://wiki.hpcc.msu.edu/display/Bioinfo/Installing+Local+Perl+Modules+with+CPAN]


