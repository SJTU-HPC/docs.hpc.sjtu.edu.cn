# <center>STRique</center>

---------

STRique is a python package to analyze repeat expansion and methylation states of short tandem repeats (STR) in Oxford Nanopore Technology (ONT) long read sequencing data.

## 在 Pi 上安装 STRique

首先申请计算节点用来编译，并输入以下指令：

```bash
$ srun -p small -n 4 --pty /bin/bash
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n test6_1 python=3.6
$ source activate test6_1
$ git clone --recursive https://github.com/giesselmann/STRique
$ cd STRique
$ pip install -r requirements.txt
$ python setup.py install
```

安装完成后，按下方操作使用：

```bash
$ cd ~/STRique/data
$ python ../scripts/STRique.py index --recursive c9orf72.fast5 > c9orf72.fofn
$ cat c9orf72.sam | python ../scripts/STRique.py count c9orf72.fofn ../models/r9_4_450bps.model ../configs/repeat_config.tsv > c9orf72.hg19.strique.tsv
$ cat c9orf72.hg19.strique.tsv | python ../scripts/STRique.py plot c9orf72.fofn --output c9orf72.pdf --format pdf
```

## 参考文献

- [STRique](https://strique.readthedocs.io/en/latest/installation/src/)
