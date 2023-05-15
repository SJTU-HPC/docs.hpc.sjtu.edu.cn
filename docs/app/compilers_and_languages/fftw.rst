.. _FFTW:

FFTW
==========

简介
----

FFTW是由麻省理工学院计算机科学实验室超级计算技术组开发的一套离散傅立叶变换(DFT)的计算库，开源、高效和标准C语言编写的代码使其得到了非常广泛的应用。Intel的数学库和Scilib(类似于Matlab的科学计算软件)都使用FFTW做FFT计算。FFTW是计算离散Fourier变换(DFT)的快速C程序的一个完整集合。




FFTW使用说明
-----------------------------

思源一号上的FFTW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 创建fftwtest目录并进入该目录：

.. code::
        
    mkdir fftwtest
    cd fftwtest

2. 在该目录下编写如下myfftw.c文件：

.. code::
        
  #include <stdio.h>
  #include "fftw3.h"

  int main()
  {
    fftw_complex *in, *out;
    fftw_plan p;
    int N= 8;
    int i;
    int j;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    for( i=0; i < N; i++)
    {
        in[i][0] = 1.0;
        in[i][1] = 0.0;
        printf("%6.2f ",in[i][0]);
    }
    printf("\n");

    p=fftw_plan_dft_1d(N,in,out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p); /* repeat as needed*/

    for(j = 0;j < N;j++)
    {
        printf("%6.2f ",out[j][0]);
    }
    printf("\n");

    fftw_destroy_plan(p);
    fftw_free(in); 
    fftw_free(out);
    return 0;
  }



3. 在该目录下编写如下fftwtest.slurm脚本：

.. code::

  #!/bin/bash

  #SBATCH --job-name=fftwtest    
  #SBATCH --partition=64c512g     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load gcc
  module load fftw/3.3.10-gcc-8.3.1-openmpi


  gcc myfftw.c -o myfftw -lfftw3 -lm

  ./myfftw


4. 使用如下命令提交作业：

.. code::

  sbatch fftwtest.slurm


5. 作业完成后在.out文件中可看到如下结果：

.. code::

    1.00   1.00   1.00   1.00   1.00   1.00   1.00   1.00 
    8.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00

pi2.0上的FFTW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；

2. 此步骤和上文完全相同；

3. 在该目录下编写如下fftwtest.slurm脚本：

.. code::

  #!/bin/bash

  #SBATCH --job-name=fftwtest    
  #SBATCH --partition=small     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load gcc
  module load fftw/3.3.8-gcc-4.8.5

  gcc myfftw.c -o myfftw -lfftw3 -lm

  ./myfftw


4. 使用如下命令提交作业：

.. code::

  sbatch fftwtest.slurm


5. 作业完成后在.out文件中可看到如下结果：

.. code::

    1.00   1.00   1.00   1.00   1.00   1.00   1.00   1.00 
    8.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00

参考资料
----------------

-  `FFTW官网 <http://www.fftw.org/>`__


