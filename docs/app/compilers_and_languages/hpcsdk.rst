NVIDIA HPC SDK
==================

NVIDIA HPC SDK是适用于GPU平台的高性能计算编译器，库和工具套件。
NVIDIA HPC SDK包括经过优化的编译器，库和软件工具，这些工具对于最大化开发人员的工作效率以及HPC应用程序的性能和可移植性至关重要。

PGI编译器和工具链已迁移至NVIDIA HPC SDK。

本文档向您展示如何使用NVIDIA HPC SDK，包含程序示例，编译，作业脚本示例。

可用版本
---------------------

+----------+--------------------------------------------------+
| 版本     | 加载方式                                         |
+==========+==================================================+        
| 21.9     | module load nvhpc/21.9-gcc-11.2.0       思源一号 |
+----------+--------------------------------------------------+
| 21.11    | module load nvhpc/20.11-gcc-4.8.5       闵行超算 |
+----------+--------------------------------------------------+


程序示例CUDA_Fortran
--------------------------------

思源超算加载NVIDIA HPC SDK环境。

.. code:: bash

   $ module load nvhpc/21.9-gcc-11.2.0

闵行超算加载NVIDIA HPC SDK环境。

.. code:: bash

   $ module load nvhpc/20.11-gcc-4.8.5


编辑 ``deviceQuery.cuf`` 文件，内容如下：

.. code:: fortran

    program deviceQuery
        use cudafor
        implicit none

        type (cudaDeviceProp) :: prop
        integer :: nDevices=0, i, ierr

        ! Number of CUDA-capable devices

        ierr = cudaGetDeviceCount(nDevices)

        if (nDevices == 0) then
            write(*,"(/,'No CUDA devices found',/)")
            stop
        else if (nDevices == 1) then
            write(*,"(/,'One CUDA device found',/)")
        else
            write(*,"(/,i0,' CUDA devices found',/)") nDevices
        end if

        ! Loop over devices

        do i = 0, nDevices-1

            write(*,"('Device Number: ',i0)") i

            ierr = cudaGetDeviceProperties(prop, i)
            if (ierr .eq. 0) then
                write(*,"('  GetDeviceProperties for device ',i0,': Passed')") i
            else
                write(*,"('  GetDeviceProperties for device ',i0,': Failed')") i
            endif

            ! General device info

            write(*,"('  Device Name: ',a)") trim(prop%name)
            write(*,"('  Compute Capability: ',i0,'.',i0)") &
                prop%major, prop%minor
            write(*,"('  Number of Multiprocessors: ',i0)") &
                prop%multiProcessorCount
            write(*,"('  Max Threads per Multiprocessor: ',i0)") &
                prop%maxThreadsPerMultiprocessor
            write(*,"('  Global Memory (GB): ',f9.3,/)") &
                prop%totalGlobalMem/1024.0**3

            ! Execution Configuration

            write(*,"('  Execution Configuration Limits')")
            write(*,"('    Max Grid Dims: ',2(i0,' x '),i0)") &
                prop%maxGridSize
            write(*,"('    Max Block Dims: ',2(i0,' x '),i0)") &
                prop%maxThreadsDim
            write(*,"('    Max Threads per Block: ',i0,/)") &
                prop%maxThreadsPerBlock

        enddo

    end program deviceQuery

程序示例：deviceQuery
-------------------------

使用nvfortran进行编译。

.. code:: bash

   $ nvfortran  -O2 -o deviceQuery.out deviceQuery.cuf



A100队列提交作业脚本示例
-----------------------------

这是一个名为\ ``a100.slurm``\ 的 **单机单卡**
作业脚本，该脚本向a100队列申请1块GPU，并在作业完成时通知。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=a100_test
   #SBATCH --partition=a100
   #SBATCH --gres=gpu:1
   #SBATCH -n 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=cublas.out
   #SBATCH --error=cublas.err

   module load nvhpc/21.9-gcc-11.2.0

   ./deviceQuery.out

用以下方式提交作业：

.. code:: bash

   $ sbatch a100.slurm



DGX2队列提交作业脚本示例
-----------------------------

这是一个名为\ ``dgx.slurm``\ 的 **单机单卡**
作业脚本，该脚本向dgx2队列申请1块GPU，并在作业完成时通知。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=dgx2_test
   #SBATCH --partition=dgx2
   #SBATCH --gres=gpu:1
   #SBATCH -n 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=cublas.out
   #SBATCH --error=cublas.err

   module load nvhpc/20.11-gcc-4.8.5

   ./deviceQuery.out

用以下方式提交作业：

.. code:: bash

   $ sbatch dgx.slurm

参考资料
--------

-  `NVIDIA HPC SDK Version 20.11 Documentation <https://docs.nvidia.com/hpc-sdk/index.html>`__
   https://docs.nvidia.com/hpc-sdk/index.html
