ReFrame
=======

- `基准程序`_

- `分子动力学`_

.. _基准程序:

基准程序
^^^^^^^^^^

一. STREAM
-------------

.. code:: bash

   STREAM version $Revision: 5.10
   Array size = 33554432
   Memory per array = 256.0 MiB (= 0.2 GiB)
   Total memory required = 768.0 MiB (= 0.8 GiB)
   Number of Threads counted = 4

运行STREAM的方式
>>>>>>>>>>>>>>>>>

思源一号

.. code:: bash

   salloc -p 64c512g -N 1 --exclusive
   ssh node***
   module load reframe
   stream

π2.0

.. code:: bash

   salloc -p cpu -N 1 --exclusive
   ssh cas***
   module load reframe
   stream

STREAM的运行结果
>>>>>>>>>>>>>>>>>

思源一号

.. code:: bash

   ------------------------------------------------------------------------------
   StreamTest
   - generic:default
      - gnu
         * num_tasks: 1
         * Copy: 71986.2 MB/s
         * Scale: 43348.9 MB/s
         * Add: 45779.5 MB/s
         * Triad: 45256.8 MB/s
   ------------------------------------------------------------------------------

π2.0

.. code:: bash

   ------------------------------------------------------------------------------
   StreamTest
   - generic:default
      - gnu
         * num_tasks: 1
         * Copy: 39972.3 MB/s
         * Scale: 46063.2 MB/s
         * Add: 48718.5 MB/s
         * Triad: 48544.8 MB/s
   ------------------------------------------------------------------------------

.. _分子动力学:

分子动力学
^^^^^^^^^^^

一. Quantum Espresso (QE)
---------------------------

.. code:: bash

   Program PWSCF v.6.7MaX
   Parallel version (MPI)
   Compiled by Intel 2021.4.0

运行QE的方式
>>>>>>>>>>>>>>>>>

思源一号

.. code:: bash

   salloc -p 64c512g -N 1 --exclusive
   ssh node***
   module load reframe
   qe
 
π2.0

.. code:: bash

   salloc -p cpu -N 1 --exclusive
   ssh cas***
   module load reframe
   qe

QE的运行结果
>>>>>>>>>>>>>>>

思源一号 ``64cores``

.. code:: bash

   ------------------------------------------------------------------------------
   QETest
   - generic:default
      - gnu
         * num_tasks: 1
         * extract_copy_perf: 5m21.33s CPU
   ------------------------------------------------------------------------------

π2.0 ``40cores``

.. code:: bash

   ------------------------------------------------------------------------------
   QETest
   - generic:default
      - gnu
         * num_tasks: 1
         * extract_copy_perf: 9m34.58s CPU
   ------------------------------------------------------------------------------

二. Amber22
-------------

运行Amber22的方式
>>>>>>>>>>>>>>>>>>>

思源一号

.. code:: bash

   salloc -p a100 -N 1 -n 6  --gres=gpu:1  --exclusive
   ssh gpu***
   module load reframe
   amber_gpu

π2.0

.. code:: bash

   salloc -p dgx2 -N 1 -n 16 --gres=gpu:1 --exclusive
   ssh vol***
   module load reframe
   amber_gpu

Amber的运行结果
>>>>>>>>>>>>>>>>

思源一号 ``16core+1GPU(a100)``

.. code:: bash

   ------------------------------------------------------------------------------
      wait
   ------------------------------------------------------------------------------

π2.0 ``6cores+1GPU(p100)``

.. code:: bash

   ------------------------------------------------------------------------------
   AMBERGPUTest
   - generic:default
      - gnu
         * num_tasks: 1
         * consumed_time: 60.65 s
   ------------------------------------------------------------------------------
