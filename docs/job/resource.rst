查看作业资源信息
================

当作业对CPU和内存的要求较高时，了解运行作业的CPU和内存的使用信息，能够保证作业顺利运行。

内存
----

内存分配策略
~~~~~~~~~~~~

+--------+--------------------------------------------+
| 集群   | 存储分配策略                               |
+========+============================================+
| π2.0   | 单节点配置为40核，180G内存；每核配比4G内存 |
+--------+--------------------------------------------+

可使用 ``seff jobid`` 命令查看单核所能使用的内存空间

.. code:: bash

   [hpc@login2 data]$ seff 9709905
   Job ID: 9709905
   Cluster: sjtupi
   User/Group: hpchgc/hpchgc
   State: RUNNING
   Nodes: 1
   Cores per node: 40
   CPU Utilized: 00:00:00
   CPU Efficiency: 0.00% of 02:22:40 core-walltime
   Job Wall-clock time: 00:03:34
   Memory Utilized: 0.00 MB (estimated maximum)
   Memory Efficiency: 0.00% of 160.00 GB (4.00 GB/core)              //(4.00 GB/core)
   WARNING: Efficiency statistics may be misleading for RUNNING jobs.

作业运行中的内存占用
~~~~~~~~~~~~~~~~~~~~

当提交作业后，使用 ``squeue`` 命令查看作业使用的节点

.. code:: bash

   [hpc@login2 test]$ squeue 
                JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              9709875       cpu  40cores      hpc  R       0:02      1 cas478

然后进入相关节点

.. code:: bash

   ssh cas478

可根据用户名查看作业占用的存储空间

.. code:: bash

    ps -u$USER -o %cpu,rss,args

示例如下： ``ps -uhpc -o %cpu,rss,args``

.. code:: bash

   %CPU    RSS COMMAND
   98.5 633512 pw.x -i ausurf.in
   98.5 652828 pw.x -i ausurf.in
   98.6 654312 pw.x -i ausurf.in
   98.6 652196 pw.x -i ausurf.in

``RSS`` 表示单核所占用的存储空间，单位为KB，上述分析可得单核上运行作业占用的存储空间大约为650MB,40核的内存利用率大约为： ``(0.65G*40)/160G: 16%``。

如果需要动态监测存储资源的使用，可进入计算节点后，输入top命令

.. code:: bash
   
      PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
   428410 hpc       20   0 5989.9m 662.5m 168.6m R 100.0  0.3   0:22.67 pw.x
   428419 hpc       20   0 5987.0m 658.9m 163.7m R 100.0  0.3   0:22.61 pw.x
   428421 hpc       20   0 5984.6m 677.8m 180.1m R 100.0  0.4   0:22.66 pw.x
   428433 hpc       20   0 6002.8m 661.7m 165.3m R 100.0  0.3   0:22.68 pw.x
   428436 hpc       20   0 5986.0m 659.0m 165.4m R 100.0  0.3   0:22.66 pw.x

上述数据中的RES列数据表示运行作业所占用的存储资源，单核大约占用650m。

作业运行结束后内存利用分析情况
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``seff jobid`` 命令

.. code:: bash

   [hpc@login2 data]$ seff 9709905
   Job ID: 9709905
   Cluster: sjtupi
   User/Group: hpchgc/hpchgc
   State: COMPLETED (exit code 0)
   Nodes: 1
   Cores per node: 40
   CPU Utilized: 06:27:20
   CPU Efficiency: 99.15% of 06:30:40 core-walltime
   Job Wall-clock time: 00:09:46
   Memory Utilized: 23.33 GB
   Memory Efficiency: 14.58% of 160.00 GB

GPU INFO
---------

下面介绍如何通过JOB ID获取GPU信息以及如何获取CUDA_VISIBLE_DEVICES变量并在程序中利用该变量

NVIDIA显卡的UUID（Universally Unique Identifier，通用唯一标识符）和BUS ID（总线标识符）是两个重要的标识符。

UUID（Universally Unique
Identifier）：UUID是一个128位的唯一标识符，用于在系统中识别每个独立的NVIDIA显卡设备。每个显卡设备都有一个唯一的UUID，可以通过调用相关命令或API来获取它。UUID在不同系统和环境中是持久的，即使重新启动系统或重新插拔显卡，UUID也不会改变。

BUS ID（总线标识符）：BUS ID是用于标识系统中不同物理或逻辑总线上的NVIDIA显卡设备的标识符。BUS ID提供了关于显卡设备如何连接到系统总线的信息，如PCI总线等。BUS ID通常采用“domain:bus:device.function”的形式表示，其中domain表示域，bus表示总线编号，device表示设备编号，function表示设备的功能编号。BUS ID主要用于管理和配置显卡设备，以及确定显卡在系统中的位置。

通过JOB ID获取GPU信息 
------------------------
下面介绍如何通过jobid查询UUID及BUSID信息：

脚本文件
~~~~~~~~

将下面脚本保存至\ ``getGPUInfo``\ 的文本中

.. code:: bash

   #!/bin/bash
   JOBID=$1
   GPUNUMS=`scontrol show jobs ${JOBID}|grep "gres:gpu" |awk -F ':' '{print $3}'`
   GPUHOST=`scontrol show jobs ${JOBID}|grep "NodeList="|sed -n '2p'|awk -F '=' '{print $2}'`
   echo JobID: ${JOBID}
   echo NodeHost: ${GPUHOST}
   echo GPUNums: ${GPUNUMS}
   echo Information of allocate GPUs:
   ssh ${GPUHOST} "echo UUID: && nvidia-smi -L"
   ssh ${GPUHOST} "echo BUS_ID: && nvidia-smi -q|grep 'Bus Id'|sed -e 's/^.*: //' -e 's/ $//'"
   
执行脚本加JOB ID
~~~~~~~~~~~~~~~~

针对正在运行的作业，执行脚本加作业ID可获取正在运行作业的节点中的GPU信息

.. code:: bash

   $ chmod +x getGPUInfo
   $ ./getGPUInfo 27180318
   JobID: 27180318
   NodeHost: gpu09
   GPUNums: 4
   Information of allocate GPUs:
   UUID:
   GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-5cd88acf-5391-8562-cd34-b543319224b4)
   GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-7bc1435d-37b5-d4b8-6ac1-df72927a54e0)
   GPU 2: NVIDIA A100-SXM4-40GB (UUID: GPU-4dedf87e-d147-83cc-c5bd-ec16324afa15)
   GPU 3: NVIDIA A100-SXM4-40GB (UUID: GPU-2bf8c199-1e4a-31cd-470b-4ba6329d9a60)
   BUS_ID:
   00000000:31:00.0
   00000000:4B:00.0
   00000000:CA:00.0
   00000000:E3:00.0

获取CUDA_VISIBLE_DEVICES
------------------------

CUDA_VISIBLE_DEVICES是一个环境变量，用于在使用CUDA编程时指定可见的GPU设备。它可以用来控制程序所使用的GPU设备的数量和顺序。

当用户申请有GPU卡的任务时，slurm系统会根据用户申请的GPU数量来设置CUDA_VISIBLE_DEVICES环境变量，只有相应编号的GPU设备会对程序可见，其他GPU设备则不可使用。

srun交互式作业
~~~~~~~~~~~~~~

在srun申请交互式作业后，可在shell中直接输出\ ``$CUDA_VISIBLE_DEVICES``\ 变量

.. code:: bash

   $  srun -n 8 -p dgx2 --gres=gpu:2 --pty /bin/bash
   srun: job 27182411 queued and waiting for resources
   srun: job 27182411 has been allocated resources
   $ echo $CUDA_VISIBLE_DEVICES
   0,1

作业脚本
~~~~~~~~

也可以在作业脚本最前面输出\ ``$CUDA_VISIBLE_DEVICES``\ 变量

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -n 8
   #SBATCH --gres=gpu:2
   #SBATCH -p dgx2

   echo $CUDA_VISIBLE_DEVICES
   ···

案例测试
~~~~~~~~

以下是一个简单的torch程序，展示了根据\ ``$CUDA_VISIBLE_DEVICES``\ 变量，设置程序使用的GPU

.. code:: bash

   $ cat pytorch_test.py
   import torch
   from torch import nn
   from torch.optim import Adam
   from torch.nn.parallel import DataParallel
   import os
   class DEMO_model(nn.Module):
           def __init__(self, in_size, out_size):
                   super().__init__()
                   self.fc = nn.Linear(in_size, out_size)
           def forward(self, inp):
                   outp = self.fc(inp)
                   print(inp.shape, outp.device)
                   return outp
   model = DEMO_model(10, 5).to('cuda')

   os.system("echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES")
   device_ids = os.environ.get('CUDA_VISIBLE_DEVICES')
   device_ids = device_ids.split(',')
   device_ids = [int(number) for number in device_ids]
   model = DataParallel(model, device_ids=device_ids) 
   adam = Adam(model.parameters())
   # 进行训练
   for i in range(1):
           x = torch.rand([128, 10])
           y = model(x) 
           loss = torch.norm(y)
           loss.backward()
           adam.step()

执行程序，需要加载torch环境

.. code:: bash

   $ srun -n 8 -p dgx2 --gres=gpu:2 -w vol08 --pty /bin/bash
   $ module load miniconda3
   $ source activate
   (base) $ conda activate pytorch-env
   (pytorch-env) $ python pytorch_test.py
   CUDA_VISIBLE_DEVICES: 0,1
   torch.Size([64, 10]) cuda:0
   torch.Size([64, 10]) cuda:1

