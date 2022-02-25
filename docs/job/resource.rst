查看作业资源信息
================

当作业对CPU和存储的要求较高时，了解运行作业的CPU和存储的使用信息，能够保证作业顺利运行。

存储
----

运行作业存储分配策略
~~~~~~~~~~~~~~~~~~~~

+--------+--------------------------------------------+
| 平台   | 存储分配策略                               |
+========+============================================+
| π2.0   | 每核配比4G内存；单节点配置为40核，180G内存 |
+--------+--------------------------------------------+

作业运行中的存储占用
~~~~~~~~~~~~~~~~~~~~

当提交作业后，使用 ``squeue`` 命令查看作业使用的节点

.. code:: bash

   [hpc@login2 test]$ squeue 
                JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              9709875       cpu  80cores      hpc  R       0:02      1 cas478

然后进入相关节点

.. code:: bash

   ssh cas478

可根据用户名查看作业占用的存储空间

.. code:: bash

    ps -u$USER -o %cpu,rss,args

示例如下： ``ps -uhpc -o %cpu,rss,args``

.. code:: bash

   %CPU   RSS COMMAND
   98.5 633512 pw.x -i ausurf.in
   98.5 652828 pw.x -i ausurf.in
   98.6 654312 pw.x -i ausurf.in
   98.6 652196 pw.x -i ausurf.in

``RSS`` 表示单核所占用的存储空间，单位为MB。

如果需要动态监测存储资源的使用，可进入计算节点后，输入top命令

.. code:: bash

   top - 10:33:29 up 84 days, 52 min,  1 user,  load average: 14.43, 16.82, 24.25
   Tasks: 754 total,  41 running, 713 sleeping,   0 stopped,   0 zombie
   %Cpu(s): 98.9 us,  1.1 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
   KiB Mem : 19649038+total, 16052179+free, 27783672 used,  8184924 buff/cache
   KiB Swap: 33554428 total, 33187480 free,   366948 used. 16633760+avail Mem 
   
      PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND                                                                    
   428410 hpchgc    20   0 5989.9m 662.5m 168.6m R 100.0  0.3   0:22.67 pw.x                                                                       
   428419 hpchgc    20   0 5987.0m 658.9m 163.7m R 100.0  0.3   0:22.61 pw.x                                                                       
   428421 hpchgc    20   0 5984.6m 677.8m 180.1m R 100.0  0.4   0:22.66 pw.x                                                                       
   428433 hpchgc    20   0 6002.8m 661.7m 165.3m R 100.0  0.3   0:22.68 pw.x                                                                       
   428436 hpchgc    20   0 5986.0m 659.0m 165.4m R 100.0  0.3   0:22.66 pw.x                                                                       
   428446 hpchgc    20   0 5991.4m 550.2m  64.2m R 100.0  0.3   0:22.73 pw.x
