常见问题
=========

Q：计算节点不能访问互联网/不能下载数据 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A：** 计算节点是通过proxy节点代理进行网络访问的，因此一些软件需要特定的代理设置。需要找到软件的配置文件，修改软件的代理设置。

a) git、wget、curl等软件支持通用变量，代理参数设置为：

.. code:: bash

 # 思源一号计算节点通用代理设置
 https_proxy=http://proxy2.pi.sjtu.edu.cn:3128
 http_proxy=http://proxy2.pi.sjtu.edu.cn:3128
 no_proxy=puppet,proxy,172.16.0.133,pi.sjtu.edu.cn

  # π2.0计算节点通用代理设置
 http_proxy=http://proxy.pi.sjtu.edu.cn:3004/
 https_proxy=http://proxy.pi.sjtu.edu.cn:3004/
 no_proxy=puppet

b) Python、MATLAB、Rstudio、fasterq-dump等软件需要查询软件官网确定配置参数：

.. code:: bash

 ### fasterq-dump文件，配置文件路径 ~/.ncbi/user-settings.mkfg

 # 思源一号节点代理设置
 /tools/prefetch/download_to_cache = "true"
 /http/proxy/enabled = "true"
 /http/proxy/path = "http:/proxy2.pi.sjtu.edu.cn:3128"

 # π2.0节点代理设置
 /tools/prefetch/download_to_cache = "true"
 /http/proxy/enabled = "true"
 /http/proxy/path = "http://proxy.pi.sjtu.edu.cn:3004"

 ### Python需要在代码里面指定代理设置，不同Python包代理参数可能不同

 # 思源一号节点代理设置
 proxies = {
     'http': 'http://proxy2.pi.sjtu.edu.cn:3128',
     'https': 'http://proxy2.pi.sjtu.edu.cn:3128',
 }
 # π2.0节点代理设置
 proxies = {
     'http': 'http://proxy.pi.sjtu.edu.cn:3004',
     'https': 'http://proxy.pi.sjtu.edu.cn:3004',
 }

 ### MATLAB

 # 思源一号节点代理设置
 proxy2.pi.sjtu.edu.cn:3128

 # π2.0节点代理设置
 proxy.hpc.sjtu.edu.cn:3004

