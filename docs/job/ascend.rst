昇腾AI平台使用文档
==================

平台硬件配置
------------

**Atlas800训练服务器（型号：9000）**

参数：

-  CPU: 4 × HUAWEI Kunpeng 920 5250 (2.6GHz, 48 cores)
-  NPU: Ascend 910A 32GB
-  架构：Arm
-  系统: Centos Linux 8

连接昇腾节点
------------

1. ssh方式(限校内IP，或使用SJTU VPN)

.. code:: shell

   ssh username@202.120.58.248

安装Miniconda
-------------

1. 下载\ `Miniconda3 Linux-aarch64
   64-bit <https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh>`__\ 。

.. code:: shell

   mkdir Ascend
   cd ./Ascend
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

2. 执行Miniconda3安装脚本。

.. code:: shell

   chmod +x ./Miniconda3-latest-Linux-aarch64.sh
   ./Miniconda3-latest-Linux-aarch64.sh

安装过程中注意:

-  建议安装路径为~/Ascend/miniconda3。
-  ``Do you wish the installer to initialize Miniconda3 by running conda init?``\ 建议选择\ ``no``\ 。

3. 激活conda并创建一个虚拟环境。

.. code:: shell

   source ~/Ascend/miniconda3/bin/activate
   conda create -n test python=3.7
   conda activate test

准备开发环境
------------

1. 安装软件相关依赖。

.. code:: shell

   pip3 install attrs
   pip3 install numpy
   pip3 install decorator
   pip3 install sympy
   pip3 install cffi
   pip3 install pyyaml
   pip3 install pathlib2
   pip3 install psutil
   pip3 install protobuf
   pip3 install scipy
   pip3 install requests
   pip3 install absl-py

2. 设置环境变量

.. code:: shell

   #修改.bashrc文件
   vim ~/.bashrc
   #在文件最后一行添加环境变量
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   #保存文件并退出
   :wq!
   #使环境变量生效
   source ~/.bashrc

安装深度学习框架
----------------

安装PyTorch
~~~~~~~~~~~

PyTorch配套的Python版本是：Python3.7.x（3.7.5 -
3.7.11）、Python3.8.x（3.8.0 - 3.8.11）、Python3.9.x（3.9.0 - 3.9.2）。

1. 下载官方torch包。

.. code:: shell

   # 安装1.8.1版本
   wget https://repo.huaweicloud.com/kunpeng/archive/Ascend/PyTorch/torch-1.8.1-cp37-cp37m-linux_aarch64.whl
   # 安装1.11.0版本
   wget https://repo.huaweicloud.com/kunpeng/archive/Ascend/PyTorch/torch-1.11.0-cp37-cp37m-linux_aarch64.whl

2. 安装torch

.. code:: shell

   # 安装1.8.1版本
   pip3 install torch-1.8.1-cp37-cp37m-linux_aarch64.whl
   # 安装1.11.0版本
   pip3 install torch-1.11.0-cp37-cp37m-linux_aarch64.whl

3. 下载PyTorch插件torch_npu。

.. code:: shell

   # 安装1.8.1版本
   wget https://gitee.com/ascend/pytorch/releases/download/v5.0.rc1-pytorch1.8.1/torch_npu-1.8.1.post1-cp37-cp37m-linux_aarch64.whl
   # 安装1.11.0版本
   wget https://gitee.com/ascend/pytorch/releases/download/v5.0.rc1-pytorch1.11.0/torch_npu-1.11.0-cp37-cp37m-linux_aarch64.whl
   #如果下载whl包时出现ERROR: cannot verify gitee.com's certificate报错，可在下载命令后加上--no-check-certificate参数避免此问题。样例代码如下所示。
   wget https://gitee.com/ascend/pytorch/releases/download/v5.0.rc1-pytorch1.11.0/torch_npu-1.11.0-cp37-cp37m-linux_aarch64.whl --no-check-certificate

4. 安装torch_npu插件

.. code:: shell

   # 安装1.8.1版本
   pip3 install torch_npu-1.8.1.post1-cp37-cp37m-linux_aarch64.whl
   # 安装1.11.0版本
   pip3 install torch_npu-1.11.0-cp37-cp37m-linux_aarch64.whl

5. 安装对应版本的torchvision

.. code:: shell

   #PyTorch 1.8.1需安装0.9.1版本，PyTorch 1.11.0需安装0.12.0版本
   pip3 install torchvision==0.9.1

6. 安装深度学习加速库Apex

.. code:: shell

   #安装1.8.1版本
   pip3 install apex --no-index --find-links https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/pytorch1_8_1/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com
   #安装1.11.0版本
   pip3 install apex --no-index --find-links https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/pytorch1_11_0/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com

安装TensorFlow
~~~~~~~~~~~~~~

1. 配置环境变量

.. code:: shell

   #修改.bashrc文件
   vim ~/.bashrc
   #在文件最后一行添加环境变量
   source /usr/local/Ascend/tfplugin/set_env.sh
   #保存并退出文件
   :wq!
   #使环境变量生效
   source ~/.bashrc

2. 安装TensorFlow1.15

.. code:: shell

   #安装TensorFlow1.15
   pip3 install tensorflow==1.15.0 --no-index --find-links  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com
   #安装TensorFlow2.6.5
   pip3 install tensorflow==2.6.5 --no-index --find-links  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com

安装昇思MindSpore
~~~~~~~~~~~~~~~~~

1. 创建虚拟环境

.. code:: shell

   conda create -n mindspore_py37 python=3.7 -y
   conda activate mindspore_py37
   #升级pip
   python -m pip install -U pip

2. 更新pip

.. code:: shell

   python -m pip install -U pip

3. 安装昇腾AI处理器配套软件包

.. code:: shell

   pip install sympy
   pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
   pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl

4. 安装MindSpore

.. code:: shell

   export MS_VERSION=2.0.0
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

5. 验证是否成功安装

.. code:: shell

   python -c "import mindspore;mindspore.run_check()"

若返回：

.. code:: shell

   MindSpore version: 2.0.0
   The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!

说明MindSpore安装成功。

运行样例
~~~~~~~~

1. 获取模型脚本并进入模型代码所在目录。

.. code:: shell

   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/PyTorch/contrib/cv/classification/MobileNetV3_large_100_for_PyTorch

2. 配置虚拟环境

.. code:: shell

   conda create -n benchmark python=3.7
   conda activate benchmark

3. 安装PyTorch框架

见\ `安装PyTorch <#安装PyTorch>`__\ 小节

4. 安装依赖

.. code:: shell

   cd ModelZoo-PyTorch/PyTorch/contrib/cv/classification/MobileNetV3_large_100_for_PyTorch

5. 获取数据集

.. code:: shell

   cd /home/tiny-imagenet-200.zip ./
   unzip tiny-imagenet-200.zip

6. 运行训练脚本

.. code:: shell

   bash ./test/train_full_1p.sh --data_path=./tiny-imagenet-200

PyTorch模型迁移
---------------

自动迁移
~~~~~~~~

1. 配置环境变量。

.. code:: shell

   export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge/:$PYTHONPATH

2. 在训练脚本中导入以下库代码。

.. code:: shell

   import torch
   import torch_npu
   .....
   from torch_npu.contrib import transfer_to_npu

迁移分析工具
~~~~~~~~~~~~

利用PyTorch迁移分析工具能够分析代码中API的支持情况。

1. 环境准备

.. code:: shell

   pip3 install pandas
   pip3 install libcst
   pip3 install jedi

2. 进入迁移工具所在路径

.. code:: shell

   cd /usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/

3. 执行脚本迁移分析任务

参数说明： - -i: 要进行迁移的原始脚本文件所在文件夹路径 - -o:
脚本迁移结果文件输出路径。 - -v: 脚本迁移结果文件输出路径。

.. code:: shell

   ./pytorch_gpu2npu.sh -i 原始脚本路径 -o 脚本迁移结果输出路径 -v 原始脚本框架版本

4. 查看结果文件

.. code:: shell

   ├── xxx_msft/xxx_msft_multi              // 脚本迁移结果输出目录
   │   ├── 生成脚本文件                 // 与迁移前的脚本文件目录结构一致
   │   ├── msFmkTranspltlog.txt         // 脚本迁移过程日志文件，日志文件限制大小为1M，若超过限制将分多个文件进行存储，最多不会超过10个
   │   ├── cuda_op_list.csv            //分析出的cuda算子列表
   │   ├── unknown_api.csv             //支持情况存疑的API列表
   │   ├── unsupported_api.csv         //不支持的API列表
   │   ├── change_list.csv              // 修改记录文件
   │   ├── run_distributed_npu.sh       // 多卡启动shell脚本
   │   ├── ascend_function              // 如果启用了Replace Unsupported APIs参数，会生成该包含等价算子的目录
   │   ├── ascend_modelarts_function
   │   │   ├── modelarts_path_manager.py    // 启用ModelArts参数，会生成该路径映射适配层代码文件
   │   │   ├── path_mapping_config.py       // 启用ModelArts参数，会生成该路径映射配置文件

关于迁移工具的高级功能，请见昇腾文档\ `《分析迁移工具》 <https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasfmkt_16_0001.html>`__\ 中的“msFmkTransplt”章节。

迁移单卡脚本为多卡脚本
~~~~~~~~~~~~~~~~~~~~~~

目前节点仅支持单机多卡（最多8卡）

1. 在主函数中适当位置修改训练代码

.. code:: python

   #传入local_rank, world_size
   local_rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])

   #用local_rank自动获取device号
   device = torch.device('npu', local_rank)

   #初始化，将通信方式设置为hccl
   torch.distributed.init_process_group(backend="hccl",rank=local_rank)

   #在初始化时确定当前的device
   torch_npu.npu.set_device(device)

   #获取训练数据集后，设置train_sampler
   train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

   #定义模型后，开启DDP模式
   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

   #将train_dataloader与train_sampler相结合
   train_dataloader = DataLoader(dataset = train_data, batch_size=batch_size, sampler = train_sampler)

2. 编写拉起多卡训练脚本

脚本命名为\ ``train.sh``

.. code:: shell

   #两卡训练示例脚本
   source env_npu.sh
   cur_path=`pwd`
   if [ $(uname -m) = "aarch64" ]
   then
       #配置多卡端口
       export MASTER_ADDR=127.0.0.1
       export MASTER_PORT=29500
       export WORLD_SIZE=2
       #配置多进程绑核
       for i in $(seq 0 1)
       do
               export LOCAL_RANK=$i
               let p_start=0+24*i
               let p_end=23+24*i
               #启动训练，参数根据训练代码进行自定义
               nohup taskset -c $p_start-$p_end python3 -u train.py --local_rank=$i > ${cur_path}/train.log 2>&1 &
       done
   else
       python3 -m torch.distributed.launch --nproc_per_node=2 train.py > ${cur_path}/train_x86.log 2>&1 &
   fi

3. 启动多卡训练

.. code:: shell

   bash ./train.sh

安装MEGA-Protein
----------------

`MEGA-Protein <https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein#mega-protein>`__\ 是北大高毅勤老师团队与华为MindSpore科学计算团队合作开发的蛋白质结构预测工具，针对AlphaFold2数据前处理耗时过长、缺少MSA时预测精度不准、缺乏通用评估结构质量工具的问题进行创新优化。

MEGA-Fold蛋白质结构预测推理
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 下载\ `MindScience套件 <https://gitee.com/mindspore/mindscience>`__\ 并进入MEGA-Protein目录

.. code:: shell

   git clone https://gitee.com/mindspore/mindscience.git
   cd ./mindscience/MindSPONGE/applications/MEGAProtein/ 

2. 配置数据库检索路径

根据数据库安装情况配置config/data.yaml中数据库搜索的相关配置database_search

.. code:: shell

   vim ./config/data.yaml

   # configuration for template search
   hhsearch_binary_path: "/home/data/megaprotein/hh-suite/build/bin/hhsearch" HHsearch可执行文件路径
   kalign_binary_path: "/home/data/megaprotein/kalign/kalign" kalign可执行文件路径
   pdb70_database_path: "/home/data/megaprotein/pdb70/pdb70" {pdb70文件夹}/pdb70
   mmcif_dir: "/home/data/megaprotein/pdb_mmcif/mmcif_files" mmcif文件夹
   obsolete_pdbs_path: "/home/data/megaprotein/pdb_mmcif/obsolete.dat" PDB IDs的映射文件路径
   max_template_date: "2100-01-01" 模板搜索截止时间，该时间点之后的模板会被过滤掉，默认值"2100-01-01"
   # configuration for Multiple Sequence Alignment
   mmseqs_binary: "/home/data/mmseqs/bin/mmseqs" MMseqs2可执行文件路径
   uniref30_path: "/home/data/megaprotein/uniref30/uniref30_2103/uniref30_2103_db" {uniref30文件夹}/uniref30_2103_db
   database_envdb_dir: "/home/data/megaprotein/colabfold_envdb_202108/colabfold_envdb_202108_db" {colabfold_envdb文件夹}/colabfold_envdb_202108_db
   a3m_result_path: "./a3m_result/" mmseqs2检索结果(msa)的保存路径，默认值"./a3m_result/"

3. 运行推理程序

.. code:: shell

   python main.py \
   --data_config ./config/data.yaml \ #数据预处理参数配置
   --model_config ./config/model.yaml \ #模型超参配置
   --run_platform Ascend \ #运行后端，Ascend或者GPU，默认Ascend
   --input_path INPUT_FILE_PATH \ #输入文件目录，可包含多个.fasta/.pkl文件
   --use_pkl \ #使用pkl数据作为输入，默认False
   --checkpoint_path CHECKPOINT_PATH \模型权重文件路径

参考资料
--------

https://support.huawei.com/enterprise/zh/doc/EDOC1100289999/4fc08621

https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/overview/index.html

https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein#mega-protein

https://gitee.com/ascend/modelzoo

FAQ
---
