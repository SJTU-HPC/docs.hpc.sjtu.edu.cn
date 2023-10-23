昇腾AI平台使用文档
==================

平台硬件配置
------------

**Atlas800训练服务器（型号：9000）**

参数：

-  CPU: 4 × HUAWEI Kunpeng 920 5250 (2.6GHz, 48 cores)
-  NPU: 8 × Ascend 910A 32GB
-  架构：Arm
-  系统: Centos Linux 8

连接昇腾节点
------------

ssh方式(从π 2.0登录Ascend计算节点)

.. code:: shell

   ssh username@ascend01  # username替换为超算账号

使用昇腾上安装的Miniconda
-------------------------

1. 激活Conda

.. code:: shell

   mkdir Ascend
   cd Ascend
   export PATH=/lustre/opt/contribute/ascend/miniconda3/bin:$PATH
   source activate

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

.. tip::

   使用conda安装虚拟环境时，由于在虚拟环境下安装了Python，也需要安装上述依赖，否则会报错缺少相关依赖文件。

2. 设置环境变量

.. code:: shell

   source /usr/local/Ascend/ascend-toolkit/set_env.sh

.. tip::

   若要进行多卡训练，需要使用新版的CANN软件。目前集群已经安装了7.0.RC1.alpha003版本的CANN，使用时需设置

   .. code:: shell
      
      source /opt/Ascend/ascend-toolkit/set_env.sh
   
   查看集群使用状况使用如下命令

   .. code:: shell

      npu-smi info
   

   可以根据集群使用状况，使用以下命令指定使用的NPU卡号

   .. code:: shell

      export ASCEND_RT_VISIBLE_DEVICES=0

   由于测试集群目前暂未配置slurm操作系统，为防止出现资源挤占，每天晚上11点会清理未释放的进程，请注意保存测试结果。

安装深度学习框架
----------------

安装PyTorch
~~~~~~~~~~~

PyTorch配套支持的Python版本是：Python3.7.x（3.7.5
-3.7.11）、Python3.8.x（3.8.0 - 3.8.11）、Python3.9.x（3.9.0 - 3.9.2）。

1. 创建虚拟环境，安装PyTorch环境依赖

.. code:: shell

   conda create -n pytorch-env python=3.7
   source activate pytorch-env
   pip3 install pyyaml
   pip3 install wheel
   pip3 install typing_extensions

2. 下载官方torch包。

.. code:: shell

   # 如需下载1.8.1版本PyTorch，使用以下命令
   wget https://repo.huaweicloud.com/kunpeng/archive/Ascend/PyTorch/torch-1.8.1-cp37-cp37m-linux_aarch64.whl
   # 如需下载1.11.0版本PyTorch，使用以下命令
   wget https://repo.huaweicloud.com/kunpeng/archive/Ascend/PyTorch/torch-1.11.0-cp37-cp37m-linux_aarch64.whl

3. 安装torch

.. code:: shell

   # 如需安装1.8.1版本PyTorch，使用以下命令
   pip3 install torch-1.8.1-cp37-cp37m-linux_aarch64.whl
   # 如需安装1.11.0版本PyTorch，使用以下命令
   pip3 install torch-1.11.0-cp37-cp37m-linux_aarch64.whl

4. 下载PyTorch插件torch_npu。

.. code:: shell

   # 如需安装1.8.1版本PyTorch插件，使用以下命令
   wget https://gitee.com/ascend/pytorch/releases/download/v5.0.rc1-pytorch1.8.1/torch_npu-1.8.1.post1-cp37-cp37m-linux_aarch64.whl
   # 如需安装1.11.0版本PyTorch插件，使用以下命令
   wget https://gitee.com/ascend/pytorch/releases/download/v5.0.rc1-pytorch1.11.0/torch_npu-1.11.0-cp37-cp37m-linux_aarch64.whl
   #如果下载whl包时出现ERROR: cannot verify gitee.com's certificate报错，可在下载命令后加上--no-check-certificate参数避免此问题。样例代码如下所示。
   wget https://gitee.com/ascend/pytorch/releases/download/v5.0.rc1-pytorch1.11.0/torch_npu-1.11.0-cp37-cp37m-linux_aarch64.whl --no-check-certificate

5. 安装torch_npu插件

.. code:: shell

   # 如需安装1.8.1版本PyTorch配套插件，使用以下命令
   pip3 install torch_npu-1.8.1.post1-cp37-cp37m-linux_aarch64.whl
   # 如需安装1.11.0版本PyTorch配套插件，使用以下命令
   pip3 install torch_npu-1.11.0-cp37-cp37m-linux_aarch64.whl

6. 安装对应版本的torchvision

.. code:: shell

   # 如需安装1.8.1版本PyTorch配套torchvision，使用以下命令
   pip3 install torchvision==0.9.1
   # 如需安装1.11.0版本PyTorch配套torchvision，使用以下命令
   pip3 install torchvision==0.12.0

7. 安装深度学习加速库Apex

.. code:: shell

   # 如需安装1.8.1版本PyTorch配套Apex，使用以下命令
   pip3 install apex --no-index --find-links https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/pytorch1_8_1/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com
   # 如需安装1.11.0版本PyTorch配套Apex，使用以下命令
   pip3 install apex --no-index --find-links https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/pytorch1_11_0/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com

8. 执行以下命令验证，若返回True则说明安装成功

.. code:: shell

   python3 -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"

安装TensorFlow
~~~~~~~~~~~~~~

1. 配置环境变量，创建虚拟环境

.. code:: shell

   source /usr/local/Ascend/tfplugin/set_env.sh
   conda create -n tensorflow-env python=3.7
   source activate tensorflow-env

2. 安装TensorFlow

.. code:: shell

   # 如需安装TensorFlow1.15版本，使用以下命令
   pip3 install tensorflow==1.15.0 --no-index --find-links  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com
   # 如需安装TensorFlow2.6.5版本，使用以下命令
   pip3 install tensorflow==2.6.5 --no-index --find-links  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com

3. 执行以下命令验证安装效果

.. code:: shell

   python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

若返回张量则表示安装成功。

安装昇思MindSpore
~~~~~~~~~~~~~~~~~

1. 创建虚拟环境

.. code:: shell

   conda create -n mindspore_py37 python=3.7 -y
   source activate mindspore_py37

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
--------

运行PyTorch样例
~~~~~~~~~~~~~~~

1. 获取模型脚本并进入模型代码所在目录。

.. code:: shell

   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/PyTorch/built-in/cv/classification/MobileNetV3-Large_ID1784_for_PyTorch
   conda activate pytorch-env

2. 安装依赖

.. code:: shell

   pip install -r 1.8_requirements.txt

3. 获取数据集

.. code:: shell

   cp /lustre/share/scidata/tiny-imagenet-200.zip ./
   unzip tiny-imagenet-200.zip

4. 运行训练脚本

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

关于迁移工具的高级功能，请见昇腾文档\ `《分析迁移工具》 <https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasfmkt_16_0001.html>`__\ 中的”msFmkTransplt”章节。

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
   source /opt/Ascend/ascend-toolkit/set_env.sh
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

1. 激活mindspore_py37环境

.. code:: shell

   conda activate mindspore_py37

2. 下载\ `MindScience套件 <https://gitee.com/mindspore/mindscience>`__

.. code:: shell

   git clone https://gitee.com/mindspore/mindscience.git

3. 安装MindSpore SPONGE 计算生物领域套件

.. code:: shell

   cd ./mindscience/MindSPONGE/
   pip install -r requirements.txt
   pip install mindsponge-ascend

4. 配置数据库检索路径

根据数据库安装情况配置config/data.yaml中数据库搜索的相关配置database_search

.. code:: shell

   cd ./applications/MEGAProtein/
   vim ./config/data.yaml

相关参数配置如下：

.. code:: shell

   # configuration for template search
   hhsearch_binary_path: "/home/megaprotein/hhsuite/bin/hhsearch" HHsearch可执行文件路径
   kalign_binary_path: "/home/megaprotein/kalign/kalign" kalign可执行文件路径
   pdb70_database_path: "/home/megaprotein/pdb70/pdb70" {pdb70文件夹}/pdb70
   mmcif_dir: "/scratch/share/AlphaFold/data/pdb_mmcif/mmcif_files" mmcif文件夹
   obsolete_pdbs_path: "/scratch/share/AlphaFold/data/pdb_mmcif/obsolete.dat" PDB IDs的映射文件路径
   max_template_date: "2100-01-01" 模板搜索截止时间，该时间点之后的模板会被过滤掉，默认值"2100-01-01"
   # configuration for Multiple Sequence Alignment
   mmseqs_binary: "/home/megaprotein/mmseqs/bin/mmseqs" MMseqs2可执行文件路径
   uniref30_path: "/home/megaprotein/uniref30_2103/uniref30_2103_db" {uniref30文件夹}/uniref30_2103_db
   database_envdb_dir: "/home/megaprotein/colabfold_envdb_202108/colabfold_envdb_202108_db" {colabfold_envdb文件夹}/colabfold_envdb_202108_db
   a3m_result_path: "./a3m_result/" mmseqs2检索结果(msa)的保存路径，默认值"./a3m_result/"

5. 下载模型权重

.. code:: shell

   wget https://download.mindspore.cn/mindscience/mindsponge/MEGAFold/checkpoint/MEGA_Fold_1.ckpt

6. 运行推理程序

.. code:: shell

   python main.py \
   --data_config ./config/data.yaml \ #数据预处理参数配置
   --model_config ./config/model.yaml \ #模型超参配置
   --run_platform Ascend \ #运行后端，Ascend或者GPU，默认Ascend
   --input_path ./examples/pkl/ \ #输入文件目录，可包含多个.fasta/.pkl文件
   --use_pkl \ #使用pkl数据作为输入，默认False
   --checkpoint_path ./MEGA_Fold_1.ckpt \模型权重文件路径

参考资料
--------

https://support.huawei.com/enterprise/zh/doc/EDOC1100289999/4fc08621

https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/overview/index.html

https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein#mega-protein

https://gitee.com/ascend/modelzoo

FAQ
---
