# <center>TensorFlow</center>

-----------

TensorFlow 是一个端到端开源机器学习平台。它拥有一个包含各种工具、库和社区资源的全面灵活生态系统，可以让研究人员推动机器学习领域的先进技术的发展，并让开发者轻松地构建和部署由机器学习提供支持的应用。

## 使用miniconda安装TensorFlow

创建名为`tf-env`的虚拟环境，激活虚拟环境，然后安装TensorFlow。

```bash
$ module load miniconda3
$ conda create -n tf-env
$ source activate tf-env
$ conda install pip
$ pip install tensorflow-gpu==2.0.0
```

## 使用miniconda提交TensorFlow作业

以下为在DGX-2上使用TensorFlow的虚拟环境作业脚本示例，其中作业使用单节点并分配2块GPU：

```bash
#!/bin/bash
#SBATCH -J test
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:2

module load cuda/10.0.130-gcc-4.8.5 cudnn

module load miniconda3
source activate tf-env

python -c 'import tensorflow as tf; \
           print(tf.__version__);   \
           print(tf.test.is_gpu_available());'
```

我们假设这个脚本文件名为`tensorflow_conda.slurm`,使用以下指令提交作业。

```bash
$ sbatch tensorflow_conda.slurm
```

## 使用singularity容器中的TensorFlow

集群中已经预置了[NVIDIA GPU CLOUD](https://ngc.nvidia.com/)提供的优化镜像，通过调用该镜像即可运行TensorFlow作业，无需单独安装，目前版本为`tensorflow-2.0.0`。该容器文件位于`/lustre/share/img/tensorflow-2.0.0.simg`


## 使用singularity容器提交TensorFlow作业

以下为在DGX-2上使用TensorFlow的容器作业脚本示例，其中作业使用单节点并分配2块GPU：

```bash
#!/bin/bash
#SBATCH -J test
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:2

IMAGE_PATH=/lustre/share/img/tensorflow-2.0.0.simg

singularity run --nv $IMAGE_PATH python -c 'import tensorflow as tf; \
                                            print(tf.__version__);   \
                                            print(tf.test.is_gpu_available());'
```

我们假设这个脚本文件名为`tensorflow_singularity.slurm`,使用以下指令提交作业。

```bash
$ sbatch tensorflow_singularity.slurm
```

## 参考文献

- [TensorFlow官方网站](https://www.tensorflow.org/)
- [NVIDIA GPU CLOUD](ngc.nvidia.com)
- [Singularity文档](https://sylabs.io/guides/3.5/user-guide/)
