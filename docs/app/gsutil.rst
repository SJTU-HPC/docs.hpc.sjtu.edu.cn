.. _gsutil:

GsUtil
====================

简介
-------------------

gsutil 是一个 Python 应用程序，可让您从命令行访问 Google Cloud Storage 。您可以使用 gsutil 执行各种存储桶和对象
管理任务。

完整步骤
---------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c conda-forge gsutil
