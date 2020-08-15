Pi超算集群用户文档
==================

本仓库维护Pi超算集群用户文档。本文档使用 `reStructuredText(rst) <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ 书写，并会被 `Sphinx <https://www.sphinx-doc.org>`_ 编译为Web站点或PDF文档。本站使用的Web主题是 `Material for Sphinx <https://bashtage.github.io/sphinx-material/>`_ 。

搭建环境
--------

Sphinx是一个将rst文档编译为HTML、PDF、eBook等出版物的python包，用于本文档网页渲染。在开始之前请确保本地已经有python环境。如果您的电脑暂时没有可用的python环境，那么我们建议您使用\ `anaconda <https://www.anaconda.com/>`__\ 创建一个。

使用以下指令安装构建这个文档所需的Python包::

   pip3 install -r requirements.txt
   pip3 install -r docs/requirements.txt

Windows用户检查sphinx是否安装完成::

   where sphinx-build

Linux和Mac用户检查sphinx是否安装陈宫个::

   which sphinx-build

在本地预览文档更新 
------------------

重新编译HTML站点::

  $ cd docs
  $ make build

重新编译PDF文档::

  $ cd docs
  $ make latexpdf

更新文档::

  $ git pull origin

