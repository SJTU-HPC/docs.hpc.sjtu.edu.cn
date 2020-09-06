Pi超算集群用户文档
==================

本仓库维护Pi超算集群用户文档。本文档使用 `reStructuredText(rst) <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ 书写，并会被 `Sphinx <https://www.sphinx-doc.org>`_ 编译为Web站点或PDF文档。本站使用的Web主题是 `Material for Sphinx <https://bashtage.github.io/sphinx-material/>`_ 。

搭建环境
--------

Sphinx是一个将rst文档编译为HTML、PDF、eBook等出版物的python包，用于本文档网页渲染。在开始之前请确保本地已经有python环境。如果您的电脑暂时没有可用的python环境，那么我们建议您使用\ `anaconda <https://www.anaconda.com/>`__\ 创建一个。

使用以下指令安装构建这个文档所需的Python包::

   pip3 install -r requirements.txt

Windows用户检查sphinx是否安装完成::

   where sphinx-build

Linux和Mac用户检查sphinx是否安装成功::

   which sphinx-build

`Pandoc <https://pandoc.org>`_ 能够将markdown文档转换为rst文档，减少重复操作::

  brew install pandoc
  pandoc -t index.md -o index.rst

在本地预览文档更新 
------------------

重新编译HTML站点::

  $ make -C docs html

重新编译PDF文档::

  $ make -C docs latexpdf

从仓库更新文档代码::

  $ git pull origin

