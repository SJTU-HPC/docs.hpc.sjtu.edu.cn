Pi超算集群用户文档
==================

本仓库维护π超算集群用户文档。本文档使用 `reStructuredText(rst) <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ 书写，并会被 `Sphinx <https://www.sphinx-doc.org>`_ 编译为Web站点或PDF文档。
Sphinx是一个将rst文档编译为HTML、PDF、eBook等出版物的python包，用于本文档网页渲染。
本站使用的Web主题是 `Material for Sphinx <https://bashtage.github.io/sphinx-material/>`_ 。

使用预编译容器镜像构建文档(推荐)
--------------------------------

我们推荐您使用预编译容器镜像构建本文档集合，包括一份简版手册、一份完整版手册和一个网站::

  docker run --rm -it -v ${PWD}:/data sjtuhpc/texlive-sphinx:focal make -C tex all
  docker run --rm -it -v ${PWD}:/data sjtuhpc/texlive-sphinx:focal make -C docs clean latexpdf
  docker run --rm -it -v ${PWD}:/data sjtuhpc/texlive-sphinx:focal make -C docs clean html

预编译镜像的Dockerfile见 https://github.com/SJTU-HPC/texlive-sphinx 。

使用本机Python环境构建文档
--------------------------

您也可以在本机Python环境安装依赖包，构建本文档集::

   pip3 install -r requirements.txt

Windows用户检查sphinx是否安装完成::

   where sphinx-build

Linux和Mac用户检查sphinx是否安装成功::

   which sphinx-build

生成HTML站点::

  $ make -C docs html

查看HTML站点内容::

  $ open docs/_build/html/index.html

生成PDF文档::

  $ make -C docs latexpdf

查看PDF文档::

  $ open docs/_build/latex/sphinx.pdf

修改和更新文档
--------------

远程仓库的 ``master`` 分支是受保护的，无法直接推送更新，需要推送到新分支，然后再从新分支发起合并请求。

新建本地分支，提交代码::

  $ git checkout -b your_new_branch
  $ git commit -m "Your commit message"

推送更新到远程仓库::

  $ git push origin your_new_branch

最后在GitLab页面中发起合并请求。
