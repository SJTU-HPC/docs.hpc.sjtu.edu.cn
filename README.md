# Pi超算集群用户文档

本仓库维护Pi超算集群用户文档。本文档使用markdown书写，并会被[mkdocs](http://www.mkdocs.org)转译为html/css/js。本站使用的主题是[mkdocs-material](https://github.com/squidfunk/mkdocs-material)的分支版本[theme](https://gitlab.com/NERSC/mkdocs-material)。

## 搭建环境

mkdocs是一个python包，用于本文档网页渲染。在开始之前请确保本地已经有python环境。如果您的电脑暂时没有可用的python环境，那么我们建议您使用[anaconda](https://www.anaconda.com/)创建一个。

使用以下指令安装指定的python包：

```
pip install -r requirements.txt
```

检查mkdocs是否安装完成：

Windows用户

```
where mkdocs
```

linux和mac用户

```
which mkdocs
```

## 启动服务器和更新文档

**更新文档需要管理员权限**

本仓库包含makefile，如果您的工作环境可以使用make指令，那么

本地启动服务器

```
make server
```

对于make不可用的情况

本地启动服务器

```
mkdocs serve
```

更新文档

```
# 将更新推送到仓库master分支自动部署
```
