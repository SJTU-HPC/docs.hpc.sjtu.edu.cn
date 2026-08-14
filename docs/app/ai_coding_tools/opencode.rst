.. _opencode:

OpenCode
========

简介
----

OpenCode 是一款开源 AI 编程代理，可在终端中阅读和修改项目文件、运行命令、分析错误，并协助完成代码开发与维护任务。本说明介绍如何在 Pi 2.0 和思源一号上启动 OpenCode，并接入上海交通大学校内大模型服务。

.. warning::

   OpenCode 可以修改文件并执行 shell 命令。建议只在明确的项目目录中启动，并优先在 Git 仓库中使用。开始任务前请检查或提交现有改动，执行删除、覆盖、提交、推送等高风险操作前应仔细确认。

使用 OpenCode 时，任务描述、引用的文件内容和必要的上下文可能会发送到所选模型服务。请勿处理涉密数据，也不要在提示词、项目文件、截图或日志中暴露 API Key、密码及其他敏感信息。

接入校内模型
------------

申请 API Key
^^^^^^^^^^^^

按照 `上海交通大学本地大模型 API 使用文档 <https://claw.sjtu.edu.cn/guide/sjtu-api/>`__ 申请 API Key。

备份已有配置
^^^^^^^^^^^^

Pi 2.0 和思源一号的用户级配置都位于 ``~/.config/opencode/opencode.jsonc``。如需快速备份当前配置，可执行：

.. code-block:: bash

   cp ~/.config/opencode/opencode.jsonc ~/.config/opencode/opencode.jsonc.bak

复制所在集群的样例配置
^^^^^^^^^^^^^^^^^^^^^^

在 Pi 2.0 上执行：

.. code-block:: bash

   cp /lustre/share/samples/opencode/opencode.jsonc ~/.config/opencode/opencode.jsonc

在思源一号上执行：

.. code-block:: bash

   cp /dssg/share/sample/opencode/opencode.jsonc ~/.config/opencode/opencode.jsonc

上述 ``cp`` 命令会覆盖当前配置。若已经自定义过 OpenCode，请先比较备份文件与样例配置，再合并需要的内容，不要直接丢弃原有的 provider、MCP、权限或快捷键设置。OpenCode 的配置格式支持 JSON 和 JSONC。

可使用以下命令查看 OpenCode 最终加载的配置：

.. code-block:: bash

   opencode debug config

该命令可能显示 provider 地址和其他配置信息。分享输出前请先检查其中是否包含敏感内容。

设置 API Key
^^^^^^^^^^^^

临时设置，仅对当前终端会话及其子进程生效：

.. code-block:: bash

   export SJTU_LLM_API_KEY="你的 API Key"

如需在后续登录中自动设置，可编辑 ``~/.bashrc``：

.. code-block:: bash

   vi ~/.bashrc

在文件末尾加入以下内容，然后重新加载配置：

.. code-block:: bash

   export SJTU_LLM_API_KEY="你的 API Key"
   source ~/.bashrc

检查环境变量是否已经生效。该命令只显示状态，不会输出 API Key：

.. code-block:: bash

   test -n "$SJTU_LLM_API_KEY" && echo "API Key 已设置" || echo "API Key 未设置"

请勿将 API Key 写入项目配置、提交到 Git，或通过终端截图和日志发送给他人。

验证模型配置
^^^^^^^^^^^^

集群样例使用 ``sjtu-llm`` provider。加载模块并检查当前可用的模型：

.. code-block:: bash

   module load opencode
   opencode models sjtu-llm

模型会随校内服务调整，请以该命令或终端界面中 ``/models`` 的实际输出为准。模型的完整标识采用 ``provider/model`` 格式。

启动 OpenCode
^^^^^^^^^^^^^

.. code-block:: bash

   # 加载集群部署的opencode
   module load opencode
   # 进入开发项目目录
   cd /path/to/your/project
   # 启动opencode
   opencode

进入终端界面后输入 ``/models``，选择 ``sjtu-llm`` 下当前可用的模型。

首次使用
--------

Pi 2.0
^^^^^^

登录 Pi 2.0 的 ``pilogin`` 节点，加载 OpenCode 模块：

.. code-block:: bash

   module load opencode

进入需要处理的项目目录并启动 OpenCode：

.. code-block:: bash

   cd /path/to/your/project
   opencode

也可以直接将项目目录作为参数：

.. code-block:: bash

   opencode /path/to/your/project

思源一号
^^^^^^^^

在 Netcatty 中选择 ``sylogin(用户名)``，或通过 SSH 登录：

.. code-block:: bash

   ssh 用户名@sylogin.hpc.sjtu.edu.cn

加载 OpenCode 模块：

.. code-block:: bash

   module load opencode

进入需要处理的项目目录并启动 OpenCode：

.. code-block:: bash

   cd /path/to/your/project
   opencode

在登录节点调用 ``opencode`` 时，集群提供的包装程序会申请调试资源，并在计算节点上启动 OpenCode。终端出现以下信息时，表示作业正在正常排队，请等待资源分配：

.. code-block:: text

   job queued and waiting for resources

如果已经位于计算节点，OpenCode 会直接运行，不会再次申请资源。请勿因等待时间较长而重复启动多个实例。

首次进入 OpenCode 后，在输入框中输入自然语言任务即可开始使用，例如：

.. code-block:: text

   请先阅读这个项目，说明它的目录结构和主要入口，不要修改文件。

基本操作
--------

* 直接输入任务：使用自然语言描述目标、限制条件和验收方式。
* 引用文件：输入 ``@`` 并选择文件，例如 ``请解释 @src/main.py 的启动流程``。
* 执行命令：以 ``!`` 开头，例如 ``!git status``。命令输出会加入当前会话上下文。
* 查看命令：输入 ``/help`` 打开命令面板。
* 切换模型：输入 ``/models``。
* 开始新会话：输入 ``/new``。
* 查看或切换历史会话：输入 ``/sessions``。
* 压缩长会话上下文：输入 ``/compact``。
* 撤销上一轮消息及其文件改动：输入 ``/undo``；之后可用 ``/redo`` 恢复。
* 退出 OpenCode：输入 ``/exit``，也可以使用 ``/quit`` 或 ``/q``。

建议把任务写得具体一些，例如：

.. code-block:: text

   请先运行现有测试定位失败原因，只修改与该错误直接相关的文件；修改后重新运行测试，并总结改动和剩余风险。

常用命令
--------

.. code-block:: bash

   # 查看版本和命令帮助
   opencode --version
   opencode --help

   # 在当前目录启动交互界面
   opencode

   # 在指定项目目录启动
   opencode /path/to/your/project

   # 继续最近一次会话
   opencode --continue

   # 使用指定模型启动，模型名以实际列表为准
   opencode --model sjtu-llm/模型名称

   # 非交互式执行一次任务
   opencode run "总结当前项目的目录结构，不要修改文件"

   # 列出校内 provider 配置的模型
   opencode models sjtu-llm

   # 查看已配置的模型服务凭据
   opencode auth list

   # 查看 OpenCode 最终加载的配置
   opencode debug config

   # 显示详细日志，辅助排查启动或请求错误
   opencode --print-logs --log-level DEBUG

在思源一号登录节点调用 ``opencode`` 时，包装程序会自动申请调试资源；出现排队提示时请等待资源分配。

推荐使用流程
------------

#. 进入正确的项目目录，运行 ``pwd`` 和 ``git status`` 确认工作范围与现有改动。
#. 先让 OpenCode 阅读代码并说明方案，明确要求暂不修改文件。
#. 确认方案后，再授权其修改与任务直接相关的文件。
#. 要求 OpenCode 运行对应测试、格式化或静态检查。
#. 最后使用 ``git diff`` 审查改动，不要只依赖模型给出的文字总结。

``/undo`` 和 ``/redo`` 依赖 Git 管理文件改动，因此项目需要是 Git 仓库。不要把它们作为唯一的恢复手段；开始复杂任务前仍建议先提交或备份重要改动。

常见问题
--------

* 提示 ``module: command not found``：确认当前位于 Pi 2.0 或思源一号的登录节点，并通过集群支持的方式重新登录，再执行 ``module load opencode``。
* 复制样例配置时提示路径不存在：确认使用了当前集群对应的路径。Pi 2.0 使用 ``/lustre/share/samples/opencode/opencode.jsonc``，思源一号使用 ``/dssg/share/sample/opencode/opencode.jsonc``。
* 思源一号出现 ``job queued and waiting for resources``：包装程序正在等待调试资源，资源可用后会自动继续，无需重复启动。
* 出现 ``401``、``Unauthorized`` 或 API Key 错误：运行 ``test -n "$SJTU_LLM_API_KEY" && echo OK`` 检查变量是否生效，并确认 API Key 未过期。
* 模型列表中没有校内模型：运行 ``opencode models sjtu-llm``。如果仍无结果，检查 ``~/.config/opencode/opencode.jsonc`` 是否存在，并用 ``opencode debug config`` 确认 ``sjtu-llm`` provider 已加载。
* 出现 ``404`` 或接口协议错误：校内模型配置中的 API 地址或协议可能不匹配。恢复当前集群提供的样例配置，或与管理员确认接口地址和兼容协议。
* JSONC 配置解析失败：检查逗号、引号和括号是否配对。可先从 ``~/.config/opencode/backups/`` 恢复最近一次备份，再逐项合并自定义配置。
* 需要更详细的错误信息：使用 ``opencode --print-logs --log-level DEBUG`` 启动。分享日志前务必删除 API Key、Authorization 请求头和其他敏感信息。

免责声明
--------

## 1. AI 输出性质

OpenCode 是一款基于大语言模型的 AI 编程代理，其生成的代码、建议、解释及其他输出内容均由人工智能模型自动产生，可能包含错误、不准确、不完整、存在偏见或虚构内容（即"幻觉"）。

## 2. 用户责任

**最终审核责任**：用户对 OpenCode 生成的所有输出（包括但不限于代码、命令、配置修改建议）承担全部审核、测试和验证责任。用户应独立评估输出内容的准确性、安全性、适用性和合规性，并在采纳前进行充分测试。

**操作风险自担**：OpenCode 能够修改项目文件、执行 Shell 命令、提交代码变更等。用户须对启用这些功能后的一切后果（包括数据丢失、安全漏洞、系统损坏等）承担全部责任。

**合规性责任**：用户须确保使用 OpenCode 的过程及结果符合所在机构的相关政策、法律法规及开源许可证要求。

## 3. 数据隐私与安全

使用 OpenCode 时，任务描述、引用的文件内容和必要的上下文可能会发送到所选的模型服务。用户应确保不输入涉密数据、API Key、密码及其他敏感信息。

用户对自身输入内容及数据安全负全部责任。

## 4. 无担保声明

OpenCode 按"现状"（"AS IS"）提供，不作任何明示或暗示的担保，包括但不限于对适销性、特定用途适用性、安全性及不侵权的担保。不保证服务不会中断、无错误或输出的绝对准确性。

## 5. 责任限制

在适用法律允许的最大范围内，“交我算”平台及 OpenCode 相关维护方不对以下情形承担任何责任：

- AI 生成内容中的错误、遗漏或不准确；
- 因使用或无法使用 OpenCode 而导致的任何直接、间接、附带、特殊或后果性损害；
- 因用户采纳 AI 建议而引发的代码质量、安全漏洞或合规性问题。

---

**使用 OpenCode 即表示您已阅读、理解并同意本免责声明的全部内容。**


参考资料
--------

* `OpenCode 官方文档 <https://opencode.ai/docs/zh-cn/>`__
* `上海交通大学本地大模型 API 使用文档 <https://claw.sjtu.edu.cn/guide/sjtu-api/>`__
