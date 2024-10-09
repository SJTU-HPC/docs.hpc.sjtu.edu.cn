.. _latex:

===============================
TeX Live
===============================

TeX Live 是一个完整的 TeX/LaTeX 发行版，提供了各种排版工具和宏包，适用于编写科学论文、技术报告、书籍等高质量文档。本文将介绍 TeX Live 的基本使用方法，包括基本命令以及在 Matplotlib 中使用 LaTeX 格式生成图表。

目录
===============================

- 常用命令
- 配置和更新
- 在 Matplotlib 中使用 LaTeX
- 将tex转换成pdf示例

常用命令
===============================

在思源一号上调用texlive

   .. code-block:: bash

      module load texlive

调用texlive后，可以使用以下命令处理 `.tex` 文件：

1. **pdflatex**: 将 LaTeX 源文件编译为 PDF 格式。
   
   .. code-block:: bash

      pdflatex yourfile.tex

2. **xelatex**: 使用 Unicode 编码支持的引擎编译 LaTeX 文档，支持多语言字体。

   .. code-block:: bash

      xelatex yourfile.tex

3. **lualatex**: 使用 LuaTeX 引擎编译 LaTeX 文档，具有更灵活的扩展性。

   .. code-block:: bash

      lualatex yourfile.tex


在 Matplotlib 中使用 LaTeX
===============================

Matplotlib 是一个强大的 Python 库，用于绘制高质量图表。结合 TeX Live，可以使用 LaTeX 语法在图表中渲染数学公式和文本，提升图表的专业性。以下是如何在 Matplotlib 中启用 LaTeX 支持的步骤：

1. 确保 TeX Live 已正确调用

2. 在conda环境中安装 Matplotlib 及 numpy 库：

   .. code-block:: bash

      conda install matplotlib numpy

3. 在 Python 脚本中启用 LaTeX 渲染：

   .. code-block:: python

      import matplotlib.pyplot as plt
      import numpy as np

      # 启用 LaTeX 渲染
      plt.rc('text', usetex=True)
      plt.rc('font', family='serif')

      # 创建示例数据
      x = np.linspace(0, 10, 100)
      y = np.sin(x)

      # 绘制带有 LaTeX 公式的图表
      plt.plot(x, y, label=r'$\sin(x)$')
      plt.title(r'Plot of $\sin(x)$', fontsize=16)
      plt.xlabel(r'$x$', fontsize=14)
      plt.ylabel(r'$\sin(x)$', fontsize=14)
      plt.legend()

      # 保存图表
      plt.savefig('latex_plot.pdf')

在该示例中，LaTeX 语法用于图表的标题、坐标轴标签和图例。Matplotlib 会通过调用系统中的 LaTeX 来渲染这些文本。生成的 PDF 图表具有高分辨率，适合用于出版物和论文中。

将tex转换成pdf示例
===============================

以下是一个简单的 LaTeX 示例，演示如何创建一个包含标题、段落和数学公式的文档：

.. code-block:: latex

    \documentclass{article}
    
    % Title Information
    \title{Sample TeX Document}
    \author{Your Name}
    \date{\today}  % You can also manually enter the date here, e.g., {October 8, 2024}

    \begin{document}

    % Create the title
    \maketitle

    \begin{abstract}
    This is a simple example of a LaTeX document. The document demonstrates basic usage of LaTeX formatting for creating a well-structured PDF file.
    \end{abstract}

    \section{Introduction}

    This is the introduction section. You can introduce the purpose of your document and provide any relevant background information. LaTeX is widely used in academic and scientific writing due to its powerful features for handling large documents and mathematical notation.

    \section{Main Content}

    In this section, you can add the main content of your document. You can use subsections, equations, tables, and figures as needed.

    \subsection{A Subsection Example}

    Here is an example of a subsection. You can also create lists and tables in LaTeX. For instance:

    \begin{itemize}
        \item First item
        \item Second item
        \item Third item
    \end{itemize}

    \subsection{Equation Example}

    LaTeX handles mathematical notation very well. For example, the quadratic formula is:

    \[
    x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
    \]

    \section{Conclusion}

    In the conclusion, summarize the key points of the document and any future work or steps to take. LaTeX makes it easy to organize and structure your writing clearly.

    \end{document}

将此内容保存为 `example.tex`，然后使用以下命令编译：

.. code-block:: bash

   pdflatex example.tex

编译成功后，将生成一个 `example.pdf` 文件。

更多信息
===============================

- `TeX Live 官方网站: <https://www.tug.org/texlive/>`__