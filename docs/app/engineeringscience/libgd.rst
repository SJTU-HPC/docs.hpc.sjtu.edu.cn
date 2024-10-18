.. _libgd:

=============================
libgd (GD Library) 
=============================


简介
======
libgd 是一个用于动态图形生成的跨平台函数库。它用于创建、操作图像，并且可以嵌入到PHP、Python等语言中。libgd支持多种图像格式，如PNG、JPEG、GIF等，并提供了丰富的绘图功能，如线条、多边形、曲线以及文本等。

.. note::
   libgd 不仅可以用于Web应用，也可以集成到桌面应用程序中，为用户提供图像处理的功能。

使用方法
==========
libgd 提供了广泛的API来创建和修改图像，以下是在思源1号上通过module工具调用libgd并编译示例程序：

- 制作两帧GIF动图:`test.cpp`

  .. code-block:: c

        #include <gd.h>
        #include <gdfonts.h>
        #include <cstdio>
        #include <cstring>

        int main() {
            // 创建GIF文件
            FILE *gifout = fopen("output.gif", "wb");
            if (!gifout) {
                fprintf(stderr, "无法打开文件以保存GIF。\n");
                return 1;
            }

            // 创建两帧图像
            int width = 200;
            int height = 100;

            // 创建第一帧图像
            gdImagePtr image = gdImageCreate(width, height);

            // 定义颜色
            int white = gdImageColorAllocate(image, 255, 255, 255);  // 背景白色
            int black = gdImageColorAllocate(image, 0, 0, 0);        // 文本黑色

            // 填充背景
            gdImageFilledRectangle(image, 0, 0, width - 1, height - 1, white);

            // 绘制第一帧文本
            const char *text1 = "Frame 1";
            gdFontPtr font = gdFontGetSmall();
            int x1 = (width - gdFontGetSmall()->w * strlen(text1)) / 2;
            int y1 = (height - gdFontGetSmall()->h) / 2;
            gdImageString(image, font, x1, y1, (unsigned char *)text1, black);

            // 开始GIF动画
            gdImageGifAnimBegin(image, gifout, 1, 0); // 使用全局颜色表，循环无限

            // 添加第一帧
            gdImageGifAnimAdd(image, gifout, 0, 0, 0, 0, 0, 0);

            // 创建第二帧图像
            gdImagePtr image2 = gdImageCreate(width, height);

            // 填充背景
            gdImageFilledRectangle(image2, 0, 0, width - 1, height - 1, white);

            // 绘制第二帧文本
            const char *text2 = "Frame 2";
            int x2 = (width - gdFontGetSmall()->w * strlen(text2)) / 2;
            int y2 = (height - gdFontGetSmall()->h) / 2;
            gdImageString(image2, font, x2, y2, (unsigned char *)text2, black);

            // 添加第二帧
            gdImageGifAnimAdd(image2, gifout, 0, 0, 0, 0, 0, 0);

            // 结束GIF动画
            gdImageGifAnimEnd(gifout);

            // 清理
            gdImageDestroy(image);
            gdImageDestroy(image2);
            fclose(gifout);
            printf("GIF图像已保存为output.gif\n");
            return 0;
        }


- 调用libgd并编译示例程序：

  .. code-block:: c

     module purge
     module load gcc/12.3.0 
     module load libgd/2.3.3-gcc-12.3.0

     g++ -o test test.cpp -lgd

- 执行程序：

  .. code-block:: c

    ./test


参考链接
===========
- `libgd 官方网站 <http://libgd.github.io/>`_

请根据实际的文档需求调整以上内容。如果使用的是特定编程语言（如PHP、Python），则需要相应地调整示例代码。
