.. _r-rgl:

r-rgl
========================

简介
----------------

Provides medium to high level functions for 3D interactive graphics, including functions modelled
on base graphics (plot3d(), etc.) as well as functions for constructing representations of geometric
objects (cube3d(), etc.). Output may be on screen using OpenGL, or to various standard 3D file formats
including WebGL, PLY, OBJ, STL as well as 2D image formats, including PNG, Postscript, SVG, PGF.

完整步骤
-------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c r r-rgl
