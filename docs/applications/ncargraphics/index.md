# NCAR Graphics

## What is NCAR Graphics?

NCAR Graphics is a collection of graphics libraries that support the display of scientific data. Several interfaces are available for visualizing data:

  *  The low-level utilities (LLUs) are the traditional C and Fortran interfaces for contouring, mapping, drawing field flows, drawing surfaces, drawing histograms, drawing X/Y plots, labeling, and more.

  *  The NCAR Command Language (NCL) is a full programming language including looping and conditionals for data selection, manipulation, and display. NCL commands can be executed one at a time from the command line, or series of commands can be gathered as a script and executed in a batch mode.

Output from these interfaces can be directed in a combination of three ways: an NCGM (NCAR Graphics Computer Graphics Metafile), an X11 window, or one of many PostScript outputs (regular, Encapsulated, or Interchange PostScript format). Other formats are available through NCGM translators and accompanying device drivers. This page does not cover the NCL interface. For an excellent tutorial and greater details, see [Getting Started With NCL](http://www.ncl.ucar.edu/get_started.shtml)).

## Usage Summary

```
% module load ncl
% ncargf77
% ncargcc
```

**NOTE**: For C LLU programs please include the include files `ncarg/ncargC.h` and `ncarg/gks.h` at the top of your C program.

**NOTE**: If you want to explictly include the NCAR compile/load options on your compile line (e.g., for use in a Makefile), you can find them by loading the ncar module and then giving the command

```
% echo $NCAR
```

or

```
% echo $NCARC
```

**NOTE**: You can use the scripts ncargf90 or ncargf77 or ncargcc in the same manner that you would use f90 or xlf or cc. These scripts take care of loading the proper NCAR Graphics libraries for you. These scripts also provide various options associated with NCAR Graphics. NERSC highly recommends that you use these scripts.

## Double Precision LLU

Only the LLUs libraries ncarg and ncarg_gks are available in double precision and not the NCL.

To port programs from Crays unchanged use compiler option `-qrealsize=8`.

For double-precision C LLU programs, please include the include files `ncarg/ncargC_dp.h` and `ncarg/gks_dp.h` at the top of your C program.

## Setting the Environment

To access the current version of NCAR Graphics, load the ncar module by typing:

```
% module load ncl
```

This makes the libraries, executables, man pages and environment variables available in your current environment. It defines the variables $NCARG_ROOT, $NCAR and $NCARGKS and updates your environment variables PATH, MANPATH, and LD_LIBRARY_PATH.

If you frequently use NCAR Graphics, you can add this module command to your .login.ext and .profile.ext so that NCAR Graphics environment is set automatically every time you log in. 

### Optional environment variables

  *  NCARG_GKS_OUTPUT may be used to direct the metafile output from the NCAR Graphics GKS package to a file (use just the file name) or to pipe it to a translator (give the name of the translator, preceded by a "|")
  *  GRAPHCAP may be used to tell the NCGM translators what default "graphcap" to use.
  *  FONTCAP may be used to tell the NCGM translators what default "font" to use.
  *  DISPLAY variable is not actually used by the translators; if you are using X Windows, it determines where the translator output is to be displayed.

## NCAR Graphics commands

Following are some commonly used commands.

### Commands which facilitate running NCAR Graphics

ncl
:   Enter NCAR Command Language environment. There is no man page for ncl

ncargrun
:   Run a user program and redirect the metafile output to either ctrans or a file.

ncargf90
:   Compile and load FORTRAN user code with the LLU libraries using Fortran 90 compiler

ncargcc
:   Compile and load C user code with the LLU libraries

ng4ex
:   Utility for executing NCL example programs

ncargex
:   Utility for executing LLU example programs which have a subroutine interface.

### ncgm display, animation, and frame level editing

ctrans
:   The translator for sequential frame display

ictrans
:   An interactive version of ctrans that accepts line commands

idt
:   A Graphical User Interface (GUI) to ctrans

med
:   A NCAR Graphics metafile editor

### fontcaps and graphcaps

fcaps
:   Report available fontcaps

gcaps
:   Report available graphcaps

### Raster image display, format conversion, and image contents

ras_formats
:   Information on supported raster formats

ras_palette
:   Format descriptions for palette files for NCAR Graphics View

rascat
:   Concatenate and convert raster files

rasview
:   raster file previewer for the X Window System

rasls
:   List information on rasterfiles

rasgetpal
:   Extract the color palette of a rasterfile and write it to standard output

### ncgm filters

ncgm2cgm
:   Filter NCAR Graphics CGM to vanilla CGM

cgm2ncgm
:   Filter vanilla CGM to NCAR Graphics CGM

### Miscellaneous commands

ncargversion
:   A command which tells you what version of NCAR Graphics you are using

ncarvversion
:   A command which tells you what version of NCAR Graphics View you are using

## Low Level Utilities (LLUs)

Following is a list of Low level utilities in NCAR Graphics, with brief descriptions. The utilities described are fully documented through UNIX man pages on the supported platforms.

### NCAR Graphics LLU

areas
:   A set of routines allowing you to create an area map from a set of edges and then to use the 
    area map for various purposes.

autograph
:   To draw graphs, each with a labeled background and each displaying one or more curves.

bivar
:   Provides bivariate interpolation and smooth surface plotting for values given at irregularly 
    distributed points

colconv
:   Allows conversion among the color models RGB, HSV, HLS, and YIQ

conpack
:   Provides a sort of tool kit of Fortran subroutines that can be called in various combinations to draw 
    different kinds of contour plots.

dashpack
:   A set of routines to draw curves using dashed-line patterns that may include gap-portion specifiers, 
    solid-portion specifiers, and label-string specifiers.

ezmap
:   Allows one to plot maps of the earth according to any of ten different projections, with parallels, 
    meridians, and continental, international, and/or U.S. state outlines

gflash
:   Allows for storing segments of a picture for insertion into any subsequent picture

gridall
:   Routines for plotting various X-Y grids

histogram
:   Utility used to generate histograms showing the distribution of values in a dataset

isosurface
:   Routines for plotting an isosurface

labelbar
:   Creates a labeled, filled label bar for use as a key for a filled plot

plotchar
:   Draws characters of high, medium, or low quality

polypack
:   a set of routines allowing polygons to be manipulated in various ways

softfill
:   Routine to fill a polygonal subset of the plotter frame

scrolled_title
:   Creates movie titles

streamlines
:   Utility for plotting a streamline representation of field flow data given two arrays containing the 
    vector field components on a uniform grid

tdpack
:   Set of routines allowing for drawing representations of three-dimensional objects

vectors
:   Utility for plotting a vector field plot given two arrays containing the vector field components on a 
    uniform grid

wmap
:   A Package for Producing Daily Weather Maps and Plotting Station Model Data

## Writing a Fortran Program

### Using LLUs

NCAR Graphics provides a convenient aid for designing and developing a graphics program - see [a pictorial guide to NCAR Graphics examples](http://ngwww.ucar.edu/fund/matter/appendixD.html). You can choose an example that produces output similar to what you want, run **ncargex** for the chosen example, and then customize the resulting source code to meet your specific needs.

Follow the steps below to quickly write a Fortran graphics program.

1. Choose an NCAR Graphics example that produces output similar to what you need for visualizing your data. [A pictorial guide to NCAR Graphics examples](http://ngwww.ucar.edu/fund/matter/appendixD.html) provides prints of the example plots available. Scan the graphics shown there until you find one that appears to be closest to the type of plot you want to create. You can Click on that graphic to move to the image that shows more examples of that type.
    
2. Pick the example that is closest to the type of plot you want to create. Then type:

    ```
    % ncargex example
    ```

    where example is the name under the plot of your choice. The ncargex command puts several files in your current working directory. The file example.f is the main program that you will want to modify, and the file example.ncgm is the NCAR Graphics Computer Graphics Metafile (NCGM) that is created when the example is compiled and executed. Other files that are created when you run ncargex include an executable file named example, and occasionally, a file full of support routines for the main program. Suppose you want to do X-Y plot and you pick example **agex01**, then type:

    ```
    % ncargex agex01
    ```

    The following files are created in your working directory:

    ```
    agex01.f agex01 agex01.ncgm
    ```

    where the file agex01.f is the example agex01 program, agex01 is the executable file and agex01.ncgm is the metafile that is created when the agex01 is compiled and executed.
    
3. As a shortcut to writing your own NCAR Graphics program from scratch, modify/customize the source code of these examples for your data requirements.

## Critical NCAR Graphics Routines

NCAR uses its low-level Graphical Kernel System (GKS) package to do the actual plotting to graphic devices. NCAR's GKS supports the following output devices aka GKS workstations: CGM, Postscript, and X windows.

### Opening and Closing GKS

The first step in any low-level or high-level NCAR Graphics program is to open and initialize GKS, and specify a graphics device to plot to and the last step is to close workstation and then close GKS at the end of program.

The examples below show two ways to do this in a FORTRAN program using the low level NCAR routines.

1. The simplest way to do this is to call OPNGKS (see 'man opngks'), but this way gives you the least flexibility. For example, you cannot open multiple devices this way. Also, the default workstation opened is CGM. To deactivate the workstation and close GKS you would call CLSGKS (see 'man clsgks').

2. The second way to initialize and open graphics devices is to explicitly call the GKS commands. An example is shown below:

```
CALL GOPKS (6,0)        ! open GKS
CALL GOPWK (1,2,1)      ! open CGM workstation
CALL GACWK (1)          ! activate workstation
```

And before you exit your program, you will need to deactivate and close all workstations and then close GKS using code similar to the following:

```
CALL GDAWK (1)          ! deactivate workstation
CALL GCLWK (1)          ! close workstation
CALL GCLKS              ! Close GKS
```

NCAR can write to multiple X workstations (see 'man gopwk'). The following code segment shows how a program might look which opened up a CGM and X workstation:

```
              PROGRAM MAIN

              ... Variable declaration and initialization ...

              CALL GOPKS (6,0)        ! open GKS
              CALL GOPWK (1,2,1)      ! open CGM workstation
              CALL GACWK (1)          ! activate workstation
              CALL GOPWK (2,0,8)      ! open X workstation
              CALL GACWK (2)          ! activate workstation

      C        ... NCAR Graphics Program ...

              CALL GDAWK (1)          ! deactivate CGM workstation
              CALL GCLWK (1)          ! close CGM workstation
              CALL GDAWK (2)          ! deactivate X workstation
              CALL GCLWK (2)          ! close X workstation
              CALL GCLKS              ! close GKS
              END
```

For more information on these routines, see their respective man pages or see NCAR web documentation Workstation Functions. For more detailed information including the structure of the Fortran program, see NCAR's Writing a graphics program. To find out more information about ncargex and its options, type:

```
% man ncargex
```

### Compiling and Loading

To compile and link Fortran graphics programs with the LLU libraries, use the command ncargf90. The following illustrates the use of the command ncargf90.

To compile and load a Fortran graphics program mygraph.f, execute:

```
% ncargf90 mygraph.f
```

This command produces an executable with a default name, a.out.

Next, suppose you want to compile and load mygraph.f to:

1. generate a plot with smoothed lines using ncargf90 option -smooth and

2. assign the executable a name of your choice, say mygraph;

then execute:

```
% ncargf90 -smooth mygraph.f -o mygraph
```

This command links to smooth object files and produces an executable, mygraph. Some common options to the ncargf90 command are given below. See the online man page ncargf90 for more details.

-smooth
:   Line smoothing with character capability.

-super
:   Line smoothing with character capability and removal of crowded lines.

-quick
:   No line smoothing

-agupwrtx
:   Autograph routines with font capabilities from PWRITX

-ictrans
:   When ncargf90 is invoked with the this option, the resulting executable will, upon invocation, send 
    its metafile output to the translator ictrans. The environment variable GRAPHCAP must be set to a supported graphics output device whenever the executable is executed.

-noX11
:   Do not link in the X library when linking the code.

For detailed information on ncargf90, type:

```
% man ncargf90
```

All f90 compiler options are available through the command ncargf90.

To compile and load C NCAR Graphics programs, use ncargcc. All cc compiler options are available through the ncargcc command also.

### Execute Your Program

Unless you have specified otherwise when compiling and loading your program, you can run your program by executing:

```
% ./a.out
```

When you run your program, one of three things will happen depending on the the output device you have chosen. Either output will be displayed directly on your X Windows workstation, or an NCGM will be created or a postscript file will be created. By default, NCGM files are named gmeta and postscript files are named gmetaXX.YY where XX is the workstation id used in the GOPWK call and YY is either ps,eps,or epsi, as appropriate.

To change the default output file names see Changing metafile (NCGM) names and postscript file names

### Viewing Output

#### View plot

NCAR Graphics offers three output options for your plots. By setting the GKS workstation type, you can make your program output plots directly to your X Window on your workstation. You can also create a metafile (NCGM) or a postscript file. For details on how to choose your output option see Writing a Fortran Program. By default, metafiles (NCGMs) are named gmeta and postscript files are named gmetaXX.YY where XX is the workstation id used in the GOPWK call and YY is either ps, eps,or epsi, as appropriate.

If you choose the X Windows output option and you are connecting from a UNIX machine via SSH, your output should appear on your workstation. For more information, see Using X Applications at NERSC
Viewing a postscript file

To view a postscript file, you can download it to your system, print it on a postscript printer; or view it with tools like xv or ghostview.

#### Viewing a metafile (NCGM)

A metafile (NCGM) may be translated onto an output device supported by NCAR Graphics with one of three commands:

    * ctrans - The metafile (NCGM) translator, it provides sequential access to metafile frames
    * ictrans - A command line interface to the ctrans, it provides random access to the metafile frames.
    * idt - A graphical user interface (GUI) to ctrans.

Man pages for the above utilities are available on NERSC machines. To obtain the listing of NCAR supported devices, issue the command:

```
% gcaps
```

A description of the devices may be found by issuing the command:

```
% man graphcap
```

The output device can be specified either through the GRAPHCAP environment variable or with the -d option on the ctrans or ictrans command.

#### ctrans

Here we provide simple examples that demonstrate the basic steps for using the metafile (NCGM) translator ctrans.

Below are the steps for sequentially displaying the contents of metafile gmeta on two different devices. If you are working on a Tektronics 4107 graphics terminal, you can view gmeta by typing either:

```
% setenv GRAPHCAP t4107
% ctrans gmeta
```

or

```
% ctrans -d t4107 gmeta
```

On a workstation running the X Window System:

```
% setenv GRAPHCAP X11 
% ctrans gmeta
```

or

```
% ctrans -d X11 gmeta
```

You can sequentially advance through the metafile gmeta by pressing the RETURN key after each frame is plotted. In a window-based environment, such as X11, make sure that the mouse sprite is in the graphics window created by ctrans. Also, in a window-based environment, clicking the left mouse button has the same effect as pressing RETURN. You can terminate processing at any time by sending ctrans an interrupt signal (typing CONTROL-c on most systems).

#### ictrans


ictrans provides random access to the frames contained in the metafile. You invoke ictrans by typing:

```
% ictrans gmeta
```

As with ctrans you need to specify the output device you want to translate to as:

```
% ictrans -d t4107 gmeta
```

or

```
% setenv GRAPHCAP t4107 
% ictrans gmeta
```

Upon invocation, ictrans responds with an account of the number of frames contained in gmeta followed by ready prompt:

```
ictrans> 1p
```

Typing 1p directs the translator to plot frame 1. When the plotting is complete, a <READY> prompt appears; you must press RETURN in your terminal window before a new ictrans> prompt will be displayed.

Many ictrans commands are available, you can find out about other commands by typing:

```
ictrans> h
```

All ictrans commands can be abbreviated to the shortest unique string, just as the h stands for help. To exit ictrans, type:

```
ictrans>q
```

#### Using idt


idt is a Graphical User Interface to the ctrans translator. Using point-and-click, you may easily view metafiles in either forward or backward directions and animate to rapidly display metafile frames in sequence. It may only be used on a workstation (or X terminal) supporting the X11 windowing system. You need to set the GRAPHCAP and DISPLAY environment variables. Invoke idt by typing:

```
% idt gmeta
```

Two idt windows will come up. Use the window with the pushbutton controls to plot your frames. 

## Related Documentation

The following is a list of related Web documentation for NCL:

   * [NCL Tools Documentation](http://www.ncl.ucar.edu/Document/Tools/)
   * [NCL Reference Manual](http://www.ncl.ucar.edu/Document/Manuals/Ref_Manual/)
   * [Getting Started using NCL](http://www.ncl.ucar.edu/Document/Manuals/Getting_Started/)
   * [Category List of NCL Applications](http://www.ncl.ucar.edu/Applications/)

## Appendix A: Changing Output File Names

### Changing metafile names

Metafiles (NCGMs) are created by an application program that calls the NCAR graphics routines. By default a metafile is named gmeta. You may wish to give a metafile a different name. Here are three different ways to do so.

1. Set the environment variable, NCARG_GKS_OUTPUT, to the desired metafile (NCGM) name prior to running your program. NCAR Graphics will use this name when it creates the metafile. E.g. to change the metafile name to plot01.cgm for your program myplot, type:
    
    ```
    % setenv NCARG_GKS_OUTPUT plot01.cgm
    % ./myplot
    ```

2. Use the NCAR graphics script ncargrun before you run your program. E.g. to change the metafile name to plot01.cgm for your program named myplot type:

    ```
    % ncargrun -o plot01.cgm myplot
    ```

    For more information see the ncargrun man page.
    
3. Set the metafile (NCGM) name from within your Fortran code. This method allows you to create more than one metafile in the same job. The following piece of code will open two metafiles with the names plot01.cgm and plot02.cgm.

```     
       CALL NGSETC('ME','plot01.cgm')
       CALL GOPWK (1,2,1)
       CALL GACWK (1)

       ...

       CALL GDAWK (1)
       CALL GCLWK (1)

       CALL NGSETC('ME','plot02.cgm')
       CALL GOPWK (1, 2, 1)
       CALL GACWK (1)

        ...

        CALL GDAWK (1)
        CALL GCLWK (1)    
```

### Changing postscript file names

The postscript file names are created by an application program that calls the NCAR Graphics routines. By default, postscript files are named gmetaXX.YY where XX is the workstation id used in the GOPWK call and YY is either ps, eps,or epsi, as appropriate. To distinguish more than one postscript file names from each other, it may be necessary to assign them names that are different from default name. Here are three ways to change the name of the postscript file name for an LLU program from the default name:

1. The first way involves setting an environment variable NCARG_GKS_PSOUTPUT to the desired postscript file name prior to running your program. NCAR Graphics will use this name when it creates the postscript file. E.g. to change the postscript file name to plot01.ps for your program myplot, type:

    ```
    % setenv NCARG_GKS_PSOUTPUT plot01.ps
    % ./myplot
    ```

2. The second way is to use NCAR Graphics script ncargrun that sets the environment variable prior to running your program. E.g. to change the postscript file name to plot01.ps for your program named myplot type:

    ```
    % ncargrun -o plot01.ps myplot
    ```

    For more information see ncargrun man page.
    
3. The third way allows for dynamically changing the postscript file name from within your Fortran code, and creating several different postscript files from within the same job. To open two postscript files with the names plot01.ps and plot02.ps, do the following:

```
       CALL NGSETC('ME','plot01.ps')
       CALL GOPWK (1,2, NGPSWK('PS','LAND','COLOR'))
       CALL GACWK (1)

       ...

        CALL GDAWK (1)
        CALL GCLWK (1)

        CALL NGSETC('ME','plot02.ps')
        CALL GOPWK (1, 2,  NGPSWK('PS','LAND','COLOR'))
        CALL GACWK (1)

        ...

        CALL GDAWK (1)
        CALL GCLWK (1)
```    