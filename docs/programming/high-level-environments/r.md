# R

![R Logo](r-logo.jpg)

R is a language and environment for statistical computing and graphics.
It provides a wide variety of statistical tools, such as linear and nonlinear modelling, 
classical statistical tests, time-series analysis, classification, clustering, graphics, 
and it is highly extensible.

R provides an Open Source route to express statistical methodologies, 
it is a GNU project with similarities to the S language and environment. 
One of R's strengths is the ease with which 
well-designed publication-quality plots can be produced, including mathematical 
symbols and formulae where needed. R is an integrated suite of software facilities 
for data manipulation, calculation and graphical display. 

R users should also be interested in
the [RStudio](https://www.google.com) web integrated development
environment hosted at NERSC.

## R at NERSC

Type the following command to launch R:

    nersc$ module load R
    nersc$ R

To run R in an interactive allocation, allocate an interactive allocation and run R inside it.

    cori$ salloc -q interactive -C knl -t 234
    cori$ module load R
    cori$ R

To run R through a batch job, make a script like the following and submit it.

    #!/bin/bash
    #SBATCH -C knl
    #SBATCH -q regular
     
    module load R
    R CMD BATCH code.R

The content of code.R might look like.

    j=1;
    imagfilename = paste('myimag', j ,'.pdf',sep='');
    pdf(file=imagfilename, width = 800, height =800)
    x=1:10;
    plot(x, main='R is fun')
    dev.off()

Submitting your job script is just

    cori$ sbatch myscript.sh

## How to Run R Code in Parallel

The following program illustrates how R can be used for 'coarse-grained
parallelization', particularly useful when chunks of the computation are
unrelated and do not need to communicate in any way. The example below uses the
[package parallel](https://stat.ethz.ch/R-manual/R-devel/library/parallel/doc/parallel.pdf) to create workers as lightweight processes via forking, and
are very useful to optimize codes that use lapply, sapply, apply and related
functions:

	library("parallel")
    f = function(x)
    {
        sum = 0
        for (i in seq(1,x)) sum = sum + i
        return(sum)
     }
    n=1000
    nCores <- detectCores()
    result = mclapply(X=1:n, FUN = f, mc.cores=nCores)

## Documentation

Extensive documentation is available [online.](http://www.r-project.org/)
You may subscribe to the one or more of [R Mailing lists.](http://www.r-project.org/mail.html)
Also, find a quick [R tutorial](https://www.nersc.gov/assets/DataAnalytics/2011/TutorialR2011.pdf) 
presented at one of our NERSC User Group Meetings.
