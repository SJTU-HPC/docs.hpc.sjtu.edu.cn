#MATLAB compiler

MATLAB Compiler™ lets you share MATLAB® programs as standalone applications. 

All applications created with MATLAB Compiler use the MATLAB Runtime, which enables royalty-free deployment to users who do not need MATLAB. You can package the runtime with the application, or have your users download it during installation.

The MATLAB web site has more about [MATLAB Compiler support for MATLAB and toolboxes](https://www.mathworks.com/products/compiler/compiler_support.html), and a helpful video about [creating standalone applications](https://www.mathworks.com/videos/getting-started-standalone-applications-using-matlab-compiler-100088.html).

The following example shows a general approach on how to build MATLAB stand-alone applications from scripts. 

##Example

On Cori over NX:

```
module load matlab; module load matlab/MCRv901
```

Copy the example below to a file in your home directory and launch MATLAB

```
matlab
```

This example shows a matrix factorization, using a function that performs a Cholesky decomposition. It will read data from and write results to CSV files. This script produces an upper triangular matrix R from the diagonal and upper triangle of matrix A, satisfying the equation `R'*R=A`. `A_path` is the path to CSV-formatted data for A and `outpath` is where R will be written to as a CSV file.


```
% decomposition.m
function R = decomposition(A_path, outpath)
% A_path: Path to CSV data
% A: Matrix positive definite
% R: upper triangular matrix R from the diagonal and upper 
% triangle of matrix A, satisfying the equation R'*R=A
 A = csvread(A_path);
 R = chol(A);
 csvwrite(outpath)
```

From the command line, one can also use the MATLAB compiler as follows:

```
mkdir build; cd build

mcc -m ../decomposition.m
```

After the build process is completed, execute the program with the following syntax and the needed input arguments. For this particular example the variables are **A: Matrix positive definite** and **outpath: path to where results will be written to**:

Under

$HOME/matlab_examples/build/decomposition/for_testing if GUI was used

or

under $HOME/matlab_examples/build if mcc was used:

```
./run_decomposition.sh $MCR_ROOT $HOME/matlab_examples/data.csv ./out.csv
```

Note that the MCR_ROOT variable is set by `module load matlab/MCRv901`. It is recommended to copy these libraries to Lustre if your application is very IO intensive.

The complete documentation for mcc can be found in the [MATLAB documentation](https://www.mathworks.com/help/mps/ml_code/mcc.html).
