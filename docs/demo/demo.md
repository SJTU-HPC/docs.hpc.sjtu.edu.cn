# Lorem

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Cori performance

A table

| System Type                                   | Cray XC40   |
|-----------------------------------------------|-------------|
| Theoretical Peak Performance (System)         | 31.4 PFlops |
| Theoretical Peak Performance (Haswell nodes)  | 2.3 PFlops  |
| Theoretical Peak Performance (Xeon Phi nodes) | 29.1 PFlops |

## Basic example

!!!warning
	Include paths are specified relative to the base `docs` directory.

A useful code for checking thread and process affinity.

```C
{!demo/example/xthi.c!}
```

### Compilation

```makefile
{!demo/example/Makefile!}
```

### Running the executable

Submit the script with the `sbatch` command:

```shell
{!demo/example/run-edison.sh!}
```

!!!warning
	The `-c ` and `--cpu_bind=` options for `srun` are **required** for hybrid jobs or jobs which do not utilize all physical cores	

	
## Some source code

Instrumented C code to measure AI

```C
// Code must be built with appropriate paths for VTune include file (ittnotify.h) and library (-littnotify)
#include <ittnotify.h>

__SSC_MARK(0x111); // start SDE tracing, note it uses 2 underscores
__itt_resume(); // start VTune, again use 2 underscores

for (k=0; k<NTIMES; k++) {
 #pragma omp parallel for
 for (j=0; j<STREAM_ARRAY_SIZE; j++)
 a[j] = b[j]+scalar*c[j];
}

__itt_pause(); // stop VTune
__SSC_MARK(0x222); // stop SDE tracing
```

## LaTex support

$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$

from:

```latex
$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$
```
