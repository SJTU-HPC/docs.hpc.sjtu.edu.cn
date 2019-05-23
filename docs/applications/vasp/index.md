# VASP

VASP is a package for performing ab initio quantum-mechanical
molecular dynamics (MD) using pseudopotentials and a plane wave basis
set. The approach implemented in VASP is based on a finite-temperature
local-density approximation (with the free energy as variational
quantity) and an exact evaluation of the instantaneous electronic
ground state at each MD step using efficient matrix diagonalization
schemes and an efficient Pulay mixing.

## Access

VASP is available only to NERSC users who already have an existing
VASP license.  If you have a VASP license, send your license
information to the VASP support at Vienna at
vasp.materialphysik@univie.ac.at (and copy vasp_licensing@nersc.gov)
requesting that they confirm your VASP license to NERSC staff at
vasp_licensing@nersc.gov in order to gain access to the VASP binaries
at NERSC.

!!! note
	VASP 5 requires a separate license. The VASP 4.6 users will
	**not** be automatically upgraded to VASP 5. You need to
	confirm your VASP 5 license separately.

It may take several business days from the date when your confirmation
request is sent to the VASP support at Vienna to you actually gain the
access to the VASP binaries provided at NERSC. If the confirmation
takes longer than 5 business days update the initial email thread with
the license information.

When your VASP license is confirmed, NERSC will add you to a unix file
group: `vasp5` for VASP 5 and `vasp` for VASP 4. You can check if you have
the VASP access at NERSC or not by typing the `groups` command. If you
are in the `vasp5` file group, then you can access both VASP 5 and VASP
4; if you are in the `vasp` file group, then you can access the VASP 4
only.

## Modules

To see what versions of vasp are available:
```shell
nersc$ module avail vasp
```

The vasp modules generally define two important environment variables
when loaded `PSEUDOPOTENTIAL_DIR` and `VDW_KERNAL_DIR` which are the
locations of the pseudopotential inputs and the `vdw_kernel.bindat`
file used in dispersion calculations.

## Vasp binaries

Where possible each version provides:

* `vasp_gam` - gamma point only build
* `vasp_ncl` - non-collinear spin
* `vasp_std` - the standard binary

However, some older versions of the vasp module provide:

* `gvasp` - gamma point only build
* `vasp_ncl` - non-collinear spin
* `vasp` - the standard binary

## Examples

### Basic

#### Edison

```bash
--8<-- "docs/applications/vasp/examples/edison-simple.sh"
```

#### Cori Haswell

!!!note
	The additional `--constraint=haswell` option is needed because
	Cori contains both Haswell and KNL nodes.

```bash
--8<-- "docs/applications/vasp/examples/cori-haswell-simple.sh"
```

#### Cori KNL

!!! note
	A different module is needed besides the default for the KNL
	optimized code.

!!! note
	The code will run with 66 MPI ranks per node. The final two cores
	will be dedicated to the operating system (`--core-spec=2`) to
	improve efficiency.

```bash
--8<-- "docs/applications/vasp/examples/cori-knl-simple.sh"
```
