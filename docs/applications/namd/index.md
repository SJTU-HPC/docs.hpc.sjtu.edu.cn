# NAMD

[NAMD](https://www.ks.uiuc.edu/Research/namd/) is a molecular dynamics (MD)
program designed for parallel computation.

## Example run script

This script requires that `${INPUT_FILE}` be specified. The script is written
such that *only* the number of nodes needs to be changed.

!!! example "Cori KNL"
	```bash
	--8<-- "docs/applications/namd/cori-knl.sh"
	```

??? example "Cori Haswell"
    ```shell
    --8<-- "docs/applications/namd/cori-hsw.sh"
    ```

### SMP

An SMP build is also provided with the `namd/$version-smp` module for
some versions. The primary use of the SMP builds is to
[reduce memory usage](https://www.ks.uiuc.edu/Research/namd/wiki/index.cgi?NamdMemoryReduction).

!!! tip 
	It is recommended to use the non-SMP builds where possible as
	those typically provide the best performance.

??? example "Cori KNL SMP"
	```bash
	--8<-- "docs/applications/namd/cori-knl-smp.sh"
	```

## Support

 * [User's Guide](https://www.ks.uiuc.edu/Research/namd/current/ug/)
 * [Problem with NAMD?](https://www.ks.uiuc.edu/Research/namd/bugreport.html)

!!! tip
	If *after* the checking the above you believe there is an
	issue with the NERSC module file a ticket with
	our [help desk](https://help.nersc.gov)
