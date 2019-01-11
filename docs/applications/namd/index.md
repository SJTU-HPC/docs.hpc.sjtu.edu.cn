# NAMD

[NAMD](www.ks.uiuc.edu/Research/namd/) is a molecular dynamics (MD)
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

??? example "Edison"
    ```shell
    --8<-- "docs/applications/namd/edison.sh"
    ```

## Support

 * [User's Guide](https://www.ks.uiuc.edu/Research/namd/current/ug/)
 * [Problem with NAMD?](https://www.ks.uiuc.edu/Research/namd/bugreport.html)

!!! tip
	If *after* the checking the above you believe there is an
	issue with the NERSC module file a ticket with
	our [help desk](https://help.nersc.gov)
