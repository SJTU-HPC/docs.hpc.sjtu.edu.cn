# NAMD

[NAMD](www.ks.uiuc.edu/Research/namd/) is a molecular dynamics (MD)
program designed for parallel computation. Full and efficient
treatment of electrostatic and van der Waals interactions are provided
via the (O(N Log N) Particle Mesh Ewald algorithm.  NAMD interoperates
with CHARMM and X-PLOR as it uses the same force field and includes a
rich set of MD features (multiple time stepping, constraints, and
dissipatitve dynamics).

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
 * **IF** after the above you believe there is an issue with the NERSC
   module file a ticket with our [help desk](https://help.nersc.gov)
