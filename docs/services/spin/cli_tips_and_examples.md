# Tips & Examples with the Rancher CLI & Docker CLI

The Rancher CLI is used to manage your Rancher applications. The CLI is available on Cori & Edison to allow NERSC users to manage their containers, and will soon be available on Genepool.

More information on the Rancher CLI can be found at [Rancher Command Line Interface (CLI)](http://rancher.com/docs/rancher/v1.6/en/cli/) on [rancher.com](http://rancher.com). Note that NERSC provides a modified version of the Rancher CLI, and not all commands shown in the Rancher documentation are available to NERSC users.


## Practices to avoid

The following list shows things that you shouldn't do from the CLI. The commands may be a poor practice, go against NERSC policies, or in some cases may actually trigger bugs or harm the environment.

Currently, this list is short and we will add to it as our experience grows.

### Don't pipe large amounts of data through the Rancher CLI

As of Feb 2018, the Rancher CLI cannot stream large amounts of data in a pipeline and doing so can cause the CLI to become stuck. This is due to a bug, and Rancher is looking into fixing this (See Rancher issue #12165). Please avoid workflows that pipe large amounts of data through the Rancher CLI. This will cause harm, and will kill your connection, as indicated by the following error message:

    stefanl@cori11:test_db $ cat load_dept_emp.dump |  rancher exec dbtest/db mysql
    ERRO[0012] Failed to handle connection: websocket: bad handshake 
    error during connect: Get http://%2Fvar%2Ffolders%2Fg8%2Fydzygkc103x9_xt_r8zs6zyr001d77%2FT%2Fdocker-sock578594745/v1.24/exec/6e644e66b9b123123fdf4459a5b23a29f3b079307a664d8b65b68d8d0268169c/json: EOF
    stefanl@cori11:test_db $

## Don't use 'rancher run'

We generally discourage using `rancher run` and encourage you to use create an Application Stack instead. In the future, we will allow a single container to be run using 'rancher run'.

`rancher run` will let you spin up a single container. The `--name` flag requires a name to be passed in the format stackName/serviceName. 

    $ rancher run --name stefanl-webapp/web registry.spin.nersc.gov/stefanl/stefanl-test-container
    1s2872
    $ rancher ps 1s2878
    ID          NAME               IMAGE                                                    STATE     HOST      IP              DOCKER         DETAIL
    1i2553342   stefan-webapp-1   registry.spin.nersc.gov/stefanl/stefanl-test-container   running   1h2       10.42.201.186   271efe4936a4   
    $ 

Note that the command spits out the ID of the Rancher Stack, in this case '1s2872'. We can use that ID to query the status of the Stack. 

If you don't use the name stackName/serviceName, Rancher will insert the name 'Default' for you, which will cause confusion. Don't do this.

    stefanl@cori07:~ $ rancher run --name stefantestweb1 httpd
    1s3027
    stefanl@cori07:~ $ rancher ps 1s3027
    ID          NAME                       IMAGE     STATE     HOST      IP           DOCKER         DETAIL
    1i2569664   Default-stefantestweb1-1   httpd     running   1h42      10.42.8.70   d24ef37499de   
    stefanl@cori07:~ $

## How to Access Spin

### Load the Spin module to access the CLI

If you have an API key to access Spin, then simply load the Spin module like so. We'll test that the connection is good by printing out our current Rancher environment.

    stefanl@cori07:~ $ module load spin
    stefanl@cori07:~ $ rancher environment ls
    ID        NAME         ORCHESTRATION   STATE     CREATED
    1a5       dev-cattle   cattle          active    2016-10-11T01:02:27Z
    stefanl@cori07:~ $ 

### Generating API keys to access Spin

The following will generate keys for the Spin Development environment (cattle-dev). We will update these instructions to provide keys for the Spin Production environment soon.

First, a NERSC staff person will need to grant your account access to Spin.

Next, generate an API key. When promoted for a username and password, use your NERSC username & password.

    stefanl@cori06:~ $ spin-keygen.sh
    Password for user stefanl?
    Success: Spin API Key generated for stefanl.
    stefanl@cori06:~ $

The Rancher CLI stores its configuration file under your home directory, at ~/.rancher/cli.json. To verify your login information, do the following:

    stefanl@cori07:~ $ rancher config --print
    {"accessKey":"ABCDEFG123456","secretKey":"ZYXWVUT098765","url":"https://rancher.spin.nersc.gov/","environment":"1a5","path":"/global/homes/s/stefanl/.rancher/cli.json"}
    stefanl@cori07:~ $

Test that the Rancher CLI is working by printing which environments you have access to. Your account is tied to one key which has access to all environments, Prod, Dev & Sandbox.

    stefanl@cori08:~ $ rancher environment
    ID         NAME         ORCHESTRATION  STATE   CREATED
    1a736936   prod-cattle  cattle         active  2017-02-27T23:59:40Z
    1a5        dev-cattle   cattle         active  2016-10-11T01:02:27Z
    1a1221788  sandbox      cattle         active  2018-03-12T21:25:22Z
    stefanl@cori08:~ $

In the SpinUp lessons, you will be using the 'sandbox' environment. Later on, you will use the 'dev-cattle' environment. Specify the environment to be used using the RANCHER_ENVIRONMENT variable. I have two services running in the sandbox environment.

    stefanl@cori08:~ $ export RANCHER_ENVIRONMENT=sandbox
    stefanl@cori08:~ $ rancher ps
    ID      TYPE     NAME                IMAGE  STATE       SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3712  service  stefanl-webapp/web  httpd  healthy     1/1    false 
    1s3713  service  stefanl-webapp/db   mysql  healthy     1/1    false 
    stefanl@cori08:~ $

If this environment variable is not set, Rancher will ask you to select the environment for most commands:

    stefanl@cori08:~ $ rancher ps
    Environments:
    [1] prod-cattle(1a736936)
    [2] dev-cattle(1a5)
    [3] sandbox(1a1221788)
    Select: 3
    ID      TYPE     NAME                IMAGE  STATE       SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3712  service  stefanl-webapp/web  httpd  healthy     1/1    false 
    1s3713  service  stefanl-webapp/db   mysql  healthy     1/1    false 
    stefanl@cori08:~ $ 
