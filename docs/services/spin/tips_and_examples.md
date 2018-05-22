# Tips & Examples with the Rancher CLI & Docker CLI

The Rancher CLI is used to manage your Rancher applications. The CLI
is available on Cori & Edison to allow NERSC users to manage their
containers, and will soon be available on Genepool.

More information on the Rancher CLI can be found at
[Rancher Command Line Interface (CLI)](http://rancher.com/docs/rancher/v1.6/en/cli/)
on [rancher.com](http://rancher.com).

!!! note
	NERSC provides a modified version of the Rancher CLI,
	and not all commands shown in the Rancher documentation are
	available to NERSC users.

## Practices to avoid

The following list shows things that you shouldn't do from the
CLI. The commands may be a poor practice, go against NERSC policies,
or in some cases may actually trigger bugs or harm the environment.

Currently, this list is short and we will add to it as our experience
grows.

### Don't pipe large amounts of data through the Rancher CLI

As of Feb 2018, the Rancher CLI cannot stream large amounts of data in
a pipeline and doing so can cause the CLI to become stuck. This is due
to a bug, and Rancher is looking into fixing this (See Rancher
issue
[#12165](https://github.com/rancher/rancher/issues/12165)). Please
avoid workflows that pipe large amounts of data through the Rancher
CLI. This will cause harm, and will kill your connection, as indicated
by the following error message:

    stefanl@cori11:test_db $ cat load_dept_emp.dump |  rancher exec dbtest/db mysql
    ERRO[0012] Failed to handle connection: websocket: bad handshake
    error during connect: Get http://%2Fvar%2Ffolders%2Fg8%2Fydzygkc103x9_xt_r8zs6zyr001d77%2FT%2Fdocker-sock578594745/v1.24/exec/6e644e66b9b123123fdf4459a5b23a29f3b079307a664d8b65b68d8d0268169c/json: EOF
    stefanl@cori11:test_db $

### Don't use 'rancher run'

We generally discourage using `rancher run` and encourage you to use
create an Application Stack instead. We are looking into uses for
`rancher run`, and may use it more in the future.

`rancher run` will let you spin up a single container. The `--name`
flag requires a name to be passed in the format stackName/serviceName.

    $ rancher run --name stefanl-webapp/web registry.spin.nersc.gov/stefanl/stefanl-test-container
    1s2872
    $ rancher ps 1s2878
    ID          NAME               IMAGE                                                    STATE     HOST      IP              DOCKER         DETAIL
    1i2553342   stefan-webapp-1   registry.spin.nersc.gov/stefanl/stefanl-test-container   running   1h2       10.42.201.186   271efe4936a4
    $

Note the command spits out the ID of the Rancher Stack, in this
	case '1s2872'. We can use that ID to query the status of the Stack.

If you don't use the name stackName/serviceName, Rancher will insert
the name 'Default' for you, which will cause confusion. Don't do this.

    stefanl@cori07:~ $ rancher run --name stefantestweb1 httpd
    1s3027
    stefanl@cori07:~ $ rancher ps 1s3027
    ID          NAME                       IMAGE     STATE     HOST      IP           DOCKER         DETAIL
    1i2569664   Default-stefantestweb1-1   httpd     running   1h42      10.42.8.70   d24ef37499de
    stefanl@cori07:~ $

## Bugs with Rancher & Docker

Here are some bugs that we've discovered with Rancher & Docker, and
workarounds if any.

### 'rancher logs' only prints 100 lines

We've discovered that, on many stacks, 'rancher logs' will only print
100 lines of the container & service logs. We are looking into the
underlying cause, as well as a workaround.

## How to Access Spin

### Load the Spin module to access the CLI

If you have an API key to access Spin, then simply load the Spin
module like so. We'll test that the connection is good by printing out
our current Rancher environment.

    stefanl@cori07:~ $ module load spin
    stefanl@cori07:~ $ rancher environment ls
    ID        NAME         ORCHESTRATION   STATE     CREATED
    1a5       dev-cattle   cattle          active    2016-10-11T01:02:27Z
    stefanl@cori07:~ $

### Generating API keys to access Spin

The following will generate keys for the Spin Development environment
(cattle-dev). We will update these instructions to provide keys for
the Spin Production environment soon.

First, a NERSC staff person will need to grant your account access to
Spin.

Next, generate an API key. When promoted for a username and password,
use your NERSC username & password.

    stefanl@cori06:~ $ spin-keygen.sh
    Password for user stefanl?
    Success: Spin API Key generated for stefanl.
    stefanl@cori06:~ $

The Rancher CLI stores its configuration file under your home
directory, at ~/.rancher/cli.json. To verify your login information,
do the following:

    stefanl@cori07:~ $ rancher config --print
    {"accessKey":"ABCDEFG123456","secretKey":"ZYXWVUT098765","url":"https://rancher.spin.nersc.gov/","environment":"1a5","path":"/global/homes/s/stefanl/.rancher/cli.json"}
    stefanl@cori07:~ $

Test that the Rancher CLI is working by printing which environments
you have access to. Your account is tied to one key which has access
to all environments, Prod, Dev & Sandbox.

    stefanl@cori08:~ $ rancher environment
    ID         NAME         ORCHESTRATION  STATE   CREATED
    1a736936   prod-cattle  cattle         active  2017-02-27T23:59:40Z
    1a5        dev-cattle   cattle         active  2016-10-11T01:02:27Z
    1a1221788  sandbox      cattle         active  2018-03-12T21:25:22Z
    stefanl@cori08:~ $

In the SpinUp lessons, you will be using the 'sandbox'
environment. Later on, you will use the 'dev-cattle'
environment. Specify the environment to be used using the
RANCHER_ENVIRONMENT variable. I have two services running in the
sandbox environment.

    stefanl@cori08:~ $ export RANCHER_ENVIRONMENT=sandbox
    stefanl@cori08:~ $ rancher ps
    ID      TYPE     NAME                IMAGE  STATE       SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3712  service  stefanl-webapp/web  httpd  healthy     1/1    false
    1s3713  service  stefanl-webapp/db   mysql  healthy     1/1    false
    stefanl@cori08:~ $

If this environment variable is not set, Rancher will ask you to
select the environment for most commands:

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

## Rancher CLI Examples

### Remove a stack

Normally, you just need to use 'rancher rm StackName' to remove a
stack:

    stefanl@cori09:stefanl-first-container $ rancher rm stefanl-first-container
    1st1603
    stefanl@cori09:stefanl-first-container $

### If Removing a stack results in the error 'you don't own this volume'

If you try to remove a stack, but Rancher may refuse with an error
like 'you don't own this volume'. This is due to a bug in Rancher. In
that case, specify that you are moving a stack with the --stack flag:

    stefanl@cori09:stefanl-first-container $ rancher rm stefanl-first-container
    error stefanl-first-container: Bad response statusCode [401]. Status [401 Unauthorized]. Body: [message=you don't own this volume] from [https://rancher.spin.nersc.gov/v2-beta/projects/1a1221788/volumes/stefanl-first-container]
    stefanl@cori09:stefanl-first-container $ rancher rm stefanl-first-container --type stack
    1st1604
    stefanl@cori09:stefanl-first-container $

## Export the Stack configuration to your directory

    stefanl@cori11:docker $ rancher export stefanl-webapp
    INFO[0000] Creating stefanl-webapp/docker-compose.yml
    INFO[0000] Creating stefanl-webapp/rancher-compose.yml
    stefanl@cori11:docker $ cat stefanl-webapp/docker-compose.yml
    version: '2'
    services:
      web:
        image: httpd
      db:
        image: mysql
    ...
    ...
    $

## Export the Stack configuration to a tar file

    stefanl@cori11:docker $ rancher export --file stefanl-webapp.tar stefanl-webapp
    stefanl@cori11:docker $ tar tf stefanl-webapp.tar
    stefanl-webapp/docker-compose.yml
    stefanl-webapp/rancher-compose.yml
    stefanl@cori11:docker $

## View the services in your stack

    stefanl@cori07:my-first-container $ rancher ps -a
    1s3039    service               stefanl-first-container/web                                         nginx                                                                          healthy        2/2       false
    1s3040    service               stefanl-first-container/app                                         registry.spin.nersc.gov/stefanl/my-first-container-app                         healthy        1/1       false
    stefanl@cori07:my-first-container $

## View the containers which comprise a service

Use 'rancher ps --containers' to view the containers which are part of
a service. In the example below, note that the 'web' service has two
containers.

    stefanl@cori07:~ $ rancher ps --containers | grep stefanl-first-container
    1i2576970   stefanl-first-container-app-1                    registry.spin.nersc.gov/stefanl/my-first-container-app                         running   1h88      10.42.218.107   82055813959c
    1i2576980   stefanl-first-container-web-1                    nginx                                                                          running   1h88      10.42.155.46    067c52f948f8
    1i2577001   stefanl-first-container-web-2                    nginx                                                                          running   1h83      10.42.153.165   6eef89399921
    stefanl@cori07:~ $

## View the logs for a service

Use 'rancher logs' to view all logs for a service within a stack:

    stefanl@cori07:~ $ rancher logs stefanl-first-container-web-1
    AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 10.42.152.36. Set the 'ServerName' directive globally to suppress this message
    AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 10.42.152.36. Set the 'ServerName' directive globally to suppress this message
    [Thu Nov 09 01:17:38.336440 2017] [mpm_event:notice] [pid 1:tid 139923965044608] AH00489: Apache/2.4.27 (Unix) configured -- resuming normal operations
    [Thu Nov 09 01:17:38.343553 2017] [core:notice] [pid 1:tid 139923965044608] AH00094: Command line: 'httpd -D FOREGROUND'
    stefanl@cori07:~ $

If a service has been scaled to more than one container, this command
will let you view logs for just a single container in that service:

    stefanl@cori06:my-first-container $ rancher logs stefanl-first-container-web-2
    128.3.135.153 - - [14/Mar/2018:00:41:23 +0000] "GET / HTTP/1.1" 200 12 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36" "-"

In this next example, we are viewing logs for the last hour, with
timestamps enabled, and are following the logs as if we were using
'tail --follow':

    stefanl@cori07:~ $ rancher logs --since 1h --timestamps --follow stefan-webapp-1
    2017-11-09T01:17:38.296570056Z AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 10.42.152.36. Set the 'ServerName' directive globally to suppress this message
    2017-11-09T01:17:38.308314039Z AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 10.42.152.36. Set the 'ServerName' directive globally to suppress this message
    2017-11-09T01:17:38.355638274Z [Thu Nov 09 01:17:38.336440 2017] [mpm_event:notice] [pid 1:tid 139923965044608] AH00489: Apache/2.4.27 (Unix) configured -- resuming normal operations
    2017-11-09T01:17:38.355655838Z [Thu Nov 09 01:17:38.343553 2017] [core:notice] [pid 1:tid 139923965044608] AH00094: Command line: 'httpd -D FOREGROUND'
    ...
    ...

## Obtain a shell on a container

    stefanl@cori07:~ $ rancher exec -it stefan-webapp-1 /bin/bash
    root@21060e7b6b52:/usr/local/apache2# ps aux
    USER        PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
    root          1  0.0  0.0  77204  2936 ?        Ss   01:17   0:00 httpd -DFOREGR
    daemon        9  0.0  0.0 366384  4144 ?        Sl   01:17   0:00 httpd -DFOREGR
    daemon       10  0.0  0.0 366384  4152 ?        Sl   01:17   0:00 httpd -DFOREGR
    daemon       11  0.0  0.0 366384  4152 ?        Sl   01:17   0:00 httpd -DFOREGR
    root         93  0.5  0.0  20240  1920 ?        Ss   01:41   0:00 /bin/bash
    root         97  0.0  0.0  17492  1144 ?        R+   01:41   0:00 ps aux
    root@21060e7b6b52:/usr/local/apache2# exit
    stefanl@cori07:~ $

## Inspect the details of a live service

'rancher inspect' will print a Service's configuration in JSON,
similar to how 'docker inspect' works.  JSON can be hard for humans to
parse, so we recommend using the
the ['jq' command line tool](https://stedolan.github.io/jq/), which is
available on all NERSC systems.

    stefanl@cori11:my-first-container $ rancher inspect stefanl-webapp/web | jq
    {
      "accountId": "1a5",
      "assignServiceIpAddress": false,
      "baseType": "service",
      "createIndex": 2,
      "created": "2017-11-09T02:44:54Z",
      "createdTS": 1510195494000,
      "currentScale": 1,
      "description": null,
      "externalId": null,
      "fqdn": null,
      "healthState": "healthy",
      "id": "1s2878",
      "instanceIds": [
        "1i2553342"
      ],
    ...
    ...
    ...
    }

To save the jq output to a file, or to pipe the output through 'grep'
or 'less', be sure to apply a filter, such as '.', such as:

    stefanl@stefanl:stefanl-wordpress $ rancher inspect stefanl-webapp/web | jq '.' | less

## Rancher CLI Examples

### Remove a stack

Use `rancher rm StackName` to remove a stack that you own:

    stefanl@cori12:~ $ rancher ps
    ID        TYPE      NAME                      IMAGE                                                             STATE     SCALE     SYSTEM    ENDPOINTS   DETAIL
    1s4146    service   stefanl-first-stack/app   registry.spin.nersc.gov/stefanl/my-first-container-app:latest     healthy   1/1       false
    1s4147    service   stefanl-first-stack/web   registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest   healthy   2/2       false
    stefanl@cori12:~ $ rancher rm stefanl-first-stack
    1st1909
    stefanl@cori12:~ $ rancher ps
    ID        TYPE      NAME      IMAGE     STATE     SCALE     SYSTEM    ENDPOINTS   DETAIL
    stefanl@cori12:~ $

### If Removing a stack results in the error 'you don't own this volume'

If you try to remove a stack and Rancher refuses with an error like 'you don't own this volume', try again and specify the name of the stack with the `--stack` flag. This error is due to an ordering bug in Rancher, and the `--stack` flag will force Rancher to look for the stack.

    stefanl@cori09:stefanl-first-container $ rancher rm stefanl-first-container
    error stefanl-first-container: Bad response statusCode [401]. Status [401 Unauthorized]. Body: [message=you don't own this volume] from [https://rancher.spin.nersc.gov/v2-beta/projects/1a1221788/volumes/stefanl-first-container]
    stefanl@cori09:stefanl-first-container $ rancher rm stefanl-first-container --type stack
    1st1604
    stefanl@cori09:stefanl-first-container $

### Remove unused services in your stack

This will remove services which are not listed in the docker-compose.yml file in your current working directory. Be careful with this one.

    rancher prune --stack stefanl-webapp

## Docker CLI examples

### Build an image and pull the latest parent images

When building an image on your laptop, use the --pull flag to ensure that your image will pull the latest parent images, if any:

    stefanl@stefanl:app $ docker image build --pull --tag spin-flask-demo-app .

### On your laptop, copy a file from inside a container with 'docker container cp'

To copy files from a local container on your laptop to your working directory, you can use this trick which we borrowed from Nginx. Start a temporary container on your laptop, and copy files using 'docker container cp' to your working directory:

    stefanl@cori11:~ $ docker container run --rm --detach --name tmp-nginx-container nginx
    Unable to find image 'nginx:latest' locally
    latest: Pulling from library/nginx
    e7bb522d92ff: Pull complete 
    6edc05228666: Pull complete 
    cd866a17e81f: Pull complete 
    Digest: sha256:285b49d42c703fdf257d1e2422765c4ba9d3e37768d6ea83d7fe2043dad6e63d
    Status: Downloaded newer image for nginx:latest
    df0716ebbca6692f88a6ad70d1b3476edcb864fce71827c907b4a9443dbf65bc
    stefanl@cori11:~ $ docker container cp tmp-nginx-container:/etc/nginx/nginx.conf nginx.conf
    stefanl@cori11:~ $ ls -l nginx.conf 
    -rw-r--r--  1 stefanl  staff  643 Dec 26 03:11 nginx.conf
    stefanl@cori11:~ $

Since the container was started with the '--rm' flag, the container will remove itself after you have stopped it.

### On Spin, copy a text file from a running container using 'cat'

    stefanl@cori11:docker $ rancher exec -it stefan-webapp-1 cat /etc/nginx/nginx.conf > nginx.conf.copy 
    stefanl@cori11:docker $ ls -ld nginx.conf.copy 
    -rw-r--r--  1 stefanl  staff  1085 Dec 11 15:05 nginx.conf.copy
    stefanl@cori11:docker $

## Troubleshooting

### Stack won't upgrade because of an error like 'Failed to start: web : Service web must be state=active'

Sometimes you'll start a stack, and it won't start all of the way because of an error with one of the services in the stack.

You might try to fix it in the Compose file, and then upgrade the Stack. Suppose that upgrade fails with an error like the following:

    stefanl@cori05:stefanl-flask-demo $ rancher up --upgrade --stack stefanl-flask-demo --file ~stefanl/docker/stefanl-flask-demo/docker-compose.yml
    INFO[0000] Secret db.stefanl-flask-demo.mongo-initdb-password already exists 
    INFO[0000] [db]: Creating                               
    INFO[0000] [app]: Creating                              
    INFO[0000] [web]: Creating                              
    INFO[0000] [web]: Created                               
    INFO[0000] [app]: Created                               
    INFO[0000] [db]: Created                                
    INFO[0000] Secret db.stefanl-flask-demo.mongo-initdb-password already exists 
    INFO[0000] [web]: Starting                              
    INFO[0000] [db]: Starting                               
    INFO[0000] [app]: Starting                              
    1s3597
    ERRO[0000] Failed Starting web : Service web must be state=active or inactive to upgrade, currently: state=updating-active 
    INFO[0000] [db]: Started                                
    INFO[0000] [app]: Started                               
    1s3596
    1s3595
    ERRO[0000] Failed to start: web : Service web must be state=active or inactive to upgrade, currently: state=updating-active 
    FATA[0000] Service web must be state=active or inactive to upgrade, currently: state=updating-active 
    stefanl@cori05:stefanl-flask-demo $

The solution here is to Stop or the problematic service, and then try the upgrade again. You may need to wait 10+ seconds, or longer, for the service to actually stop correctly.

    stefanl@cori05:stefanl-flask-demo $ rancher stop stefanl-flask-demo/web
    1s3595
    stefanl@cori05:stefanl-flask-demo $ rancher up --upgrade --stack stefanl-flask-demo --file ~stefanl/docker/stefanl-flask-demo/docker-compose.yml
    INFO[0000] Secret db.stefanl-flask-demo.mongo-initdb-password already exists 
    INFO[0000] [app]: Creating                              
    INFO[0000] [db]: Creating                               
    INFO[0000] [web]: Creating                              
    INFO[0000] [web]: Created                               
    INFO[0000] [app]: Created                               
    INFO[0000] [db]: Created                                
    INFO[0000] Secret db.stefanl-flask-demo.mongo-initdb-password already exists 
    INFO[0000] [web]: Starting                              
    INFO[0000] [app]: Starting                              
    INFO[0000] [db]: Starting                               
    1s3595
    INFO[0001] Upgrading web                                
    INFO[0001] [db]: Started                                
    INFO[0001] [app]: Started                               
    INFO[0029] [web]: Started                               
    1s3597
    1s3596
    stefanl-flask-demo-app-1 | 2018-04-10T23:41:04.364630881Z [2018-04-10 23:41:04 +0000] [1] [DEBUG] Current configuration:
    stefanl-flask-demo-app-1 | 2018-04-10T23:41:04.364688315Z   config: None
    ...

### Is your directory set to o+x?

    stefanl@cori11:stefanl-flask-demo $ rancher logs --service --follow --tail 10 stefanl-flask-demo/web
    2018-04-12T22:51:19Z   0s 41599f54 ERROR stefanl-flask-demo/web(1s3680) 1i2589840 service.activate.exception: Expected state running but got error: Error response from daemon: error while creating mount source path '/global/project/projectdirs/isguser/spin/stefanl-flask-demo/web/nginx-proxy.conf': mkdir /global/project/projectdirs/isguser/spin/stefanl-flask-demo/web/nginx-proxy.conf: permission denied

What's happening here? The Docker daemon cannot access your directory because the o+x bit is not set. Notice the part which says `mkdir /global/… permission denied`? Docker cannot see the file on the host, therefore it believes that file does not exist. By default, Docker will try to create a directory using the path provided, but does not have permission to do so.

The real cause of this error is the lack of the 'o+x' bit on the directory. Notice how the bit is missing on the web directory?

    stefanl@cori08:spin $ ls -ld /project/projectdirs/isguser/spin /project/projectdirs/isguser/spin/stefanl-flask-demo/ /project/projectdirs/isguser/spin/stefanl-flask-demo/web/
    drwxrwx--x 7 stefanl isguser 512 Apr 12 14:40 /project/projectdirs/isguser/spin
    drwxrwx--x 7 stefanl isguser 512 Apr 12 14:40 /project/projectdirs/isguser/spin
    drwxrwx--x 5 stefanl stefanl 512 Apr 12 15:06 /project/projectdirs/isguser/spin/stefanl-flask-demo/
    drwxrwx--- 3 stefanl stefanl 512 Apr 12 14:41 /project/projectdirs/isguser/spin/stefanl-flask-demo/web/
    stefanl@cori08:spin $ 

The fix is:

    stefanl@cori08:spin $ chmod o+x /project/projectdirs/isguser/spin/stefanl-flask-demo/web/
    stefanl@cori08:spin $ ls -ld /project/projectdirs/isguser/spin /project/projectdirs/isguser/spin/stefanl-flask-demo/ /project/projectdirs/isguser/spin/stefanl-flask-demo/web/
    drwxrwx--x 7 stefanl isguser 512 Apr 12 14:40 /project/projectdirs/isguser/spin
    drwxrwx--x 7 stefanl isguser 512 Apr 12 14:40 /project/projectdirs/isguser/spin
    drwxrwx--x 5 stefanl stefanl 512 Apr 12 15:06 /project/projectdirs/isguser/spin/stefanl-flask-demo/
    drwxrwx--x 3 stefanl stefanl 512 Apr 12 14:41 /project/projectdirs/isguser/spin/stefanl-flask-demo/web/


