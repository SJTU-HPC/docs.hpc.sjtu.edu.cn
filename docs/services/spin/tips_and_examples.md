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

    nersc:test_db $ cat load_dept_emp.dump |  rancher exec dbtest/db mysql
    ERRO[0012] Failed to handle connection: websocket: bad handshake
    error during connect: Get http://%2Fvar%2Ffolders%2Fg8%2Fydzygkc103x9_xt_r8zs6zyr001d77%2FT%2Fdocker-sock578594745/v1.24/exec/6e644e66b9b123123fdf4459a5b23a29f3b079307a664d8b65b68d8d0268169c/json: EOF
    nersc:test_db $

### Don't use 'rancher run'

We generally discourage using `rancher run` and encourage you to use
create an Application Stack instead. We are looking into uses for
`rancher run`, and may use it more in the future.

`rancher run` will let you spin up a single container. The `--name`
flag requires a name to be passed in the format stackName/serviceName.

    nersc$ rancher run --name elvis-webapp/web registry.spin.nersc.gov/elvis/nginx-myteam
    1s2872
    nersc$ rancher ps 1s2878
    ID          NAME              IMAGE                                        STATE     HOST      IP              DOCKER         DETAIL
    1i2553342   elvis-webapp-1   registry.spin.nersc.gov/elvis/nginx-myteam   running   1h2       10.42.201.186   271efe4936a4
    nersc$

Note the command spits out the ID of the Rancher Stack, in this case '1s2872'. We can use that ID to query the status of the Stack.

If you don't use the name stackName/serviceName, Rancher will insert
the name 'Default' for you, which will cause confusion. Don't do this.

    nersc$ rancher run --name elvistestweb1 httpd
    1s3027
    nersc$ rancher ps 1s3027
    ID          NAME                       IMAGE     STATE     HOST      IP           DOCKER         DETAIL
    1i2569664   Default-elvistestweb1-1   httpd     running   1h42      10.42.8.70   d24ef37499de
    nersc$

## Bugs with Rancher & Docker

Here are some bugs that we've discovered with Rancher & Docker, and
workarounds if any.

### 'rancher logs' only prints 100 lines

We've discovered that, on many stacks, 'rancher logs' will only print
100 lines of the container & service logs. We are looking into the
underlying cause, as well as a workaround.

## Accessing Spin

Spin is accessed from Cori, Edison & Denovo.

### Load the Spin module to access the CLI

If you have an API key to access Spin, then simply load the Spin module like so. Running a non-intrusive command like `rancher environment` will test that the connection is good by printing out our current Rancher environment.

    nersc$ module load spin
    nersc$ rancher environment
    ID          NAME          ORCHESTRATION   STATE     CREATED
    1a736936    prod-cattle   cattle          active    2017-02-27T23:59:40Z
    1a5         dev-cattle    cattle          active    2016-10-11T01:02:27Z
    1a1221788   sandbox       cattle          active    2018-03-12T21:25:22Z
    nersc$

### Generating API keys to access Spin

First, a NERSC staff person will need to grant your account access to
Spin. Request access through our ticketing system.

Next, generate an API key. When promoted for a username and password,
use your NERSC username & password.

    nersc$ spin-keygen.sh
    Password for user elvis?
    Success: Spin API Key generated for elvis.
    nersc$

The Rancher CLI stores its configuration file under your home
directory, at `~/.rancher/cli.json`. To verify your login information,
do the following:

    nersc$ rancher config --print
    {"accessKey":"ABCDEFG123456","secretKey":"ZYXWVUT098765","url":"https://rancher.spin.nersc.gov/","environment":"","path":"/global/homes/e/elvis/.rancher/cli.json"}
    nersc$

Test that the Rancher CLI is working by printing which environments
you have access to. Your account is tied to one key which has access
to all environments, Prod, Dev & Sandbox.

    nersc$ rancher environment
    ID         NAME         ORCHESTRATION  STATE   CREATED
    1a736936   prod-cattle  cattle         active  2017-02-27T23:59:40Z
    1a5        dev-cattle   cattle         active  2016-10-11T01:02:27Z
    1a1221788  sandbox      cattle         active  2018-03-12T21:25:22Z
    nersc$

Specify the environment to be used using the
RANCHER_ENVIRONMENT variable. In this example, I have two services running in the
sandbox environment.

    nersc$ export RANCHER_ENVIRONMENT=sandbox
    nersc$ rancher ps
    ID      TYPE     NAME                IMAGE  STATE       SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3712  service  elvis-webapp/web  httpd  healthy     1/1    false
    1s3713  service  elvis-webapp/db   mysql  healthy     1/1    false
    nersc$

If this environment variable is not set, Rancher will ask you to
select the environment for most commands:

    nersc$ rancher ps
    Environments:
    [1] prod-cattle(1a736936)
    [2] dev-cattle(1a5)
    [3] sandbox(1a1221788)
    Select: 3
    ID      TYPE     NAME                IMAGE  STATE       SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3712  service  elvis-webapp/web  httpd  healthy     1/1    false
    1s3713  service  elvis-webapp/db   mysql  healthy     1/1    false
    nersc$

## Rancher CLI Examples

### Remove a stack

Normally, you just need to use 'rancher rm StackName' to remove a
stack:

    nersc$ rancher rm elvis-first-stack
    1st1603
    nersc$

### If Removing a stack results in the error 'you don't own this volume'

If you try to remove a stack, but Rancher may refuse with an error
like 'you don't own this volume'. This is due to a bug in Rancher. In
that case, specify that you are moving a stack with the --stack flag:

    nersc$ rancher rm elvis-first-stack
    error elvis-first-stack: Bad response statusCode [401]. Status [401 Unauthorized]. Body: [message=you don't own this volume] from [https://rancher.spin.nersc.gov/v2-beta/projects/1a1221788/volumes/elvis-first-stack]
    nersc$ rancher rm elvis-first-stack --type stack
    1st1604
    nersc$

### Export the Stack configuration to your directory

    nersc:~ $ cd ~/docker/elvis-webapp
    nersc:elvis-webapp $ rancher export elvis-webapp
    INFO[0000] Creating elvis-webapp/docker-compose.yml
    INFO[0000] Creating elvis-webapp/rancher-compose.yml
    nersc:docker $ cat elvis-webapp/docker-compose.yml
    version: '2'
    services:
      web:
        image: httpd
      db:
        image: mysql
    ...
    ...
    nersc:elvis-webapp $

### Export the Stack configuration to a tar file

    nersc:~ $ cd ~/docker
    nersc:docker $ rancher export --file elvis-webapp.tar elvis-webapp
    nersc:docker $ tar tf elvis-webapp.tar
    elvis-webapp/docker-compose.yml
    elvis-webapp/rancher-compose.yml
    nersc:docker $

### View the services in your stack

    nersc$ rancher ps
    ID      TYPE     NAME                  IMAGE                                                          STATE    SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s4204  service  elvis-flask-demo/web  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  healthy  2/2    false
    1s4205  service  elvis-flask-demo/app  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v2           healthy  1/1    false
    1s4206  service  elvis-flask-demo/db   mongo:latest                                                   healthy  1/1    false
    nersc$

### View all services in your stack, including stopped containers

`rancher ps --all` will show all services in a stack. However, the stopped containers are not very obvious. The following example was taken after upgrading a service. Note that the 'SCALE' column says `2/1` which means that two containers exist, but only one is running. 

    nersc$ rancher ps --all
    ID      TYPE     NAME                  IMAGE                                                          STATE     SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3939  service  elvis-flask-demo/db   mongo:latest                                                   healthy   1/1    false
    1s3940  service  elvis-flask-demo/app  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1           upgraded  2/1    false
    1s3941  service  elvis-flask-demo/web  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  healthy   1/1    false
    nersc$

Adding the `--containers` flag will make the stopped containers more obvious:

    nersc$ rancher ps --all --containers
    ID         NAME                    IMAGE                                                          STATE    HOST  IP             DOCKER        DETAIL
    1i2596137  elvis-flask-demo-app-1  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1           running  1h83  10.42.183.166  065asd9e0a
    1i2596138  elvis-flask-demo-db-1   mongo:latest                                                   stopped  1h83  10.42.87.90    1f6920d6a1e9
    1i2596146  elvis-flask-demo-web-1  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  running  1h82  10.42.44.155   66f48c9e36ee
    1i2596160  elvis-flask-demo-db-1   mongo:latest                                                   running  1h83  10.42.90.251   065fe407ae58
    1i2596161  elvis-flask-demo-app-1  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1           running  1h83  10.42.183.175  16faa310be0a
    nersc$

### View the containers which comprise a service

Use 'rancher ps --containers' to view the containers which are part of
a service. In the example below, note that the 'web' service has two
containers.

    nersc$ rancher ps --containers | grep elvis-first-stack
    1i2576970   elvis-first-stack-app-1                    registry.spin.nersc.gov/elvis/my-first-container-app                         running   1h88      10.42.218.107   82055813959c
    1i2576980   elvis-first-stack-web-1                    nginx                                                                          running   1h88      10.42.155.46    067c52f948f8
    1i2577001   elvis-first-stack-web-2                    nginx                                                                          running   1h83      10.42.153.165   6eef89399921
    nersc$

### View the logs for services and containers

Logs may be viewed using the `rancher logs` command. The command may use the *service* name, like `elvis-first-stack/web`, or the *container* name, like 'elvis-first-stack-web-1'.

  * Remember that a service may have one or more containers (Containers are 'instances' of a service). Calling this command via the service name will show you logs for all containers in that service, if you have more than one. The individual services in the logs are noted by the presence of `01`, `02` at the beginning of the line. In the example below, notice how the line begins with a `01` or a `02` which indicates which container owns that log line.

    nersc$ rancher logs elvis-flask-demo/web
    01 2018-05-23T00:15:26.486199100Z 128.3.135.223 - - [23/May/2018:00:15:26 +0000] "GET /static/CPvalid1_nodsRNA_40x_Tiles_p1745DAPI.png HTTP/1.1" 200 82055 "http://128.55.206.22:60000/fields/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36" "-"
    02 2018-05-23T00:17:21.196355808Z 128.3.135.223 - - [23/May/2018:00:17:21 +0000] "GET /fields/ HTTP/1.1" 200 19322 "http://128.55.206.22:60000/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/53
    nersc$

To view the logs for just a single container, print the container name instead of the service name. The container name can be found using `rancher ps --containers` as shown above.

    nersc$ rancher logs elvis-flask-demo-web-2
    128.3.135.153 - - [14/Mar/2018:00:41:23 +0000] "GET / HTTP/1.1" 200 12 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36" "-"

In this next example, we are viewing logs for the last hour, with
timestamps enabled, and are following the logs as if we were using
'tail --follow':

    nersc$ rancher logs --since 1h --timestamps --follow elvis-webapp-1
    2017-11-09T01:17:38.296570056Z AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 10.42.152.36. Set the 'ServerName' directive globally to suppress this message
    2017-11-09T01:17:38.308314039Z AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 10.42.152.36. Set the 'ServerName' directive globally to suppress this message
    2017-11-09T01:17:38.355638274Z [Thu Nov 09 01:17:38.336440 2017] [mpm_event:notice] [pid 1:tid 139923965044608] AH00489: Apache/2.4.27 (Unix) configured -- resuming normal operations
    2017-11-09T01:17:38.355655838Z [Thu Nov 09 01:17:38.343553 2017] [core:notice] [pid 1:tid 139923965044608] AH00094: Command line: 'httpd -D FOREGROUND'
    ...
    ...

### Obtain a shell on a container

Use `rancher exec -it NAME /bin/bash` to start a bash shell on a container. The NAME may be the service name, or an individual container name.

    nersc$ rancher exec -it elvis-webapp-1 /bin/bash
    root@21060e7b6b52:/usr/local/apache2# ps aux
    USER        PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
    root          1  0.0  0.0  77204  2936 ?        Ss   01:17   0:00 httpd -DFOREGR
    daemon        9  0.0  0.0 366384  4144 ?        Sl   01:17   0:00 httpd -DFOREGR
    daemon       10  0.0  0.0 366384  4152 ?        Sl   01:17   0:00 httpd -DFOREGR
    daemon       11  0.0  0.0 366384  4152 ?        Sl   01:17   0:00 httpd -DFOREGR
    root         93  0.5  0.0  20240  1920 ?        Ss   01:41   0:00 /bin/bash
    root         97  0.0  0.0  17492  1144 ?        R+   01:41   0:00 ps aux
    root@21060e7b6b52:/usr/local/apache2# exit
    nersc$

### Inspect the details of a live service

'rancher inspect' will print a Service's configuration in JSON,
similar to how 'docker inspect' works.  JSON can be hard for humans to
parse, so we recommend using the the ['jq' command line tool](https://stedolan.github.io/jq/), which is
available on all NERSC systems.

    nersc$ rancher inspect elvis-webapp/web | jq
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

    nersc$ rancher inspect elvis-webapp/web | jq '.' | less

## Rancher CLI Examples

### Remove a stack

Use `rancher rm StackName` to remove a stack that you own:

    nersc$ rancher ps
    ID        TYPE      NAME                      IMAGE                                                             STATE     SCALE     SYSTEM    ENDPOINTS   DETAIL
    1s4146    service   elvis-first-stack/app   registry.spin.nersc.gov/elvis/my-first-container-app:latest     healthy   1/1       false
    1s4147    service   elvis-first-stack/web   registry.spin.nersc.gov/elvis/my-first-container-nginx:latest   healthy   2/2       false
    nersc$ rancher rm elvis-first-stack
    1st1909
    nersc$ rancher ps
    ID        TYPE      NAME      IMAGE     STATE     SCALE     SYSTEM    ENDPOINTS   DETAIL
    nersc$

### If Removing a stack results in the error 'you don't own this volume'

If you try to remove a stack and Rancher refuses with an error like 'you don't own this volume', try again and specify the name of the stack with the `--stack` flag. This error is due to an ordering bug in Rancher, and the `--stack` flag will force Rancher to look for the stack.

    nersc$ rancher rm elvis-first-stack
    error elvis-first-stack: Bad response statusCode [401]. Status [401 Unauthorized]. Body: [message=you don't own this volume] from [https://rancher.spin.nersc.gov/v2-beta/projects/1a1221788/volumes/elvis-first-stack]
    nersc$ rancher rm elvis-first-stack --type stack
    1st1604
    nersc$

### Remove unused services in your stack

This will remove services which are not listed in the docker-compose.yml file in your current working directory. We don't use this very often. Be careful with this.

    rancher prune --stack elvis-webapp

## Docker CLI examples

### Build an image and pull the latest parent images

When building an image on your laptop, use the --pull flag to ensure that your image will pull the latest parent images, if any:

    elvis@laptop:app $ docker image build --pull --tag spin-flask-demo-app .

### On your laptop, copy a file from inside a container with 'docker container cp'

To copy files from a local container on your laptop to your working directory, you can use this trick which we borrowed from Nginx. Start a temporary container on your laptop, and copy files using 'docker container cp' to your working directory:

    laptop$ docker container run --rm --detach --name tmp-nginx-container nginx
    Unable to find image 'nginx:latest' locally
    latest: Pulling from library/nginx
    e7bb522d92ff: Pull complete 
    6edc05228666: Pull complete 
    cd866a17e81f: Pull complete 
    Digest: sha256:285b49d42c703fdf257d1e2422765c4ba9d3e37768d6ea83d7fe2043dad6e63d
    Status: Downloaded newer image for nginx:latest
    df0716ebbca6692f88a6ad70d1b3476edcb864fce71827c907b4a9443dbf65bc
    laptop$ docker container cp tmp-nginx-container:/etc/nginx/nginx.conf nginx.conf
    laptop$ ls -l nginx.conf 
    -rw-r--r--  1 elvis staff  643 Dec 26 03:11 nginx.conf
    laptop$

Since the container was started with the '--rm' flag, the container will remove itself after you have stopped it.

### On Spin, copy a text file from a running container using 'cat'

    nersc:~ $ cd ~/docker/my-project
    nersc:my-project $ rancher exec -it elvis-webapp-1 cat /etc/nginx/nginx.conf > nginx.conf.copy 
    nersc:my-project $ ls -ld nginx.conf.copy 
    -rw-r--r--  1 elvis staff  1085 Dec 11 15:05 nginx.conf.copy
    nersc:my-project $

## Troubleshooting

### Stack won't upgrade because of an error like 'Failed to start: web : Service web must be state=active'

Sometimes you'll start a stack, and it won't start all of the way because of an error with one of the services in the stack.

You might try to fix it in the Compose file, and then upgrade the Stack. Suppose that upgrade fails with an error like the following:

    nersc:elvis-flask-demo $ rancher up --upgrade
    INFO[0000] Secret db.elvis-flask-demo.mongo-initdb-password already exists 
    INFO[0000] [db]: Creating                               
    INFO[0000] [app]: Creating                              
    INFO[0000] [web]: Creating                              
    INFO[0000] [web]: Created                               
    INFO[0000] [app]: Created                               
    INFO[0000] [db]: Created                                
    INFO[0000] Secret db.elvis-flask-demo.mongo-initdb-password already exists 
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
    nersc:elvis-flask-demo $

The solution here is to Stop or the problematic service, and then try the upgrade again. You may need to wait 10+ seconds, or longer, for the service to actually stop correctly.

    nersc:elvis-flask-demo $ rancher stop elvis-flask-demo/web
    1s3595
    nersc:elvis-flask-demo $ rancher up --upgrade --stack elvis-flask-demo --file ~elvis/docker/elvis-flask-demo/docker-compose.yml
    INFO[0000] Secret db.elvis-flask-demo.mongo-initdb-password already exists 
    INFO[0000] [app]: Creating                              
    INFO[0000] [db]: Creating                               
    INFO[0000] [web]: Creating                              
    INFO[0000] [web]: Created                               
    INFO[0000] [app]: Created                               
    INFO[0000] [db]: Created                                
    INFO[0000] Secret db.elvis-flask-demo.mongo-initdb-password already exists 
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
    elvis-flask-demo-app-1 | 2018-04-10T23:41:04.364630881Z [2018-04-10 23:41:04 +0000] [1] [DEBUG] Current configuration:
    elvis-flask-demo-app-1 | 2018-04-10T23:41:04.364688315Z   config: None
    ...

### Is your directory set to o+x?

Let's imagine that you started your stack, but the stack isn't working correctly. To troubleshoot, you use the 'rancher logs' command and discover the following errori which says `permission denied`.

    nersc$ rancher logs --service --follow --tail 10 elvis-flask-demo/web
    2018-04-12T22:51:19Z   0s 41599f54 ERROR elvis-flask-demo/web(1s3680) 1i2589840 service.activate.exception: Expected state running but got error: Error response from daemon: error while creating mount source path '/global/project/projectdirs/myteam/spin/elvis-flask-demo/web/nginx-proxy.conf': mkdir /global/project/projectdirs/myteam/spin/elvis-flask-demo/web/nginx-proxy.conf: permission denied

What's happening here? The Docker daemon cannot access your directory because the o+x bit is not set. Notice the part which says `mkdir /global/… permission denied`? Docker cannot see the file on the host, therefore it believes that file does not exist. By default, Docker will try to create a directory using the path provided, but does not have permission to do so. We don't actually want Docker to create anything. We just want it to use what exists already.

The real cause of this error is the lack of the 'o+x' bit on the directory. Notice how the bit is missing on the `.../elvis-flask-demo/web` subdirectory?

    nersc$ ls -ld /project/projectdirs/myteam/spin /project/projectdirs/myteam/spin/elvis-flask-demo/ /project/projectdirs/myteam/spin/elvis-flask-demo/web/
    drwxrwx--x 7 elvis myteam 512 Apr 12 14:40 /project/projectdirs/myteam/spin
    drwxrwx--x 7 elvis myteam 512 Apr 12 14:40 /project/projectdirs/myteam/spin
    drwxrwx--x 5 elvis elvis 512 Apr 12 15:06 /project/projectdirs/myteam/spin/elvis-flask-demo/
    drwxrwx--- 3 elvis elvis 512 Apr 12 14:41 /project/projectdirs/myteam/spin/elvis-flask-demo/web/
    nersc$ 

The fix is:

    nersc$ chmod o+x /project/projectdirs/myteam/spin/elvis-flask-demo/web/
    nersc$ ls -ld /project/projectdirs/myteam/spin /project/projectdirs/myteam/spin/elvis-flask-demo/ /project/projectdirs/myteam/spin/elvis-flask-demo/web/
    drwxrwx--x 7 elvis myteam 512 Apr 12 14:40 /project/projectdirs/myteam/spin
    drwxrwx--x 7 elvis myteam 512 Apr 12 14:40 /project/projectdirs/myteam/spin
    drwxrwx--x 5 elvis elvis 512 Apr 12 15:06 /project/projectdirs/myteam/spin/elvis-flask-demo/
    drwxrwx--x 3 elvis elvis 512 Apr 12 14:41 /project/projectdirs/myteam/spin/elvis-flask-demo/web/
    nersc$


