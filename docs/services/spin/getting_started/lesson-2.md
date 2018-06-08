# Spin Getting Started Guide: Lesson 2: Run Your Application in Spin

## Lesson 2 Overview

In Lesson 1, you created an application stack comprised of two services: a
'web' service based on a lightly-customized image based on the Nginx image from
Docker Hub, and your own custom image named 'app'. Both of these images are
stored on your laptop. Lesson 2 will show you how to migrate this application
from your laptop to Spin.

Here are some things to know beforehand:

### The Rancher CLI

These lessons use the **Rancher CLI** to build and manage services and stacks
in Spin. The Rancher CLI is loaded from the Spin module on all major NERSC
systems, and activated with the command `rancher`. The Rancher CLI must be
used from a NERSC system and cannot be used from outside of NERSC, as we
maintain a modified version of the Rancher CLI which is optimized to work
with the NERSC environment. While Rancher also provides a Web Interface,
it is currently only available to NERSC Administrators.

All Rancher commands communicate with the Spin infrastructure using an API key.
You will generate an API key below.

### Spin Environments

Spin has two main environments for NERSC users:

* **dev-cattle** is for use with applications which are under development
* **prod-cattle** is used for production services

During normal development, you will first deploy your application to the
development environments, and will copy it to production when ready. Currently,
NERSC's Infrastructure Services Group (ISG) must approve all applications
before they run in the production environment.

A third environment named **sandbox** will be used exclusively if you are taking
the SpinUp sessions.

The name **cattle** refers to the container *Orchestrator* which we use to manage
containers and is part of Rancher. Rancher names many of their components with
'Ranch'-themed names, such as 'Longhorn' or 'Wagyu'. To read more information
on Rancher, please read the [Spin Getting Started Guide overview](index.md).

### Security Requirements

Note that all applications sent to Spin must follow the NERSC security requirements,
which are outlined in the [Spin Best Practices Guide](../best_practices.md). If
the application breaks one of the security requirements, Spin will refuse to
run the application and will print an error, such as in the following example:

    nersc$ rancher up
    INFO[0001] Creating stack my-first-stack
    INFO[0002] [app]: Creating
    INFO[0002] [web]: Creating
    INFO[0002] Creating service web
    INFO[0002] Creating service app
    ERRO[0002] Failed Creating web : Bad response statusCode [401]. Status [401 Unauthorized]. Body: [message=you cannot run a stack with your user: elvis] from [https://rancher.spin.nersc.gov/v2-beta/projects/1a5/scripts/transform]

## Prerequisites for Lesson 2

Before you review lesson 2, be sure to understand the concepts of [Spin Getting
Started Guide: Lesson 1: Building your first application on your laptop](lesson-1.md).

You need also need:

* SSH access on a NERSC Login node, such as cori.nersc.gov or edison.nersc.gov.
* An account on Spin. To do that, please see the [Spin Getting Started Guide: How do I get started?](index.md) Test your Spin account by quickly logging into the Spin Registry from your laptop. You will see the message 'Login Succeeded':

        elvis@laptop:~ $ docker login https://registry.spin.nersc.gov/
        Username: elvis
        Password:
        Login Succeeded
        elvis@laptop:~
* Access to a Project directory on the NERSC Global Filesystem, and the ability to run `chmod o+x` on that directory to allow the user 'nobody' to read files in that directory. This requirement is explained in detail below in [Part 2: Prepare your application to run in Spin](#part-2-prepare-your-application-to-run-in-spin).

### Generate API keys on a system such as Cori or Edison

Communication with the Rancher server requires an NERSC account and an API key
tied to that account. Follow the steps below to generate an API key.

!!! Bug
    The Rancher module does not currently work on Denovo. We will get it
    working soon.

1. Log in to a NERSC login node and load the Spin module. The Spin module loads
   the Rancher CLI and related utilities.

        elvis@laptop:~ $ ssh cori.nersc.gov
        nersc$ module load spin
        nersc$ which rancher
        /global/common/shared/isg/spin/0.1/bin/rancher
        nersc$

2. Generate the API Key using the command `spin-keygen.sh`. These keys will be
   used to access the Rancher infrastructure, and are stored under
   `~/.rancher/cli.json`.

        nersc$ spin-keygen.sh
        Password for user elvis?
        Success: Spin API Key generated for elvis.
        nersc$

3. (Optional) View your configuration file. The Rancher CLI stores its
   configuration file under your home directory, at `~/.rancher/cli.json`, and
   can be verified using the command `rancher config --print`. The command will
   print out your access key, the secret key, URL for the Rancher Server API,
   the environment used for this key (normally blank) and the path to the key
   file. The private key is used to authenticate as you to the Rancher API.
   Never share your private key.

        nersc$ rancher config --print
        {"accessKey":"ABCDEFG123456","secretKey":"ZYXWVUT098765","url":"https://rancher.spin.nersc.gov/v2-beta/schemas", "environment":"", "path":"/global/homes/e/elvis/.rancher/cli.json"}
        nersc$

4. Verify that you account is working correctly by running the command
   `rancher environment`. This command will communicate to the Rancher Server
   API using your API key. If `rancher environment` returns a list of
   environments, your account is working correctly. If the command prints an
   error such as **401 Unauthorized**, your account is not functioning. Please
   contact us for help.
     * In the following example I ran the `rancher environment` command, which printed
      out all three environments that I have access too. Environments are explained
      in greater detail in the [Overview](#lesson-2-overview) section, above.

            nersc$ rancher environment
            ID         NAME         ORCHESTRATION  STATE   CREATED
            1a736936   prod-cattle  cattle         active  2017-02-27T23:59:40Z
            1a5        dev-cattle   cattle         active  2016-10-11T01:02:27Z
            1a1221788  sandbox      cattle         active  2018-03-12T21:25:22Z
            nersc$

If everything ran successfully, you are ready to proceed.

## Part 1: Ship your image from your laptop to the Spin Registry

The Application Stack in Lesson 1 used a custom image named
'my-first-container-app' which is currently stored on your laptop. Before you
can use the container in Spin, this custom image must be shipped ("pushed") to
the Spin Registry.

A 'Registry' holds Docker images. The Spin Registry is a private registry for
NERSC users to hold Docker images which should not be available publicly, and
is available at https://registry.spin.nersc.gov/. The [Docker
Hub](https://hub.docker.com/) registry is a place to hold Docker images for the
general public, and is the default for most public images. If you have a paid
account on Docker Hub, you can also store your own private images. For more
information on registries in general, see the Docker documentation at
https://docs.docker.com/registry/introduction/.

### Step 1: Tag your 'my-first-container-app' image

Here, we will use a Docker image 'tag' to create a new name for your image to
be used in Spin. Docker image tags are used for a few purposes, such as:

* Create a friendly name for your image. If there was no name for an image, we would have to refer to it by an image ID like **ecc5b5995f8b** which isn't very friendly.
* Provide versioning information to an image, such as a version number or a date help track development versions.
* Help to locating your images on a shared, multi-user environment such as Spin. Remember that Spin is a shared resource. If everyone named their containers "my-first-container", it will be difficult for administrators to locate your specific container in Spin.
* Set a URL to a private image registry, such as the Spin Registry.

Images stored in the Spin registry will be named following a format of
**registry.spin.nersc.gov/[project]/[container name]**, where *project*
is the name of your NERSC project for collaborative projects, or your NERSC
username for individual project. We'll use your NERSC username as part of these lessons.

On your laptop, list your custom image which are stored locally:

    elvis@laptop:~ $ docker image list my-first-container-app
    REPOSITORY              TAG     IMAGE ID      CREATED            SIZE
    my-first-container-app  latest  c4f1cd0eb01c  About an hour ago  165MB
    elvis@laptop:~ $

Now tag your image with the following command, replacing *[project]* with
your NERSC username.

    docker image tag my-first-container-app registry.spin.nersc.gov/[project]/my-first-container-app

List the images again. You will see output like the following:

    elvis@laptop:~ $ docker image tag my-first-container-app registry.spin.nersc.gov/elvis/my-first-container-app
    elvis@laptop:~ $ docker image list | grep my-first-container-app
    my-first-container-app                                latest  ecc5b5995f8b  About an hour ago 165MB
    registry.spin.nersc.gov/elvis/my-first-container-app  latest  ecc5b5995f8b  About an hour ago 165MB
    elvis@laptop:~ $

Notice a few things about the list now:

* Your image has an "Image ID" like **ecc5b5995f8b**, but now has two names: One local, and a second one ready for the remote registry. The Image ID is the same for both, which means that both tags refer to the exact same image. An image ID is unique.
* The name for an image is technically called a 'Repository' in the Docker CLI.
* The URL for the Spin Registry, registry.spin.nersc.gov, is now part of the image name. For a public repository like the Docker Hub, the URL is not printed, but is set to 'docker.io' (e.g. Docker Hub).
* We didn't explicitly give this image a version tag, so Docker defaulted to the version of 'latest'. For more advanced projects, we recommend that you avoid 'latest' and instead set a version number as part of the tag.

### Step 2: Tag your 'my-first-container-nginx' image

Now, do the same thing for your 'nginx' image, replacing *[project]* with your
NERSC username. The output will be similar to the above.

    docker image tag my-first-container-nginx registry.spin.nersc.gov/[project]/my-first-container-nginx

### Step 3: Log into the Spin Registry

You need to authenticate to the private registry before using it. Usually, it's
simplest to log in to the Spin Registry before working with it. You can skip
this step, but you will asked to authenticate later while pushing the image.

From your laptop, log in to the Spin registry using your NERSC username &
password:

    elvis@laptop:~ $ docker login https://registry.spin.nersc.gov/
    Username (elvis):
    Password:
    Login Succeeded
    elvis@laptop:~ $

While some portions of Spin are restricted to NERSC & LBL networks, the Spin
Registry is available from the public Internet. This allows all NERSC users to
build and collaborate with other NERSC users.

If you cannot log into the Spin Registry, it may mean that you do not have an
account in Spin yet. If this is the case, please see the [Spin Getting Started
Guide: How do I get started?](index.md)

### Step 4: Push your images to the Spin Registry from your laptop

Once you have logged into the Spin Registry, push your image to the registry
with the following commands, replacing *[project]* with your NERSC username:

    docker image push registry.spin.nersc.gov/[project]/my-first-container-app

    docker image push registry.spin.nersc.gov/[project]/my-first-container-nginx

Each push command will print output like the following:

    elvis@laptop:~ $ docker image push registry.spin.nersc.gov/elvis/my-first-container-app
    The push refers to repository [registry.spin.nersc.gov/elvis/my-first-container-app]
    f596782719e9: Pushed
    0980b60a97f3: Pushed
    c9c93665072f: Pushed
    45f0f161f074: Layer already exists
    latest: digest: sha256:2daa3fa3c0bf26b25d4a9f393f2cdcff8065248fd279f6c4e23e7593fc295246 size: 1154
    elvis@laptop:~ $

Now the images are available in the Spin Registry and can be pulled into the Rancher environment.

## Part 2: Prepare your application to run in Spin

In this part, we'll prepare your application to run in Spin by copying required
files to your Project directory on the NERSC Global Filesystem, and modifying
your Docker Compose file to work with Spin.

Files stored on the NERSC Global Filesystem must be made available to Spin, and there are, some important considerations & groups to keep in mind:

* **Running as root is not allowed for any container which uses the NERSC
  Global Filesystem.** Containers which mount directories from the NERSC Global
  Filesystem must run under the Linux UID or UID/GID of a NERSC User Account or
  a NERSC Collaboration Account, and will need a few other security enhancements.
  This policy will be enforced by the Spin software.
* Many containers in the Docker community run as 'root' by default, even though
  this practice goes against security recommendations and [Docker's
  recommendations](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user).
* It is also possible to run the Spin containers under your own Unix username
  and group instead of a UID & GID.  We'll cover how to do this in the [Spin
  Best Practices Guide](../best_practices.md).
* The permissions for all files used by your Spin application must be readable
  by this account.
* We are looking into ways to improve this experience, and are waiting for
  improvements in the Linux Kernel and in Docker itself.

### Step 1: Make your Project directory available to Spin

We will be creating some subdirectories within your Project directory which
will store your Docker Compose file, as well as files used by your application.

The application subdirectories will be mounted inside the container by way of a
Docker **bind mount**. This means that you must set the permissions of your
Project directory to be available to Spin as follows:

* **All directories and subdirectories used by your Spin application must have
  the 'executable' bit set to allow 'other'users to browse the directory.**
  (o+x) so that the Docker daemons can successfully mount your directory. For
  example permission mode 0741 would work on a parent directory, but 0740 would
  not.
* The directories do not need to be world readable. The Docker daemon has no
 need for the 'read' or 'write' bit to be set on any directories. It only needs
 the 'executable' bit set.
* Spin will be accessing these directories as the special system user 'nobody'.
* Any files in these directories need to be readable as the user & group which
  is used within the container. In production, we encourage you to run the
  application as your NERSC Collaboration User. For these lessons, you can
  instead run as your own account, using your own UID & GID.
* Since this application will be running as your own UID & GID, make
  sure that your account can read the files.
* The Docker Compose file simply needs to be readable by your account, or
  whomever is running the `rancher` command.
* All other permissions on these directories are up to you and your team.

Head back your session on Cori, and create the Spin Project directory with a
command similar to this:

    SPIN_DIRECTORY=/global/project/projectdirs/YOUR_COLLAB_DIRECTORY/YOURUSERNAME-first-stack
    mkdir $SPIN_DIRECTORY
    mkdir $SPIN_DIRECTORY/web
    chmod o+x $SPIN_DIRECTORY $SPIN_DIRECTORY/web

And now, we copy the Nginx configuration file from your laptop to this
directory. For your own future projects, you can do this step by hand, using
Git, or whatever method your team chooses.

    cd $SPIN_DIRECTORY/web
    vim nginx-proxy.conf
    ...

In the following example, I created these files as the user **elvis**, and my
default group is **superscience**. Notice how the directories
*elvis-first-stack* and *elvis-first-stack/web* have the executable bit set for
**other**, while *nginx-proxy.conf* is only readable by me and my group.

    elvis@nersc:~ $ cd /global/project/projectdirs/superscience/spin/elvis-first-stack
    elvis@nersc:elvis-first-stack $ find . -ls
    507571599 1 drwx-----x 3 elvis superscience 512 Mar 14 17:49 .
    507571617 1 drwx-----x 2 elvis superscience 512 Mar 14 17:52 ./web
    507571673 1 -rw-r----- 1 elvis superscience 299 Mar 14 17:52 ./web/nginx-proxy.conf
    elvis@nersc:elvis-first-stack $

!!! Warning "Security reminder about public & private web services and the NERSC Global Filesystems"
    NERSC's security policy requires the following for any services which read
    and write to the NERSC Global Filesystems: Unauthenticated services, such as a
    Public webserver, are permitted to read from the Global Filesystems, but be
    sure that the access is *read only*. Public services which *write* to the
    Global Filesystems must authenticate all traffic so that you team and the
    NERSC staff can trace any write activity to an individual user or process
    if necessary. Private services, available only within NERSC can also write
    to the filesystem.

### Step 2: Modify your Docker Compose file

The Docker Compose file used on your laptop will need to be modified to work
with Spin.

In your project directory on Cori (e.g.
`/global/project/projectdirs/YOUR_COLLAB_DIRECTORY/YOURUSERNAME-first-stack`,
create a new Docker Compose file.  The name of the directory matters. By
default, Rancher will name the stack after your directory name. In these
exercises we make sure that the directory name and the stack name match just to
keep things simple. You can also specify the stack name using the `--stack`
flag.

!!! info
    Technically, the Docker Compose file can live elsewhere on the filesystem,
    such as in your home directory under `~/docker/`. The Project directory is
    just a convenient place to store this file. Some people would prefer to keep
    the configuration of an application separate from the application directory
    itself.

Add the following text to your docker-compose.yml file, but replace the values
such as *YOURUSERNAME*, *YOUR_COLLAB_DIRECTORY*, and
*YOUR_NERSC_UID:YOUR_NERSC_GID* with your specific information.

TODO fix indenting

    version: '2'
    services:
      app:
        image: registry.spin.nersc.gov/YOURUSERNAME/my-first-container-app:latest
        cap_drop:
        - ALL
      web:
        image: registry.spin.nersc.gov/YOURUSERNAME/my-first-container-nginx:latest
        ports:
        - "60000:8080"
        volumes:
        - /global/project/projectdirs/YOUR_COLLAB_DIRECTORY/YOURUSERNAME-first-stack/web/nginx-proxy.conf:/etc/nginx/conf.d/default.conf:ro
        user: YOUR_NERSC_UID:YOUR_NERSC_GID
        group_add:
        - nginx
        cap_drop:
        - ALL

Compare this Docker Compose file to the version on your laptop, and note the
major differences:

* These services use your custom images which you built and pushed to the
  Registry.
* The Nginx configuration file lives in your Project directory. The ':ro'
  parameter, specified at the very end of the line, will make the file Read
  Only. All files & directories in Spin should be Read Only whenever possible.
* Ports
    * The application will expose the external port 60000 and will map that to
      the private port of 8080, which is used by Nginx as mentioned in the
     Dockerfile from Lesson 1.
    * The port range 60000-60050 are made available for Spin development, and
      you may use any port in that range for the exercise here. These ports are
      open to the world, and bots may start scanning your app.
    * The port range 50000-50050 is also available during development, but is
      restricted to NERSC & LBL networks only (Including the JGI & LBL
      networks).
    * Sensitive ports, such as MySQL (port 3306), Postgres (5432) are
      restricted to NERSC & LBL networks only.
    * A detailed list of Ports and their accessibility can be found in the
      [Spin Best Practices Guide](../best_practices.md), under "Networking".
* Users and groups
    * The application runs as your UID & GID account, so it can access files
      outside the container. In Linux terms, this is your primary UID and the
      GID of a group which you are a member of.
    * The secondary group 'nginx' is added to the account, so the account can
      access files inside the container.
* Security:
    * All Linux kernel capabilities are dropped with the **cap_add: ALL** parameter to improve the security of the application.

!!! info "Linux Kernel 'capabilities'"
    Linux Kernel 'capabilities' are fine-grained controls over superuser
    capabilities. Docker ships with a [small, restricted set of Kernel capabilities by default,](https://docs.docker.com/engine/security/security/#linux-kernel-capabilities)
    and is fairly secure out of the box. By dropping all remaining
    capabilities, we're taking it a step further to make our container even
    more secure. If you needed to add specific capabilities back to the
    container, you can add them with the **cap_add:** parameter, which is
    discussed more in the [Spin Best Practices Guide](../best_practices.md).

### Step 3: Sanity checks

Let's verify our config file before proceeding using `rancher up --render`. If
the file is free from validation errors, Docker Compose will print out the
contents of the file, like so:

TODO fix indent

    elvis@nersc:elvis-first-stack $ ls -ld docker-compose.yml
    -rw-rw---- 1 elvis elvis 455 May 15 11:58 docker-compose.yml
    elvis@nersc:elvis-first-stack $ rancher up --render
    version: '2'
    services:
      app:
        image: registry.spin.nersc.gov/elvis/my-first-container-app:latest
        cap_drop:
        - ALL
      web:
        image: registry.spin.nersc.gov/elvis/my-first-container-nginx:latest
        ports:
        - "60050:8080"
        volumes:
        - /global/project/projectdirs/isguser/spin/elvis-first-stack/web/nginx-proxy.conf:/etc/nginx/conf.d/default.conf:ro
        user: 46311:71216
        group_add:
        - nginx
        cap_drop:
        - ALL

    elvis@nersc:elvis-first-stack $

If the file contained any validation errors, Rancher will complain with an
error similar to the following:

    elvis@nersc:elvis-first-stack $ rancher up --render
    ERRO[0001] Could not parse config for project elvis-first-stack : yaml: line 4: mapping values are not allowed in this context
    FATA[0001] Failed to read project: yaml: line 4: mapping values are not allowed in this context
    elvis@nersc:elvis-first-stack $

Note that `rancher up --render` won't catch **all** errors, but it will catch
most syntax errors.

While you're at it, quickly verify that the path to the Nginx file matches the
path specified in the Compose file. List the file, grep the path from
docker-compose.yml , and make sure they match:

    ls /global/project/projectdirs/YOUR_COLLAB_DIRECTORY/elvis-first-stack/web/nginx-proxy.conf/web/nginx-proxy.conf

    grep /global/project/projectdirs/YOUR_COLLAB_DIRECTORY/elvis-first-stack/web/nginx-proxy.conf docker-compose.yml

!!! info rancher-compose.yml
    Note that Rancher also supports a second configuration file named
    rancher-compose.yml, but that is for advanced use cases such as scaling. We may
    cover it in a future lesson.

## Part 3: Start the stack

Now that your Docker compose files are available, and all required files are
available on the NERSC Global Filesystem, it's time to start your stack, with
this command. By default, Rancher will create a stack named after your current
working directory, which should be named like **USERNAME-first-stack**. If you
want to name the stack something different, use the `--stack` flag to specify
the name.

!!! Tip "Tip: Simplify your workflow with `RANCHER_ENVIRONMENT`"

    Most Rancher commands only operate on stacks & services within one environment,
    and will need to know which environment to use. If you simply run a command
    now, Rancher will ask you to select the environment for most commands. This can
    be a hassle.

    To simplify your workflow, use the **RANCHER_ENVIRONMENT** variable to specify
    the environment to be used:

        nersc$ export RANCHER_ENVIRONMENT=dev-cattle
        nersc$ rancher ps
        ID      TYPE     NAME              IMAGE  STATE    SCALE  SYSTEM  ENDPOINTS  DETAIL
        1s3712  service  elvis-webapp/web  httpd  healthy  1/1    false
        1s3713  service  elvis-webapp/db   mysql  healthy  1/1    false
        nersc$

    If RANCHER_ENVIRONMENT is not set, Rancher will ask you to select the
    environment for most commands:

        nersc$ rancher ps
        Environments:
        [1] prod-cattle(1a736936)
        [2] dev-cattle(1a5)
        [3] sandbox(1a1221788)
        Select: 3
        ID      TYPE     NAME              IMAGE  STATE    SCALE  SYSTEM  ENDPOINTS  DETAIL
        1s3712  service  elvis-webapp/web  httpd  healthy  1/1    false
        1s3713  service  elvis-webapp/db   mysql  healthy  1/1    false
        nersc$

### Start the stack with `rancher up`

Start your stack with the `rancher up` command, using the `-d` flag to send all
logs to the background.  Make sure you use the 'sandbox' environment for this
first application.

    export RANCHER_ENVIRONMENT=sandbox
    rancher up -d

`rancher up -d` will print output like the following:

    elvis@nersc:elvis-first-stack $ export RANCHER_ENVIRONMENT=sandbox
    elvis@nersc:elvis-first-stack $ rancher up -d
    INFO[0001] Creating stack stefanl-first-stack
    INFO[0001] [app]: Creating
    INFO[0001] [web]: Creating
    INFO[0001] Creating service web
    INFO[0002] Creating service app
    INFO[0003] [app]: Created
    INFO[0003] [web]: Created
    INFO[0003] [app]: Starting
    INFO[0003] [web]: Starting
    1s4783
    INFO[0009] [app]: Started
    INFO[0016] [web]: Started
    1s4784
    elvis@nersc:elvis-first-stack $

!!! tip
    We used the `-d` flag to send our logs to the background. To send the
    application logs to the foreground instead, omit the `-d` flag. Use
    **Control-C** to stop viewing the logs and send the logs to the background.
    The application will continue running in the backgroud.

View your stacks using `rancher stack ls`. If you have more than one stack,
this will show all of them. Right now, we only have one stack.

    elvis@nersc:elvis-first-stack $ rancher stack ls
    ID        NAME                STATE     CATALOG   SERVICES   SYSTEM    DETAIL    AVAILABLE UPGRADES
    1st3323   elvis-first-stack   healthy             2          false
    elvis@nersc:elvis-first-stack $

View your services using `rancher ps`. This command will show all services in
all of your stacks.

    elvis@nersc:elvis-first-stack $ rancher ps
    ID        TYPE      NAME                    IMAGE                                                             STATE     SCALE     SYSTEM    ENDPOINTS   DETAIL
    1s4783    service   elvis-first-stack/web   registry.spin.nersc.gov/elvis/my-first-container-nginx:latest   healthy   1/1       false
    1s4784    service   elvis-first-stack/app   registry.spin.nersc.gov/elvis/my-first-container-app:latest     healthy   1/1       false
    elvis@nersc:elvis-first-stack $

!!! tip 
    Stacks can also be created using the `rancher stack create StackName` command.
    Unlike `rancher up`, `rancher stack create` requires you to specify the name of
    the stack. We use `rancher up` in our lessons.

### View the app with your browser

You may now open your the application with a browser, but we'll need the URL
first. In our classroom environment, we will use a temporary URL instead.

!!! info
    In Production, web services will normally be served from behind the Spin
    reverse-proxy service, and you would normally create a proper friendly
    hostname such as **mystack.nersc.gov** and **mystack-dev.nersc.gov** (or
    another domain, such as **lbl.gov**) and use a DNS alias (a CNAME) to point
    to the Spin loadbalancer. We'll cover that more later.

Find the FQDN and Port number for your service using `rancher inspect` along
with the [`jq` tool](https://stedolan.github.io/jq/) tool. `rancher inspect`
will print information about your stack in JSON, and `jq` makes the output
friendlier.

The example below shows how to obtain the FQDN for the stack, as well as the **ipAddress** which is part of the **publicEndpoints** array.

        nersc$ rancher inspect elvis-first-stack/web | jq '.fqdn'
        "web.elvis-first-stack.sandbox.stable.spin.nersc.org"
        nersc$ rancher inspect elvis-first-stack/web | jq '.publicEndpoints'
        [
          {
            "hostId": "1h83",
            "instanceId": "1i2601738",
            "ipAddress": "128.55.206.19",
            "port": 60000,
            "serviceId": "1s4783",
            "type": "publicEndpoint"
          }
        ]
        nersc$

Notice two important points here:

* The FQDN is a stable endpoint which will always point to your stack, and we
  recommend that you use the FQDN instead of the IP address. The IP address
  will change over time as the service moves from one node to another, while
  the FQDN will be kept up to date dynamically.
* The hostname ends with **nersc.org** instead of **nersc.gov**. Nersc.org is
  used by Spin to allow dynamic DNS updates.

Go ahead and plug the FQDN & port number into your browser to view the stack.
If the URL does not work, it's likely that the DNS records have not propogated
yet. Wait a few minutes and try again, or try the IP address instead.

## Part 4: A simple upgrade to your stack

In this example, we will perform a very simple upgrade just to show how it's
done. Upgrading a stack has two mandatory steps:

1. Upgrading the stack
2. Confirming that the upgrade was successful

We'll show Upgrading & confirming here, and will discuss more advanced upgrades
and rolling back the stack, later.

Change your application to listen on 60040 instead of port 60000. Edit the
Docker Compose file, scroll to the 'web' service and change the ports from 60000:

    ports:
    - "60000:8080"

To 60040:

    ports:
    - "60040:8080"

Use `rancher up --upgrade` to upgrade your stack. Be sure that the directory
name matches the name of your stack, or specify your stack here using `--stack
YOURUSERNAME-first-stack`:

    rancher up --upgrade -d

Notice that we're using the '-d' flag here to send the logs to the background,
like we did in ‘Part 3: Start the stack’ above.

When the stack is up, browse to your service on the new por that you used above.
The command will print output like the following. Look for the line
which says **Upgrading web**, which shows that the web service was upgraded.

    elvis@nersc:elvis-first-stack $ rancher up --upgrade -d
    INFO[0000] [app]: Creating
    INFO[0000] [web]: Creating
    INFO[0000] [app]: Created
    INFO[0000] [web]: Created
    INFO[0000] [web]: Starting
    INFO[0000] [app]: Starting
    1s3042
    INFO[0000] [app]: Started
    INFO[0000] Upgrading web
    1s3041
    INFO[0008] [web]: Started

### Confirming the upgrade

The web service was upgraded. However, the old version of the container is
still there in case you want to roll back to that version. You must remove the
old containers as soon as you are happy with the new version.

!!! Warning "Required step: Confirming the upgrade"
    The step to confirm an upgrade is required.  Certain commands, such as
    `rancher stop`, will not function until you remove the old containers.  We
    will cover this more in Lesson 3.

First, let's look at your stack in more detail.

`rancher ps --containers` will show the both sets of containers: The new
upgraded containers, and the previous non-upgraded containers. Run this
command now. Notice how there are two containers named 'something-web-1'. One is
running, and one is stopped:

    elvis@nersc:elvis-first-stack $ rancher ps --containers
    ID         NAME                     IMAGE                                                          STATE    HOST  IP             DOCKER        DETAIL
    1i2599342  elvis-first-stack-app-1  registry.spin.nersc.gov/elvis/my-first-container-app:latest    running  1h84  10.42.165.210  c099b99d60d2
    1i2599346  elvis-first-stack-web-1  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  stopped  1h87  10.42.14.161   843972812e3b
    1i2599347  elvis-first-stack-web-1  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  running  1h85  10.42.66.5     bf128471d412
    elvis@nersc:elvis-first-stack $

`rancher ps` without the `--containers` flag will subtlely show your running
and stopped containers. Run that command now. For the web service, notice that
the **STATE** column says `upgraded`, and the **SCALE** column says `2/1`
which means 'Two containers exist. One is in use.'

    elvis@nersc:elvis-first-stack $ rancher ps
    ID      TYPE     NAME                   IMAGE                                                          STATE     SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s4146  service  elvis-first-stack/app  registry.spin.nersc.gov/elvis/my-first-container-app:latest    healthy   1/1    false
    1s4147  service  elvis-first-stack/web  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  upgraded  2/1    false
    elvis@nersc:elvis-first-stack $

Now that we've seen the new and old containers, and we're satisified that the
upgrade went well, confirm that the upgrade was successful by executing the
command `rancher up -d --upgrade --confirm-upgrade`.  This will delete the old
containers.

    elvis@nersc:elvis-first-stack $ rancher up -d --upgrade --confirm-upgrade
    INFO[0002] [web]: Creating
    INFO[0002] [app]: Creating
    INFO[0003] [web]: Created
    INFO[0003] [app]: Created
    INFO[0003] [web]: Starting
    INFO[0003] [app]: Starting
    1s4147
    INFO[0004] [app]: Started
    INFO[0006] [web]: Started
    1s4146

After running this command, note that the container previously marked as
'stopped' is gone.

    elvis@nersc:elvis-first-stack $ rancher ps --containers
    ID         NAME                     IMAGE                                                          STATE    HOST  IP             DOCKER        DETAIL
    1i2599342  elvis-first-stack-app-1  registry.spin.nersc.gov/elvis/my-first-container-app:latest    running  1h84  10.42.165.210  c099b99d60d2
    1i2599347  elvis-first-stack-web-1  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  running  1h85  10.42.66.5     bf128471d412
    elvis@nersc:elvis-first-stack $

## Other ways to work with your stack

The Rancher CLI can do many things. Here are some common tasks you can do with
the Rancher CLI. A more comprehensive list can be found at [Spin: Tips & Examples](../tips_and_examples.md).

### View all of your stacks

    nersc$ rancher stacks ls
    ID       NAME                     STATE    CATALOG  SERVICES  SYSTEM  DETAIL  AVAILABLE  UPGRADES
    1st1663  elvis-mongodb          healthy  2        false
    1st1670  elvis-flask-demo-live  healthy  3        false
    1st1671  elvis-first-stack      healthy  2        false
    nersc$

### View the services in your stack

Here, we're specifying the --all flag to show all running and non-running
containers, if there are any:

    nersc$ rancher ps --all
    ID        TYPE      NAME                    IMAGE                                                           STATE     SCALE     SYSTEM    ENDPOINTS   DETAIL
    1s4783    service   elvis-first-stack/web   registry.spin.nersc.gov/elvis/my-first-container-nginx:latest   healthy   1/1       false
    1s4784    service   elvis-first-stack/app   registry.spin.nersc.gov/elvis/my-first-container-app:latest     healthy   1/1       false
    nersc$

### Scale a service in your stack

Here, we'll scale the web service from one container to two:

    nersc$ rancher scale elvis-first-stack/web=2
    elvis-first-stack/web
    nersc$ rancher ps --containers
    1i2569728 elvis-first-stack-web-1 nginx running 1h42 10.42.79.198 1c442b1d19a3
    1i2569732 elvis-first-stack-web-2 nginx running 1h2 10.42.33.67 6684fffd5270
    nersc$

### View the logs for your stack

We scaled the Web service to 2 containers above. Note how this command will
show the logs for both containers in the service, for web-1 and web-2:

    nersc$ rancher logs elvis-first-stack/web
    elvis-first-stack-web-1 | 131.243.223.227 - - [18/Jan/2018:18:12:05 +0000] "GET / HTTP/1.1" 200 12 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36" "-"
    elvis-first-stack-web-2 | 131.243.223.227 - - [18/Jan/2018:18:16:50 +0000] "GET / HTTP/1.1" 200 12 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36" "-"
    ...
    ...

### Stop your stack

Stop your stack using `rancher stop`. Use the `rancher ps` command with the
`--all` flag to show all containers, running or not.

    nersc$ rancher stop elvis-first-stack
    1st1443
    nersc$ rancher ps --all
    ID      TYPE     NAME                   IMAGE                                                          STATE     SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3748  service  elvis-first-stack/app  registry.spin.nersc.gov/elvis/my-first-container-app:latest    inactive  1/1    false
    1s3749  service  elvis-first-stack/web  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  inactive  1/1    false
    nersc$

### Start your Stack

    nersc$ rancher start elvis-first-stack
    1st1443
    nersc$ rancher ps
    ID      TYPE     NAME                   IMAGE                                                          STATE    SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3748  service  elvis-first-stack/app  registry.spin.nersc.gov/elvis/my-first-container-app:latest    healthy  1/1    false
    1s3749  service  elvis-first-stack/web  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  healthy  1/1    false
    nersc$

### Obtain a shell into a container

**Remember: Any changes that you make to a running container will be removed**
next time the container is restarted. **Containers are ephemeral by design, not
persistent.** This is one big difference between Virtual Machines and
Containers. To make a permanent change, you must change image or the Docker
Compose file. For persistent data, we use Rancher NFS volumes, mounts on the
Global Filesystem (shown above) and will provide other options in the future.

    nersc$ rancher exec -it elvis-first-stack-web-1 /bin/bash
    root@1c442b1d19a3:/# ls
    bin dev home lib64 mnt proc run srv tmp var
    boot etc lib media opt root sbin sys usr
    root@1c442b1d19a3:/# exit
    exit
    nersc$

### Removing your stack

Use `rancher rm --type stack` to remove your stack.

    nersc$ rancher ps
    ID      TYPE     NAME                   IMAGE                                                          STATE    SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3971  service  elvis-first-stack/app  registry.spin.nersc.gov/elvis/my-first-container-app:latest    healthy  1/1    false
    1s3972  service  elvis-first-stack/web  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  healthy  1/1    false

    nersc$ rancher rm elvis-first-stack --type stack
    1st1668

    nersc$ rancher ps
    ID      TYPE     NAME                   IMAGE                                                          STATE    SCALE  SYSTEM  ENDPOINTS  DETAIL
    nersc

If you forget the `--type stack` flag, Rancher may complain with an error like
*'You don't own volume your-stack-here'** This is an ordering problem within
Rancher which only occurs sometimes. Using the `--type stack` forces Rancher to
do the correct thing.

## Other useful command line tasks

For more examples of how to use the Rancher CLI, please see [Spin: Tips &
Examples](../tips_and_examples.md).

# Next Steps: Lesson 3

The next lesson, [Spin Getting Started Guide: Lesson 3: Storage, Secrets &
Managing your services](lesson-3.md) will show you how to use Rancher NFS for
persistant storage, Rancher Secrets to store files and other sensitive data,
and will also cover more advanced upgrades.
