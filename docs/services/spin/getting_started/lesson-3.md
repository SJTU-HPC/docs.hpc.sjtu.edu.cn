# Spin Getting Started Guide: Lesson 3: Storage, Secrets & Managing your services

## Lesson 3 overview: Storage, Secrets & Upgrading your Stack

In Lessons 1 & 2, we built a simple application an ran it in Spin. In
Lesson 3, we will show you how to use Rancher NFS for persistent
storage, Rancher Secrets to store sensitive information such as
passwords and private keys, and use Docker's *Build, Ship, Run*
workflow to apply changes to our live stacks.

We encourage you to build your own custom stack for these
exercises. If you don't have your own stack right now, we provide an
example stack below which you can build for a reference.

### Prerequisites for Lesson 3

* Lesson 3 has the same prerequisites as Lessons 2 & 1.
* The lessons below are intended to be run with a custom stack that
  you have designed. If you don't have a custom stack right now, we
  provide an example stack for you.

## Rancher features

### Persistent data

Remember that containers are ephemeral by design. This means that once
a container is removed, all data within that container is gone
forever. Some containers, like web containers, usually don't mind if
their data goes away; but database containers usually need their data
to remain persistent.

Services which depend on persistent data must store their data on
persistent storage, or on an external Docker volume. In Lesson 2, we
showed how to store persistent data by mounting a project directory as
a volume into your containers. The NERSC Global File System may not be
ideal for all workloads, so we also provide the Rancher NFS filesystem
which is a high performance filesystem dedicated to Spin and isn't
affected by NERSC Global File System maintenance periods. Unlike using
the Project filesystems, Rancher NFS doesn't have the same
requirements for a UID/GID or directory permissions, and may be more
suited for standard, non-custom containers.

We will be expanding Spin to use a larger and faster storage system
for persistent storage in mid-2018. The system may use Rancher NFS, or
may use a similar storage technology.

[Example 3](#example-3-upgrading-to-use-rancher-nfs) shows how to use
Rancher NFS to provide persistent storage for a container running
MongoDB.

### Passwords and other sensitive information

All sensitive data for a container, such as passwords and private
certificates, should be stored in a Secret instead of in Docker
environment variables. This is because Docker environment variables
are all stored as plain text inside of a Docker Compose file and in
the container environment. You might want to share your Docker Compose
file, but you can't do that if it contains a password.

Secrets are only available to Stacks which are owned by you or your
group.

## Our example stack

The examples below will use an example stack provided by NERSC, which
is similar to the stack used in our video demonstration on
the [Spin page](../index.md).

[Lesson 2](lesson-2.md) already showed you how to build a simple
application for Spin, so we don't cover that here too much. The
purpose of this stack is to demonstrate important Spin features, and
thus the documentation of this stack is light. If you already have
your own custom stack, or want to build your own stack, feel free to
simply use this stack as a reference for your own application.

!!! Important
    The examples below will use one of the four Docker
    Compose files provided in the directory. To use one of these
    files, use the Rancher CLI with the `--file` flag, like so:

        rancher up --file docker-compose.yml.exampleX

### Create the Project Directory

First, create a Project directory to store the files, and clone the
Git repository into the working directory:

    WORKDIR=/global/project/projectdirs/YOUR_PROJECT/YOUR_SPIN_WORKDIR
    mkdir $WORKDIR
    cd $WORKDIR
    git clone https://github.com/stefanlasiewski/spin-flask-demo.git .

For each file, be sure to change all placeholders with your specific
information, including:

* Replace *YOURUSERNAME* with your NERSC username
* Replace *YOUR_UID* & *YOUR_GID* with your own NERSC UserID & GroupID
* Replace *YOUR_PROJECT* & *YOUR_SPIN_WORKDIR* with your project &
  working directory.

Change the permissions of the directories used for this Spin project
to include the `o+x` bit, including the parent working directory. As
explained in Lesson 2, this is required so that the Spin daemons can
access files inside your subdirectories. The permissions on your
directory should show the `o+x` bit set, like so:

    elvis@nersc:~ $ cd /global/project/projectdirs/superscience/spin/elvis-flask-demo/
    elvis@nersc:elvis-flask-demo $ ls -ld . web/ web/images
    drwxrwx--x  5  elvis  elvis  131072  May 14 11:28  .
    drwxrwx--x  3  elvis  elvis  512     May 11 16:03  web/
    drwxrwx--x  2  elvis  elvis  131072  May 11 16:03  web/images
    elvis@nersc:elvis-flask-demo $

### Examine the Stack

The directory contains four different Docker Compose files to
demonstrate certain features. Take a look at
`docker-compose.yml.example1`, and notice that the stack consists of
three services:

* **app**: An application service which serves dynamic code and works
  with a Mongo container.
* **db**: A Mongo service to host the data.
    * In example 1, the container stores all data inside the
      container. Example 3 shows how to upgrade the service to use
      Rancher NFS.
* **web** A webserver service which serves static images from the
  filesystem. This Web service uses the `my-first-container-nginx`
  image which we built in the previous exercise, which still uses the
  Project filesystem.
    * The Web service mounts two volumes from the NERSC Global
      Filesystem.
        * The first volume places a custom Nginx configuration file to
          `/etc/nginx/conf.d/default.conf`. This configuration file
          sets up a simple Reverse Proxy to the `app` container.
        * The second volume mounts a set of images stored on the
          Global Filesystem into the container at `/srv` . Nginx
          serves the images from this directory.
    * This image runs with your UID & GID so that it can read the
      files from the Global Filesystem. The `nginx` group is added as
      a secondary group so that your user has permissions to do things
      inside the container.
* Each service drops as many capabilities as possible to improve the
  security of the container. The Web & App services drop all Kernel
  capabilities, while db container drops all, and then adds a few
  necessary capabilities back.
* Notice that this particular Docker Compose file uses a plain text
  password. As mentioned earlier, you shouldn't normally keep a plain
  text password in the Compose file. We will upgrade the stack to use
  Secrets below.

### Build your app image

This stack uses a custom application which must be built. To do that,
**head back to your laptop** and clone the repo:

    cd ~/docker/
    git clone https://github.com/stefanlasiewski/spin-flask-demo.git
    cd spin-flask-demo/app

Now, build the app and push it to the repo. Be sure to replace
**YOURUSERNAME** with your NERSC username.

The following command will build the image, while simultaneously
tagging the image for the registry. Also notice that we're tagging
this image with the `:v1` tag instead of relying on the default
:latest tag. Later on, we'll build a new image and will tag it as
`:v1`.

    docker image build --tag registry.spin.nersc.gov/YOURUSERNAME/spin-flask-demo-app:v1 .
    docker image push registry.spin.nersc.gov/YOURUSERNAME/spin-flask-demo-app:v1

If all of these commands were successful, the image is now available
in the Spin registry and can be used by the applications below.

## Example 1: Start the stack

Start the stack, and use the `--file` flag to call our special Docker
Compose file for **example 1**.

    rancher up -d --file docker-compose.yml.example1

You should see output similar to this. Look for the lines at the end
which say something like 'INFO[0010] [service]: Started'.

    elvis@nersc:elvis-flask-demo $ rancher up -d --stack elvis-flask-demo --file docker-compose.yml.example1
    INFO[0000] Creating stack elvis-flask-demo
    INFO[0000] [db]: Creating
    INFO[0000] [app]: Creating
    INFO[0000] [web]: Creating
    INFO[0000] Creating service db
    INFO[0000] Creating service web
    INFO[0000] Creating service app
    INFO[0001] [db]: Created
    INFO[0001] [web]: Created
    INFO[0002] [app]: Created
    INFO[0002] [db]: Starting
    INFO[0002] [app]: Starting
    INFO[0002] [web]: Starting
    1s3684
    INFO[0008] [app]: Started
    INFO[0010] [db]: Started
    1s3683
    INFO[0010] [web]: Started
    1s3685
    elvis@nersc:elvis-flask-demo $

### Look at your stack in a browser

Before you open a browser and check out the end result, you'll need to
find the FQDN & port number for your stack using the `rancher inspect`
& `jq` utilities like we did in lesson 2.

In the following example, we are inspecting the `web` container in the
`elvis-flask-demo` stack. From this output, I know my web service is
available at
`https://web.elvis-flask-demo.dev-cattle.stable.spin.nersc.org:8080/`. The
web service is also available at an IP address, but note that the IP
address may change from time to time.

    elvis@nersc:elvis-flask-demo $ rancher inspect elvis-flask-demo/web | jq '.fqdn'
    "web.elvis-flask-demo.dev-cattle.stable.spin.nersc.org"
    elvis@nersc:elvis-flask-demo $ rancher inspect elvis-flask-demo/web | jq '.publicEndpoints'
    [
      {
        "hostId": "1h83",
        "instanceId": "1i2589831",
        "ipAddress": "128.55.206.19",
        "port": 8080,
        "serviceId": "1s3677",
        "type": "publicEndpoint"
      }
    ]
    elvis@nersc:elvis-flask-demo $

If everything went well, you should see a screen similar to the
following:

![web.elvis-flask-demo.dev-cattle.stable.spin.nersc.org.png](../Images/web.elvis-flask-demo.dev-cattle.stable.spin.nersc.org.png)

## Upgrading your stack with the "Build, Ship, Run" workflow

Most changes to your application stack will be applied using a concept
called **upgrading**, which follows the Docker workflow of **"Build,
Ship, Run"** with a few steps in between. Reasons for upgrading tasks
could include:

* Instructing Docker Compose to do something differently
* Pushing out a new version of your Docker image

We'll provide two examples here to show you how to upgrade your
stack. The first will make a change via the Docker Compose file, and
the second will make a change to the Docker Image which is then
implemented via the Docker Compose file.

The general steps for upgrading are as follows, and are covered in
more detail below. Some steps are not necessary every single time, and
may be skipped.

* **Build** the application, or use an existing image.

    * **Download** the configuration to your local directory using
      `rancher export`, if you don't have the files already.
    * **Modify** your application. Depending on your changes, you may
      simply be modifying the Docker Compose files, or may instead be
      changing the Dockerfile to create an updated Docker Image.
    * **Test** the application on your laptop.

* **Ship** the new images to the registry, if necessary.
* **Run** the application stack.

    * **Apply** the changes to your running stack using `rancher up
      --upgrade`.
    * **Verify** that the application is working as expected. These
      specifics steps here depend on your app and your workflow: You
      might need to visit the website, test the database, etc.
        * If the application is working well, you must **Remove** the
          old containers using `rancher up --upgrade
          --confirm-upgrade`
        * If the application is not working well, you can **rollback**
          to the old containers to restore the old working version.

!!! info
    For details on the Rancher upgrade commands, see the Rancher
    documentation at
    https://rancher.com/docs/rancher/v1.6/en/cattle/upgrading/ . Note
    that the Rancher documentation often refers to `rancher-compose`,
    which is an older command. Spin uses the `rancher` command
    instead, which is newer but has most of the same options. The
    Rancher documentation mentions *in-service upgrades* and *rolling
    upgrades*. Rolling upgrades are rarely used, and are not covered
    in these lessons.

## Example 2: Use "Build, Ship, Run" to use Secrets

Earlier we mentioned that it's a bad idea to store plain text
passwords in a Docker Compose file. We'll rectify that problem here by
converting our passwords into Rancher **Secrets**. Secrets should be
used to store any sensitive data, such as passwords or private keys,
and explained further in
the [Spin Best Practices Guide](../best_practices.md).

### Build

We've already built an application stack, but we'd like to make some
changes to it here to convert our plain text passwords to secrets, and
will modify the Docker Compose file to make use of that new secret.

#### (optional) Download (export) the configuration files for a stack

The configuration for an application stack is stored in the Docker
Compose files. If you don't have these files already, you would
normally obtain the files using `rancher export`.

Exporting a stack can be useful for collaboration and to store the
configuration files in a version control system. You can only export a
stack that you have permission to do so. For more info on Exporting a
stack, see the [Spin Tips](../tips_and_examples.md)
guide.

**We've already obtained our Docker Compose files from the Git repo**,
and don't actually need to obtain the files again using `rancher
export`. **To test this**, try out the command in a temporary
directory as follows. Note that `rancher export` will place the files
in a subdirectory which is named after the stack.

    elvis@nersc:elvis-flask-demo $ mkdir tmp
    elvis@nersc:elvis-flask-demo $ cd tmp/
    elvis@nersc:tmp $ rancher export elvis-flask-demo
    INFO[0001] Creating elvis-flask-demo/docker-compose.yml
    INFO[0001] Creating elvis-flask-demo/rancher-compose.yml
    elvis@nersc:tmp $ ls -l elvis-flask-demo/
    total 129
    -rw------- 1 elvis elvis 881 May 14 12:08 docker-compose.yml
    -rw------- 1 elvis elvis 171 May 14 12:08 rancher-compose.yml
    elvis@nersc:tmp $

#### Modify

In this example, we'll update our image to use Rancher Secrets. Take a
look at the file named `docker-compose.yml.example2` and compare it to
example1. Notice how the `web` container is the same, but the `app`
and `db` services now have changed. Both the App and DB containers use
Rancher **secrets** to store the password.

* Scroll to the bottom of the file and look at the section labeled
  `secrets`.  This section will read your secret from a local file off
  of the filesystem, and will store it as a secret named
  `db.YOUR_STACK_NAME.mongo-initdb-password`. We'll create this file
  in the next step.

        secrets:
        db.elvis-flask-demo.mongo-initdb-password:
        file: mongo-initdb-password

* The `app` and `db` containers each contain the following section,
  which mounts the secret into each container as a special volume with
  the filename of `/run/secrets/mongodb-initdb-password`, which strict
  ownership & permissions:

        secrets:
        - mode: '0444'
          uid: '0'
          gid: '0'
          source: db.elvis-flask-demo.mongo-initdb-password
          target: mongo-initdb-password

* The Flask and Mongo software will each read the environment variable
  `MONGO_INITDB_ROOT_PASSWORD_FILE`. The variable ends in `_FILE`,
  which informs the container to look for a file with that name, and
  then read it into the application. Without secrets, you would
  normally store the password in an environment variable named
  `MONGO_INITDB_ROOT_PASSWORD`.
* Secrets must be named in the in the format `[service name].[stack
  name].[filename]`, which would be
  `db.elvis-flask-demo.mongodb-initdb-password` here.

        #... (top of file removed for brevity)
        #...
        app:
          image: registry.spin.nersc.gov/USERNAME/spin-flask-demo-app:v1
          environment:
            MONGO_INITDB_ROOT_USERNAME: mongouser
            MONGO_INITDB_ROOT_PASSWORD_FILE: /run/secrets/mongo-initdb-password
          cap_drop:
          - ALL
          secrets:
          - mode: '0444'
            uid: '0'
            gid: '0'
            source: db.USERNAME-flask-demo.mongo-initdb-password
            target: mongo-initdb-password
        db:
          image: mongo:latest
          environment:
            MONGO_INITDB_ROOT_USERNAME: mongouser
            MONGO_INITDB_ROOT_PASSWORD_FILE: /run/secrets/mongo-initdb-password
          cap_drop:
          - ALL
          cap_add:
          - CHOWN
          - SETGID
          - SETUID
          secrets:
          - mode: '0444'
            uid: '0'
            gid: '0'
            source: db.USERNAME-flask-demo.mongo-initdb-password
            target: mongo-initdb-password

To use the secret, we first need to create a file which holds the
Secret. This file will be read when the Docker Compose file called.

    cat > mongo-initdb-password
    YourPasswordHere

Be sure that the permissions on the file are strict.

    chmod 600 mongo-initdb-password

### Ship

The second stage of *Build, Ship, Run* is often to ship the image to
the registry. However, We didn't change the Docker image here, and
we're skipping this step.

### Run

#### Apply the changes to your Stack

Once you are satisfied with the changes, you can apply them to the
running stack using `rancher up --upgrade`, like so:

    rancher up -d --upgrade --file docker-compose.yml.example2

You will see output like the following.

* The command will print lines like `Creating`, `Created`, `Starting`,
  `Started` even though the service was already started earlier. These
  messages are a little confusing, but are simply indicating that
  Rancher is inspecting the stack to see what needs to be upgraded.
* Notice the lines which say `Upgrading app` and `Upgrading db`.
* For each service in your stack, Spin will determine if an upgrade is
  necessary. If so, it will shut down the old containers in that
  service, and spawn new containers with the changes. The old
  containers will still reside on the hosts if you need to roll
  back. We'll cover that in the next step.

        elvis@nersc:elvis-flask-demo $ rancher up -d --upgrade --file docker-compose.yml.example2
        INFO[0000] Creating secret db.elvis-flask-demo.mongo-initdb-password with contents from file /project/projectdirs/superscience/spin/elvis-flask-demo/mongo-initdb-password
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
        1s3685
        INFO[0000] [web]: Started
        INFO[0000] Upgrading app
        INFO[0000] Upgrading db
        INFO[0010] [app]: Started
        1s3684
        INFO[0012] [db]: Started
        1s3683

If this upgrade didn't work, you can try forcing it with the `--force`
flag, like so. `--force-upgrade` will stop the old containers and spawn
new containers, regardless if there was a change or not.

    rancher up -d --upgrade --force-upgrade --file docker-compose.yml.example2

#### Verify that the stack is working

The application should be running now.

As mentioned in the previous step, upgrading a stack will shut down
the old set of containers and will spawn new ones in their place. The
old containers are preserved to give you a chance to verify that the
services are working correctly. The steps to verify the stack are up
to you and your team, but may involve tasks such as manually checking
that the web service works or automatically checking that the database
works.

#### Remove the old containers, or roll back to the previous version

If you are satisfied that the upgrade worked and that the stack is
working cleanly, you must remove the old containers. If the upgrade
had problems, you can roll back to the previous, working version of
the containers. Keep the following in mind:

* Rancher only stores two sets of containers: The current set, and the
  previous set.
* Once you have confirmed the upgrade, the old containers are removed
  and you cannot rollback to a previous version. This is not version
  control!
* We recommend that you store your configuration files in git or
  another version control system so that you can roll back to older
  versions of your software if necessary.

!!! Warning "Required Step: Confirming the upgrade"
    You must delete the old containers, or roll back to the previous
    version. The application stack is running, but you must follow
    this step. Many rancher commands, such as `rancher stop`,
    `rancher restart` and more will not work until you do this.

##### View the old and new containers

The old containers can be seen using the `rancher ps --all`
command. Note how the `app` and `db` service each are in the state of
`upgraded`, and that the scale says `2/1` which indicates that there
are two containers, but only the new one is running.

    nersc$ rancher ps --all
    ID      TYPE     NAME                  IMAGE                                                            STATE     SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3939  service  elvis-flask-demo/db   mongo:latest                                                     upgraded  2/1    false
    1s3940  service  elvis-flask-demo/app  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1             upgraded  2/1    false
    1s3941  service  elvis-flask-demo/web  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest    healthy   1/1    false
    nersc$

Running `rancher ps` with the `--containers` flag shows this a little
more clearly. The old containers are clearly visible here with the
state of **stopped**.

    nersc$ rancher ps --all --containers
    ID         NAME                    IMAGE                                                          STATE    HOST  IP             DOCKER        DETAIL
    1i2596137  elvis-flask-demo-app-1  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1           running  1h83  10.42.183.166  065asd9e0a
    1i2596138  elvis-flask-demo-db-1   mongo:latest                                                   stopped  1h83  10.42.87.90    1f6920d6a1e9
    1i2596146  elvis-flask-demo-web-1  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  running  1h82  10.42.44.155   66f48c9e36ee
    1i2596160  elvis-flask-demo-db-1   mongo:latest                                                   running  1h83  10.42.90.251   065fe407ae58
    1i2596161  elvis-flask-demo-app-1  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1           running  1h83  10.42.183.175  16faa310be0a
    nersc$

##### Remove the old containers

Assuming that that the upgrade went well and the application is
behaving as expected, it's time to confirm the upgrade and remove the
old containers by issuing the `--confirm-upgrade` flag. Rolling back
containers is discussed further below.

In the following example, I manually confirmed that my stack was
healthy after the upgrade. I will now issue the `--confirm-upgrade`
command to clear out the old containers.

    elvis@nersc:elvis-flask-demo $ rancher up -d --upgrade --confirm-upgrade --file docker-compose.yml.example2
    INFO[0001] Secret db.elvis-flask-demo.mongo-initdb-password already exists
    INFO[0001] [app]: Creating
    INFO[0001] [db]: Creating
    INFO[0001] [web]: Creating
    INFO[0001] [web]: Created
    INFO[0001] [app]: Created
    INFO[0001] [db]: Created
    INFO[0001] Secret db.elvis-flask-demo.mongo-initdb-password already exists
    INFO[0001] [web]: Starting
    INFO[0001] [app]: Starting
    INFO[0001] [db]: Starting
    1s3940
    INFO[0003] [web]: Started
    INFO[0003] [app]: Started
    INFO[0004] [db]: Started
    1s3939
    1s3941
    elvis@nersc:elvis-flask-demo $ rancher ps --all --containers
    ID         NAME                    IMAGE                                                          STATE    HOST  IP             DOCKER        DETAIL
    1i2596167  elvis-flask-demo-web-1  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  running  1h82  10.42.217.32   0b70b5c92b9f
    1i2596168  elvis-flask-demo-app-1  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1           running  1h83  10.42.217.189  a803b3f7d0a9
    1i2596169  elvis-flask-demo-db-1   mongo:latest                                                   running  1h82  10.42.86.62    3f9693ab5216
    elvis@nersc:elvis-flask-demo $

##### Rolling back to the old version (Optional exercise)

If something went wrong with the upgrade, you can **roll back** to the
previous version of your service.

To actually test the rollback feature, we will need to do some extra
work.  Remember that we removed the containers in the previous step,
and there are currently no containers to roll back too. Therefore, to
test the rollback feature, we are simulate an upgrade using the
`--force-upgrade` flag to force an upgrade regardless if there was a
change or not. This will create the second set of containers which
will be used in the exercise below.

    elvis@nersc:elvis-flask-demo $ rancher up -d --upgrade --force-upgrade --file docker-compose.yml.example2
    INFO[0000] [web]: Creating
    INFO[0000] [app]: Creating
    INFO[0000] [db]: Creating
    INFO[0000] [app]: Created
    INFO[0000] [db]: Created
    INFO[0000] [web]: Created
    INFO[0000] [web]: Starting
    INFO[0000] [app]: Starting
    INFO[0000] [db]: Starting
    1s3684
    INFO[0000] [web]: Started
    INFO[0000] Upgrading app
    INFO[0000] Upgrading db
    INFO[0007] [app]: Started
    INFO[0008] [db]: Started
    1s3683
    1s3685
    elvis@nersc:elvis-flask-demo $

The stack now has one set of running containers, and a second set of
stopped containers. Notice how the new containers have the same name,
but the `ID` is newer. For instance, the `elvis-flask-demo-db-1`
service has one stopped container with the ID of `1i2596321` and a new
running container with the ID of `1i2596330`.

    elvis@nersc:elvis-flask-demo $ rancher ps --all --containers
    ID NAME IMAGE STATE HOST IP DOCKER DETAIL
    1i2596321 elvis-flask-demo-db-1 mongo:latest stopped 1h85 10.42.33.150 874f960c26fc
    1i2596330 elvis-flask-demo-db-1 mongo:latest running 1h84 10.42.241.80 00fc04746cb6
    1i2596320 elvis-flask-demo-app-1 registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1 stopped 1h85 10.42.98.230 0ed5817cd109
    1i2596329 elvis-flask-demo-app-1 registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1 running 1h84 10.42.15.88 6e01f3fdfd96
    1i2596318 elvis-flask-demo-web-1 registry.spin.nersc.gov/elvis/my-first-container-nginx:latest stopped 1h86 10.42.123.139 8c1b3add29ff
    1i2596331 elvis-flask-demo-web-1 registry.spin.nersc.gov/elvis/my-first-container-nginx:latest running 1h86 10.42.52.63 1523342601c1
    elvis@nersc:elvis-flask-demo $

Issue the `--rollback` flag, and then view the containers
again. Notice how the new containers were removed, and the old
containers were restored. The `elvis-flask-demo-db-1` service only has
one container, with the ID of `1i2596321`. The other, newer instance
with ID `1i2596330` was removed.

    elvis@nersc:elvis-flask-demo $ rancher up --upgrade -d --rollback --file docker-compose.yml.example2
    INFO[0002] Secret db.elvis-flask-demo.mongo-initdb-password already exists
    INFO[0002] [db]: Creating
    INFO[0002] [web]: Creating
    INFO[0002] [app]: Creating
    INFO[0004] [db]: Created
    INFO[0004] [web]: Created
    INFO[0004] [app]: Created
    INFO[0005] Secret db.elvis-flask-demo.mongo-initdb-password already exists
    INFO[0005] [db]: Starting
    INFO[0005] [web]: Starting
    INFO[0005] [app]: Starting
    1s3956
    INFO[0017] [app]: Started
    INFO[0022] [db]: Started
    INFO[0022] [web]: Started
    1s3955
    1s3954
    elvis@nersc:elvis-flask-demo $ rancher ps --all --containers
    ID         NAME                    IMAGE                                                          STATE    HOST  IP             DOCKER        DETAIL
    1i2596321  elvis-flask-demo-db-1   mongo:latest                                                   stopped  1h85  10.42.33.150   874f960c26fc
    1i2596320  elvis-flask-demo-app-1  registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1           running  1h85  10.42.98.230   0ed5817cd109
    1i2596318  elvis-flask-demo-web-1  registry.spin.nersc.gov/elvis/my-first-container-nginx:latest  running  1h86  10.42.123.139  8c1b3add29ff
    elvis@nersc:elvis-flask-demo $

## Example 3: Upgrading to use Rancher NFS

NOTE: The following example is currently not working. We are working
on this.

Rancher NFS is a high-performance filesystem dedicated to Spin, and
provides a persistent storage alternative to the NERSC Project
Directories. Usage of Rancher NFS is fairly straightforward. In
docker-compose.yml.example3, we show you how to use Rancher NFS for
your MongoDB service. Open up the file and notice the `volume:`
section at the bottom:

    volumes:
      db.USERNAME-flask-demo:
        driver: rancher-nfs

This section defines a `volumes:` section in the YAML. Notice that
`volumes:`, `secrets:` and `services:` are all defined at the same
level (They are all indented). The whitespace here is important.

This section defines a volume named `USERNAME-flask-demo` which is
created using the Rancher NFS driver.

In order to use Rancher NFS volumes, you must map it to a path inside
the container where the service expects to find the
information. You'll generally find this information in the
documentation for the image. For example, according to *Where to Store
Data* at https://hub.docker.com/_/mongo/, MongoDB stores it's data at
/data/db. Therefore, in the Docker Compose file, we inform the service
to mount a volume named `USERNAME-flask-demo`, and mount it into the
container at /data/db:

    volumes:
    - db.USERNAME-flask-demo:/data/db

But wait! Before you use the new volume, you must actually create it.

Like Rancher Secrets, Rancher NFS volumes must be named in the format
`[service name].[stack name]`, which would be `db.USERNAME-flask-demo`
here. To create the volume, use the `rancher volume create` command,
like so:

    elvis@nersc:elvis-flask-demo $ rancher volume create --driver rancher-nfs db.elvis-flask-demo
    1v2795790
    elvis@nersc:elvis-flask-demo $

The volume is created, but will remain inactive until the a service
starts using it. To see the volume, call `docker volume ls` with the
`--all` flag. The `--all` flag lists all volumes owned by you, active
or not:

    elvis@nersc:elvis-flask-demo $ rancher volume ls --all
    ID         NAME                 STATE     DRIVER       DETAIL
    1v2795790  db.elvis-flask-demo  inactive  rancher-nfs
    elvis@nersc:elvis-flask-demo $

Now start the example stack, and watch the magic. Notice the line
which says `Creating volume template db.elvis-flask-demo`:

    elvis@nersc:elvis-flask-demo $ rancher up -d --upgrade --file docker-compose.yml.example3
    INFO[0001] Creating volume template db.elvis-flask-demo
    INFO[0001] Secret db.elvis-flask-demo.mongo-initdb-password already exists
    INFO[0001] [app]: Creating
    INFO[0001] [db]: Creating
    INFO[0001] [web]: Creating
    INFO[0001] [web]: Created
    INFO[0001] [app]: Created
    INFO[0001] [db]: Created
    INFO[0002] Existing volume template found for db.elvis-flask-demo
    INFO[0002] Secret db.elvis-flask-demo.mongo-initdb-password already exists
    INFO[0002] [db]: Starting
    INFO[0002] [web]: Starting
    INFO[0002] [app]: Starting
    1s3962
    INFO[0002] [web]: Started
    INFO[0002] [app]: Started
    1s3961
    INFO[0003] Upgrading db
    INFO[0017] [db]: Started
    1s3960
    elvis@nersc:elvis-flask-demo $

If everything worked, your stack should now be running with Rancher NFS.

## Example 4: Upgrade the image used in a stack

In Example 2 & 3, we upgraded the entire stack through Docker
Compose. Here, in example 3, we'll make a simple change to the image
used by the `app` service, and will push it out to our live stack.

Upgrading a container image used in your stack follows the same Build,
Ship, Run workflow shown above.

### Build on your laptop

Since the image exists on your laptop, we need to upgrade it
there. Switch back to your laptop into the directory where we built
the image in the first place

In the templates/page.html file, replace the `<body>` tag with `<body
style="background-color:#E6E6FA">`. This is a very simple change just
to demonstrate how to upgrade an image.

    elvis@elvis:~ $ cd ~/docker/elvis-flask-demo/app/
    elvis@elvis:app $ ls
    Dockerfile app.py docker-entrypoint.sh templates
    elvis@elvis:app $ vi templates/page.html

### Ship to the Registry

Rebuild the image, assign the new version tag of `:v2` to the image,
and push it to the Spin Registry. Remember to replace **YOURUSERNAME**
with your NERSC username.

    docker image build --tag registry.spin.nersc.gov/YOURUSERNAME/spin-flask-demo-app:v2 .

    docker image push registry.spin.nersc.gov/YOURUSERNAME/spin-flask-demo-app:v2

You should see output like this:

    elvis@elvis:app $ docker image build --tag registry.spin.nersc.gov/elvis/spin-flask-demo-app:v2 .
    Sending build context to Docker daemon 15.36kB
    Step 1/14 : FROM ubuntu:latest
    ---> f975c5035748
    ...
    Successfully built 7a56ed8f82d1
    Successfully tagged registry.spin.nersc.gov/elvis/spin-flask-demo-app:v2
    elvis@elvis:app $

    elvis@elvis:app $ docker image push registry.spin.nersc.gov/elvis/spin-flask-demo-app:v2
    The push refers to repository [registry.spin.nersc.gov/elvis/spin-flask-demo-app]
    8a89ef639332: Pushed
    6add53f81c0e: Pushed
    484dca8a893d: Layer already exists
    ab4028b3324e: Layer already exists
    cb4922a242f5: Layer already exists
    9a62f59ae797: Layer already exists
    db584c622b50: Layer already exists
    52a7ea2bb533: Layer already exists
    52f389ea437e: Layer already exists
    88888b9b1b5b: Layer already exists
    a94e0d5a7c40: Layer already exists
    v2: digest: sha256:af48d9c343d88f27f3fd69088b612bbb344aff5d847163ffffc0000268fb3bd9 size: 2616
    elvis@elvis:app $

If everything was successful, the new image with the `:v2` flag should
exist in the Spin repository.

### Run the new image on Spin

Switch back to Cori, update your Docker Compose file to use the new
image. We've done this for you with docker-compose.yml.example4. Go
ahead and compare docker-compose.yml.example4 with the the file
docker-compose.yml.example2 that we used in the previous example. The
only difference is this line which grabs the image:


    elvis@nersc:elvis-flask-demo $ diff docker-compose.yml.example4 docker-compose.yml.example3
    16c16
    < image: registry.spin.nersc.gov/elvis/spin-flask-demo-app:v2
    ---
    > image: registry.spin.nersc.gov/elvis/spin-flask-demo-app:v1
    elvis@nersc:elvis-flask-demo $

And then upgrade your stack to use the new image:

    elvis@nersc:elvis-flask-demo $ rancher up -d --upgrade --file docker-compose.yml.example4
    INFO[0000] Secret db.elvis-flask-demo.mongo-initdb-password already exists
    INFO[0000] [db]: Creating
    INFO[0000] [web]: Creating
    INFO[0000] [app]: Creating
    INFO[0000] [web]: Created
    INFO[0000] [app]: Created
    INFO[0000] [db]: Created
    INFO[0000] Secret db.elvis-flask-demo.mongo-initdb-password already exists
    INFO[0000] [web]: Starting
    INFO[0000] [app]: Starting
    INFO[0000] [db]: Starting
    1s3692
    INFO[0000] [web]: Started
    INFO[0000] [db]: Started
    1s3693
    INFO[0000] Upgrading app
    INFO[0008] [app]: Started
    1s3694
    elvis@nersc:elvis-flask-demo $

Revisit the site, and you'll see that the background color has
changed.

This concludes lesson 3. We hope you learned how to upgrade your
stacks using the Rancher upgrade command, and to use Rancher NFS &
Rancher Secrets.
