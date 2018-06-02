# Spin Getting Started Guide: Lesson 2: Run Your Application in Spin

## Lesson 2 Overview

In Lesson 1, you created an application stack comprised of a 'web' service created from a standard Nginx image from Docker Hub, and your own custom image named 'app' which is stored on your laptop. Lesson 2 will show you how to migrate this application from your laptop to Spin.

### The Rancher CLI

These lessons use the Rancher CLI to build and manage services and stacks in Spin. The Rancher CLI is loaded from the Spin module on all major NERSC systems, and activated with the command 'rancher'. The Rancher CLI must be used from a NERSC system and cannot be used from your laptop, as we maintain a modified version of the Rancher CLI which is optimized to work with the NERSC environment. While Rancher also provides a Web Interface, it is currently only available to NERSC Administrators.

All Rancher commands communicate with the Spin infrastructure using an API key. You will generate an API key below.

### Spin Environments

Spin has two main environments for NERSC users:

* 'dev-cattle' is for use with applications which are under development
* 'prod-cattle' is used for production services.

A third environment named 'sandbox' will be used exclusively if you are taking the SpinUp sessions.

During normal development, you will first deploy your application to the development environments, and will copy it to production when ready. Currently, ISG must approve all applications before they run in the production environment.

The name 'cattle' refers to the container 'Orchestrator' which we use to manage containers and is part of Rancher. Rancher names many of their components with 'Ranch'-themed names, such as 'Longhorn' or 'Wagyu'. To read more information on Rancher, please read the [Spin Getting Started Guide overview](index.md).

### Security Audit

All applications sent to Spin are automatically audited at the API to ensure that they follow our security requirements, which are outlined in the [Spin Best Practices Guide](https://docs.nersc.gov/services/spin/best_practices/). The Rancher CLI will print an error if an application breaks one of the security requirements.

## Prerequisites for Lesson 2

Before you review lesson 2, be sure to understand the concepts of [Spin Getting Started Guide: Lesson 1: Building your first application on your laptop](lesson-1.md).

You need also need:

* SSH access on a NERSC Login node, such as cori.nersc.gov or edison.nersc.gov.
* Access to a Project directory on the NERSC Global Filesystem, and the ability to run 'chmod o+x' on that directory to allow the user 'nobody' to read files in that directory. This requirement is explained in detail below in 'Part 2: Prepare your application to run in Spin'.
* An account on Spin. To do that, please see the [Spin Getting Started Guide: How do I get started?](index.md) Test your Spin account by quickly logging into the Spin Registry from your laptop. You should see the message 'Login Succeeded':

        stefanl@stefanl:~ $ docker login https://registry.spin.nersc.gov/
        Username: stefanl
        Password:
        Login Succeeded
        stefanl@stefanl:~

### Generate API keys on a system such as Cori or Edison

Communication with the Rancher server requires an NERSC account and an API key tied to that account. Follow the steps below to generate an API key.

1. Log in to a NERSC login node and load the Spin module. The Spin module loads the Rancher CLI and related utilities.
    * The Rancher module does not currently work on Denovo. We will get it working soon.

    ```
    stefanl@stefanl:~ $ ssh cori.nersc.gov
    stefanl@cori07:~ $ module load spin
    stefanl@cori07:~ $ which rancher
    /global/common/shared/isg/spin/0.1/bin/rancher
    stefanl@cori07:~ $
    ```

2. Generate the API Key if you have not already done so. These keys will be used to access the Rancher infrastructure.

        stefanl@cori07:~ $ spin-keygen.sh
        Password for user stefanl?
        Success: Spin API Key generated for stefanl.
        stefanl@cori07:~ $

3. (Optional) View. The Rancher CLI stores its configuration file under your home directory, at ~/.rancher/cli.json, and can be verified using the command 'rancher config --print'. The command will print out your access key, the secret key, URL for the Rancher Server API, the environment used for this key (normally blank) and the path to the key file. The private key is used to authenticate as you to the Rancher API. Never share your private key.

        stefanl@cori07:~ $ rancher config --print
        {"accessKey":"ABCDEFG123456","secretKey":"ZYXWVUT098765","url":"https://rancher.spin.nersc.gov/v2-beta/schemas", "environment":"", "path":"/global/homes/s/stefanl/.rancher/cli.json"}
        stefanl@cori07:~ $

4. Validate that you account is working correctly by running the command 'rancher environment'. This command will communicate to the Rancher Server API using your API key. If 'rancher environment' returns a list of environments, your account is working correctly. If the command prints an error such as '401 Unauthorized', your account is not functioning. Please contact us for help.

In the following example I ran the 'rancher environment' command, which printed out all three environments that I have access too. Environments are explained in greater detail in the 'Overview' section, above.

    stefanl@cori08:~ $ rancher environment
    ID         NAME         ORCHESTRATION  STATE   CREATED
    1a736936   prod-cattle  cattle         active  2017-02-27T23:59:40Z
    1a5        dev-cattle   cattle         active  2016-10-11T01:02:27Z
    1a1221788  sandbox      cattle         active  2018-03-12T21:25:22Z
    stefanl@cori08:~ $

Most Rancher commands only operate on stacks & services within one environment, and will need to know which environment to use. If you simply run a command now, Rancher will ask you to select the environment for most commands. This can be a hassle:

If 'RANCHER_ENVIRONMENT' is not set, Rancher will ask you to select the environment for most commands, which can be a hassle. (Note that I have two services running in the sandbox environment, just to illustrate how this command would work. You have not spun up any services yet, and will see no processes here.)

    stefanl@cori08:~ $ rancher ps
    Environments:
    [1] prod-cattle(1a736936)
    [2] dev-cattle(1a5)
    [3] sandbox(1a1221788)
    Select: 3
    ID      TYPE     NAME                IMAGE  STATE    SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3712  service  stefanl-webapp/web  httpd  healthy  1/1    false
    1s3713  service  stefanl-webapp/db   mysql  healthy  1/1    false
    stefanl@cori08:~ $

To simplify your workflow, use the 'RANCHER_ENVIRONMENT' variable to specify the environment to be used.

As mentioned earlier, SpinUp students will be using the 'sandbox' environment. Outside of the SpinUp lessons, you will be using the 'dev-cattle' environment for these lessons. In the following example, I'm using the 'RANCHER_ENVIRONMENT' variable to specify the 'sandbox' environment.

    stefanl@cori08:~ $ export RANCHER_ENVIRONMENT=sandbox
    stefanl@cori08:~ $ rancher ps
    ID      TYPE     NAME                IMAGE  STATE    SCALE  SYSTEM  ENDPOINTS  DETAIL
    1s3712  service  stefanl-webapp/web  httpd  healthy  1/1    false
    1s3713  service  stefanl-webapp/db   mysql  healthy  1/1    false
    stefanl@cori08:~ $

If everything above worked well, you should be ready to go.

## Part 1: Ship your image from your laptop to the Spin Registry

The Application Stack in Lesson 1 used a custom image named 'my-first-container-app' which is currently stored on your laptop. Before you can use the container in Spin, this custom image must be shipped ("pushed") to the Spin Registry.

A 'Registry' holds Docker images. The Spin Registry is a private registry for NERSC users to hold Docker images which should not be available publicly, and is available at https://registry.spin.nersc.gov/. The [Docker Hub](https://hub.docker.com/) registry is a place to hold Docker images for the general public, and is the default for most public images. If you have a paid account on Docker Hub, you can also store your own private images. For more information on registries in general, see the Docker documentation at https://docs.docker.com/registry/introduction/. 

### Step 1: Tag your 'my-first-container-app' image

Here, we will use a Docker image 'tag' to create a new name for your image to be used in Spin. Docker image tags are used for a few purposes, such as:

* Create a friendly name for your image. If there was no name for an image, we would have to refer to it by an image ID like 'ecc5b5995f8b' which isn't very friendly.
* Provide versioning information to an image, such as a version number help track development versions.
* Help to locating your images on a shared, multi-user environment such as Spin. Remember that Spin is a shared resource. If everyone named their containers "my-first-container", it will be difficult for administrators to locate your specific container in Spin.
* Set a URL to a private image registry, such as the Spin Registry.

These images will be named following a format of 'registry.spin.nersc.gov/USERNAME/ContainerName', where 'USERNAME' is your NERSC username. When collaborating with a team or a group, these images may instead use the format of 'registry.spin.nersc.gov/TEAMNAME/ContainerName'.

On your laptop, list your custom image which are stored locally:

    stefanl@stefanl:~ $ docker image list my-first-container-app
    REPOSITORY TAG IMAGE ID CREATED SIZE
    my-first-container-app latest c4f1cd0eb01c About an hour ago 165MB
    stefanl@stefanl:~ $

Now tag your image with the following command, replacing 'YOURUSERNAME' with your NERSC username.

    docker image tag my-first-container-app registry.spin.nersc.gov/YOURUSERNAME/my-first-container-app

List the images again. You should see output like the following:

    stefanl@stefanl:~ $ docker image tag my-first-container-app registry.spin.nersc.gov/stefanl/my-first-container-app
    stefanl@stefanl:~ $ docker image list | grep my-first-container-app
    my-first-container-app latest ecc5b5995f8b 2 weeks ago 165MB
    registry.spin.nersc.gov/stefanl/my-first-container-app latest ecc5b5995f8b About an hour ago 165MB
    stefanl@stefanl:~ $

Notice a few things:

* Your image has an "Image ID" like 'ecc5b5995f8b', but now has two names: One local, and one ready for the remote registry. The Image ID is the same for both, which means that both tags refer to the exact same image. An image ID is unique.
* The name for an image is technically called a 'Repository' in the Docker CLI.
* We didn't explicitly give this image a version tag, so Docker defaulted to the version of 'latest'. For more advanced projects, we recommend that you avoid 'latest' and instead set a version number as part of the tag.
* The URL for the Spin Registry, registry.spin.nersc.gov, is now part of the image name. For a public repository like the Docker Hub, the URL is not printed, but is set to 'docker.io' (e.g. Docker Hub).

### Step 2: Tag your 'my-first-container-nginx' image

Now, do the same thing for your 'web' image, replacing 'YOURUSERNAME' with your NERSC username:

    docker image tag my-first-container-nginx registry.spin.nersc.gov/YOURUSERNAME/my-first-container-nginx


### Step 3: Log into the Spin Registry

You need to authenticate to the private registry before using it. Usually, it's simplest to log in to the Spin Registry before working with it, especially if you need to troubleshoot any connection issues. You can skip this step, but you will asked to authenticate later while pushing the image.

From your laptop, log in now using your NERSC username & password if you haven't already done so.

    stefanl@stefanl:~ $ docker login https://registry.spin.nersc.gov/
    Username (stefanl):
    Password:
    Login Succeeded
    stefanl@stefanl:~ $

While some portions of Spin are restricted to NERSC & LBL networks, the Spin Registry is available from the public Internet. This allows all NERSC users to build and collaborate with other NERSC users.

If you cannot log into the Spin Registry, it may mean that you do not have an account in Spin yet. If this is the case, please see the [Spin Getting Started Guide: How do I get started?](index.md)

### Step 4: Push your images to the Spin Registry from your laptop

Once you have logged into the Spin Registry, push your image to the registry with the following commands, replacing 'YOURUSERNAME' with your NERSC username:

    docker image push registry.spin.nersc.gov/YOURUSERNAME/my-first-container-app

    docker image push registry.spin.nersc.gov/YOURUSERNAME/my-first-container-nginx

Each push command should print output like the following:

    stefanl@stefanl:~ $ docker image push registry.spin.nersc.gov/stefanl/my-first-container-app
    The push refers to repository [registry.spin.nersc.gov/stefanl/my-first-container-app]
    f596782719e9: Pushed
    0980b60a97f3: Pushed
    c9c93665072f: Pushed
    45f0f161f074: Layer already exists
    latest: digest: sha256:2daa3fa3c0bf26b25d4a9f393f2cdcff8065248fd279f6c4e23e7593fc295246 size: 1154
    stefanl@stefanl:~ $

Now the images are available in the Spin Registry and can be pulled into the Rancher environment.

## Part 2: Prepare your application to run in Spin

In this part, we'll prepare your application to run in Spin by copying required assets to your Project directory on the NERSC Global Filesystem, and modifying your Docker Compose file to work with Spin.

When using the NERSC Global Filesystem, there are some important considerations about users & groups to keep in mind:

* Many containers in the Docker community run as 'root' by default, even though this practice goes against security recommendations and [Docker's recommendations](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user).
* Running as root is not allowed for any container which uses the NERSC Global Filesystem. Containers which mount directories from the NERSC Global Filesystem must run under the Linux UID or UID/GID of a NERSC User Account or a Collaboration Account, and will need to have other changes made for security. This policy will be enforced by the Spin software.
* It is also possible to run the Spin containers under your own Unix username and group. The containers can also run as a Collaboration Account if you prefer. We'll cover how to do this in the [Spin Best Practices Guide](https://docs.nersc.gov/services/spin/best_practices/).
* The permissions for all files used by your Spin application must be readable by this account.
* We are looking into ways to improve this experience, and are waiting for improvements in the Linux Kernel and in Docker itself.

### Step 1: Make your Project directory available to Spin

We will be creating some subdirectories within your Project directory which will store your Docker Compose file, as well as files used by your application.

The application subdirectories will be mounted inside the container by way of a Docker 'volume mount'. This means that you must set the permissions of your Project directory to be available to Spin as follows:

* All directories and subdirectories used by your Spin application must have the 'executable' bit set to allow 'other' (o+x) so that the Docker daemons can successfully mount your directory. For example permission mode 0741 would work on a parent directory, but 0740 would not.
* The directories do not need to be world readable. The Docker daemon has no need for the 'read' or 'write' bit to be set on any directories. It only need the 'executable' bit set.
* Spin will be accessing these directories as the special system user 'nobody'.
* Any files in these directories need to be readable as the user & group which is used within the container. Normally, we encourage you to run the application as your NERSC Collaboration User. For these lessons, you can instead run as your own account, using your own UID & GID.
* Since this application will be running as your own UID & GID, and just make sure that your account can read the files.
* The Docker Compose file simply needs to be readable by your account, or whomever is running the 'rancher' command.

* All other permissions on these directories are up to you and your team.

Head back your session on Cori, and create the Spin Project directory with a command similar to this:

    PROJECT_DIRECTORY=/global/project/projectdirs/YOUR_PROJECT_DIRECTORY/YOURUSERNAME-first-stack
    mkdir $PROJECT_DIRECTORY
    mkdir $PROJECT_DIRECTORY/web
    chmod o+x $PROJECT_DIRECTORY $PROJECT_DIRECTORY/web

And now, we copy the Nginx configuration file from your laptop to this directory. For your own future projects, you can do this step by hand, using Git, or whatever method your team chooses.

    cd $PROJECT_DIRECTORY/web
    vim nginx-proxy.conf
    ...

In the following example, I created these files as 'stefanl'. My group is 'isg'. Notice how the directories 'stefanl-first-stack' and 'stefanl-first-stack/web' have the executable bit set for 'other', while 'nginx-proxy.conf' is only readable by me and my group.

    stefanl@cori07:~ $ cd /global/project/projectdirs/isguser/spin/stefanl-first-stack
    stefanl@cori07:stefanl-first-stack $ find . -ls
    507571599 1 drwx-----x 3 stefanl isg 512 Mar 14 17:49 .
    507571617 1 drwx-----x 2 stefanl isg 512 Mar 14 17:52 ./web
    507571673 1 -rw-r----- 1 stefanl isg 299 Mar 14 17:52 ./web/nginx-proxy.conf
    stefanl@cori07:stefanl-first-stack $

!!! info "A quick security reminder about public & private web services and the NERSC Global Filesystems"
    As a reminder about NERSC's security policy for services which read and write to the NERSC Global Filesystems. Public services, such as a Public webserver, are permitted read from the Global Filesystems, but be sure that the access is *read only*. Public services which *write* to the Global Filesystems must authenticate all users so that you team and the NERSC staff can trace any write activity to an account if necessary. Private services, available only within NERSC can also write to the filesystem.

### Step 2: Modify your Docker Compose file

We will need to modify the version of the docker-compose.yml file on your laptop to use our modified Nginx container, and will implement a few more security settings.

In your project directory on Cori (e.g. '/global/project/projectdirs/YOUR_PROJECT_DIRECTORY/YOURUSERNAME-first-stack', create a new Docker Compose file. Technically, the Docker Compose file can live elsewhere on the filesystem, such as in your home directory under ~/docker/ as long as you have access to it. The Project directory is just a convenient place to store this file.

The name of the directory matters. By default, Rancher will name the stack after your directory name. In these exercises we make sure that the directory name and the stack name match just to keep things simple. You can also specify the stack name using the '--stack' flag.

Add the following text to your docker-compose.yml file, but replace the values such as 'YOURUSERNAME', 'YOUR_PROJECT_DIRECTORY, and 'YOUR_NERSC_UID:YOUR_NERSC_GID' with your specific information.

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
    - /global/project/projectdirs/YOUR_PROJECT_DIRECTORY/YOURUSERNAME-first-stack/web/nginx-proxy.conf:/etc/nginx/conf.d/default.conf:ro
    user: YOUR_NERSC_UID:YOUR_NERSC_GID
    group_add:
    - nginx
    cap_drop:
    - ALL

Compare this Docker Compose file to the version on your laptop, and note note that the major differences:

* These services use your custom images which you built and pushed to the Registry.
* The Nginx configuration file lives in your Project directory. The ':ro' parameter, specified at the very end of the line, will make the file Read Only. All files & directories in Spin should be Read Only whenever possible.
* Ports
    * The application will expose the external port 60000 and will map that to the private port of 8080, which is used by Nginx as mentioned in the Dockerfile from Lesson 1.
    * The port range 60000-60050 are made available for Spin development, and you may use any port in that range for the exercise here. These ports are open to the world, and bots may start scanning your app.
    * The port range 50000-50050 is also available during development, but is restricted to NERSC & LBL networks only (Including the JGI & LBL networks).
    * Sensitive ports, such as MySQL (port 3306), Postgres (5432) are restricted to NERSC & LBL networks only.
    * A detailed list of Ports and their accessibility can be found in the [Spin Best Practices Guide](https://docs.nersc.gov/services/spin/best_practices/), under "Networking".
* Users and groups
    * The application runs as your UID & GID account, so it can access files outside the container. In Linux terms, this is your primary UID and the GID of a group which you are a member of.
    * The secondary group 'nginx' is added to the account, so the account can access files inside the container.
* Security:
    * All Linux kernel capabilities are dropped with the 'cap_add: ALL' parameter to improve the security of the application.
    * Linux Kernel 'capabilities' are fine-grained controls over superuser capabilities. Docker ships with a [small, restricted set of Kernel capabilities by default,](https://docs.docker.com/engine/security/security/#linux-kernel-capabilities) and is fairly secure out of the box. By dropping all remaining capabilities, we're taking it a step further to make our container even more secure. If you needed to add specific capabilities back to the container, you can add them with the 'cap_add:' parameter, which is discussed more in the [Spin Best Practices Guide](https://docs.nersc.gov/services/spin/best_practices/).

### Step 3: Sanity checks

Let's verify our config file before proceeding using 'rancher up --render'. If the Docker Compose file is free from validation errors, Docker Compose will print out the contents of the file, like so:

    stefanl@cori07:stefanl-first-stack $ ls -ld docker-compose.yml
    -rw-rw---- 1 stefanl stefanl 455 May 15 11:58 docker-compose.yml
    stefanl@cori07:stefanl-first-stack $ rancher up --render
    version: '2'
    services:
    app:
    image: registry.spin.nersc.gov/stefanl/my-first-container-app:latest
    cap_drop:
    - ALL
    web:
    image: registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest
    ports:
    - "60050:8080"
    volumes:
    - /global/project/projectdirs/isguser/spin/stefanl-first-stack/web/nginx-proxy.conf:/etc/nginx/conf.d/default.conf:ro
    user: 46311:71216
    group_add:
    - nginx
    cap_drop:
    - ALL

    stefanl@cori07:stefanl-first-stack $

If the Docker Compose file contained any validation errors, Rancher will complain with an error similar to the following:

    stefanl@cori07:stefanl-first-stack $ rancher up --render
    ERRO[0001] Could not parse config for project stefanl-first-stack : yaml: line 4: mapping values are not allowed in this context
    FATA[0001] Failed to read project: yaml: line 4: mapping values are not allowed in this context
    stefanl@cori07:stefanl-first-stack $

While you're at it, do a quick verification step and make sure that path to the Nginx file matches the path specified in the Compose file. List the file, and grep the path from the docker-compose.yml file to make sure they match:

    ls /global/project/projectdirs/YOUR_PROJECT_DIRECTORY/stefanl-first-stack/web/nginx-proxy.conf/web/nginx-proxy.conf

    grep /global/project/projectdirs/YOUR_PROJECT_DIRECTORY/stefanl-first-stack/web/nginx-proxy.conf docker-compose.yml

Note that Rancher also supports a second configuration file named rancher-compose.yml, but that is for advanced use cases such as scaling. We may cover it in a future lesson.

## Part 3: Start the stack

Make sure you use the 'sandbox' environment for this first application. We mentioned this above, but let's set it to be sure.

    export RANCHER_ENVIRONMENT=sandbox

Now that your Docker compose files are available, and all required files are available on the NERSC Global Filesystem, it's time to start your stack, with this command. By default, Rancher will create a stack named after your current working directory, which should be named like 'USERNAME-first-stack'. If you want to name the stack something different, use the '--stack' flag to specify the name.

Start your stack with the 'rancher up' command, using the '-d' flag to send all logs to the background.

    rancher up -d

You should see a log output like the following:

    stefanl@cori07:stefanl-first-stack $ rancher up -d
    INFO[0002] [app]: Creating
    INFO[0002] [web]: Creating
    INFO[0002] Creating service app
    INFO[0002] Creating service web
    INFO[0004] [app]: Created
    INFO[0005] [web]: Created
    INFO[0005] [app]: Starting
    INFO[0005] [web]: Starting
    1s3971
    INFO[0014] [app]: Started
    1s3972
    INFO[0015] [web]: Started
    stefanl@cori07:stefanl-first-stack $

We used the '-d' flag to send our logs to the background. To send the application logs to the foreground, omit the '-d' flag, and when you're done, use 'Ctrl-C' to stop viewing the logs, which will send the logs to the background.

Stacks can also be created using the 'rancher stack create StackName' command. Unlike 'rancher up', 'rancher stack create' requires you to specify the name of the stack. We use 'rancher up' in our lessons.

View your stacks using 'rancher stack ls'. If you have more than one stack, this will show all of them. Right now, we only have one stack.

    stefanl@cori07:stefanl-first-stack $ rancher stack ls
    ID NAME STATE CATALOG SERVICES SYSTEM DETAIL AVAILABLE UPGRADES
    1st1671 stefanl-first-stack healthy 2 false
    stefanl@cori07:stefanl-first-stack $

View your services using 'rancher ps'. This command will show all services in all of your stacks.

    stefanl@cori07:stefanl-first-stack $ rancher ps
    ID TYPE NAME IMAGE STATE SCALE SYSTEM ENDPOINTS DETAIL
    1s3971 service stefanl-first-stack/app registry.spin.nersc.gov/stefanl/my-first-container-app:latest healthy 1/1 false
    1s3972 service stefanl-first-stack/web registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest healthy 1/1 false
    stefanl@cori07:stefanl-first-stack $

#### Find the URL for your service

You may now open your the application with a browser, but we'll need the URL first. In Production, web services will normally be served from behind the Spin loadbalancer, and you would normally create a proper friendly hostname such as 'mystack.nersc.gov', 'mystack.lbl.gov', 'mystack.jgi.doe.gov' or 'mystack.org', and use a CNAME to point it to the loadbalancer. We'll cover that more later.

For now, you'll need to find the FQDN and Port number on your own. Use 'rancher inspect' along with the ['jq' tool](https://stedolan.github.io/jq/). 'rancher inspect' will print information about your stack in JSON, and 'jq' tool makes the output friendlier.

In the example below, I'm obtaining the FQDN, and the port number of the public endpoint for my web service, 'stefanl-first-stack/web'. Note the following:

* The FQDN is a stable endpoint which will always point to your stack, and we recommend that you use it instead of the IP address. The IP address will change over time as the service moves from one node to another, while the FQDN will be kept up to date dynamically.
* Note that the hostname below ends with **nersc.org** instead of **nersc.gov**. Nersc.org is used by Spin. to allow dynamic DNS updates.
* The information below shows that the web service can be reached at http://web.YOUR-STACK-NAME.dev-cattle.stable.spin.nersc.org:60000/. Go ahead and plug that URL into your browser to test the stack. If the URL does not work, wait 5 minutes and try again, or try the IP address instead.

        stefanl@cori07:stefanl-first-stack $ rancher inspect stefanl-first-stack/web | jq '.fqdn'
        "web.stefanl-first-stack.sandbox.stable.spin.nersc.org"
        stefanl@cori07:stefanl-first-stack $ rancher inspect stefanl-first-stack/web | jq '.publicEndpoints'
        [
        {
        "hostId": "1h85",
        "instanceId": "1i2597508",
        "ipAddress": "128.55.206.21",
        "port": 60000,
        "serviceId": "1s3972",
        "type": "publicEndpoint"
        }
        ]
        stefanl@cori07:stefanl-first-stack $

## Part 4: A simple upgrade to your stack

In the previous example, we created a web service which listens on port 60000. In this example, we will modify our application to listen on 60040. This is a very simple upgrade just to show how it's done. We'll cover more advanced upgrades in the next lesson.

Edit the Docker Compose file, and change the ports from 60000:

    ports:
    - "60000:8080"

To 60040:

    ports:
    - "60040:8080"

And now, push the upgrade to your stack using the 'rancher up --upgrade' command. Be sure that the directory name matches the name of your stack, or specify your stack here using '--stack YOURUSERNAME-first-stack':

    rancher up --upgrade -d

We're using the '-d' flag here to send the logs to the background, like we did in ‘Part 3: Start the stack’ above. To send logs to your screen, remove the '-d' flag, and hit 'Ctrl-C' to exit the logs command (The application will continue running in the background).

And finally, browse to your service on the new port, at http://web.YOUR-STACK-NAME.dev-cattle.stable.spin.nersc.org:60040/ (Or the port that you used above) . You should see logs like the following. Note the line which says 'Upgrading web'.

    stefanl@cori03:stefanl-first-stack $ rancher up --upgrade -d
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

The web service was upgraded. However, the old version of the container is still there in case you want to roll back to that version. You must remove the old containers as soon as you are happy with the new version. Certain commands, such as 'rancher stop', will not function until you remove the old containers. We will cover this more in Lesson 3.

'rancher ps --containers' will show the new version of the containers, along with the old version. Run this command now. Note how there are two containers named 'something-web-1'. One is running, and one is stopped:

    stefanl@cori06:stefanl-first-stack $ rancher ps --containers
    ID NAME IMAGE STATE HOST IP DOCKER DETAIL
    1i2599342 stefanl-first-stack-app-1 registry.spin.nersc.gov/stefanl/my-first-container-app:latest running 1h84 10.42.165.210 c099b99d60d2
    1i2599346 stefanl-first-stack-web-1 registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest stopped 1h87 10.42.14.161 843972812e3b
    1i2599347 stefanl-first-stack-web-1 registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest running 1h85 10.42.66.5 bf128471d412
    stefanl@cori06:stefanl-first-stack $

'rancher ps' without the '--containers' flag will also show your running and stopped containers, but it's subtle. Look under the 'SCALE' column, and notice that your 'web' service is says '2/1' which means 'Two containers exist. One is in use.'

    stefanl@cori06:stefanl-first-stack $ rancher ps
    ID TYPE NAME IMAGE STATE SCALE SYSTEM ENDPOINTS DETAIL
    1s4146 service stefanl-first-stack/app registry.spin.nersc.gov/stefanl/my-first-container-app:latest healthy 1/1 false
    1s4147 service stefanl-first-stack/web registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest upgraded 2/1 false
    stefanl@cori06:stefanl-first-stack $

To remove the old containers, run the command 'rancher up -d --upgrade --confirm-upgrade' which will confirm that the upgrade was successful and delete the old containers. After running this command, note that the 'stopped' container was removed.

    stefanl@cori06:stefanl-first-stack $ rancher up -d --upgrade --confirm-upgrade
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
    stefanl@cori06:stefanl-first-stack $ rancher ps --containers
    ID NAME IMAGE STATE HOST IP DOCKER DETAIL
    1i2599342 stefanl-first-stack-app-1 registry.spin.nersc.gov/stefanl/my-first-container-app:latest running 1h84 10.42.165.210 c099b99d60d2
    1i2599347 stefanl-first-stack-web-1 registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest running 1h85 10.42.66.5 bf128471d412
    stefanl@cori06:stefanl-first-stack $

## Other ways to work with your stack

The Rancher CLI can do many things. Here are some common tasks you can do with the Rancher CLI. A more comprehensive list can be found at [Spin: Tips & Examples with the Rancher CLI & Docker CLI](http://docs.nersc.gov/services/spin/tips_and_examples/).

### View all of your stacks

    stefanl@cori07:stefanl-first-stack $ rancher stacks ls
    ID NAME STATE CATALOG SERVICES SYSTEM DETAIL AVAILABLE UPGRADES
    1st1663 stefanl-mongodb healthy 2 false
    1st1670 stefanl-flask-demo-live healthy 3 false
    1st1671 stefanl-first-stack healthy 2 false
    stefanl@cori07:stefanl-first-stack $

### View the services in your stack

Here, we're specifying the --all flag to show all running and non-running containers, if there are any:

    stefanl@cori07:~ $ rancher ps --all
    1s3039 service stefanl-first-stack/web nginx healthy 1/1 false
    1s3040 service stefanl-first-stack/app registry.spin.nersc.gov/stefanl/my-first-container-app healthy 1/1 false
    stefanl@cori07:~ $

### Scale a service in your stack

Here, we'll scale the web service from one container to two:

    stefanl@cori10:~ $ rancher scale stefanl-first-stack/web=2
    stefanl-first-stack/web
    stefanl@cori10:~ $ rancher ps --containers
    1i2569728 stefanl-first-stack-web-1 nginx running 1h42 10.42.79.198 1c442b1d19a3
    1i2569732 stefanl-first-stack-web-2 nginx running 1h2 10.42.33.67 6684fffd5270
    stefanl@cori10:~ $

### View the logs for your stack

We scaled the Web service to 2 containers above. Note how this command will show the logs for both containers in the service, for web-1 and web-2:

    stefanl@cori10:~ $ rancher logs stefanl-first-stack/web
    stefanl-first-stack-web-1 | 131.243.223.227 - - [18/Jan/2018:18:12:05 +0000] "GET / HTTP/1.1" 200 12 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36" "-"
    stefanl-first-stack-web-2 | 131.243.223.227 - - [18/Jan/2018:18:16:50 +0000] "GET / HTTP/1.1" 200 12 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36" "-"
    ...
    ...

### Stop your stack

Stop your stack using 'rancher stop'. Notice that I'm specifying the 'rancher ps' command with the '--all' flag, which shows all containers, running or not.

    stefanl@cori10:stefanl-first-stack $ rancher stop stefanl-first-stack
    1st1443
    stefanl@cori10:stefanl-first-stack $ rancher ps --all
    ID TYPE NAME IMAGE STATE SCALE SYSTEM ENDPOINTS DETAIL
    1s3748 service stefanl-first-stack/app registry.spin.nersc.gov/stefanl/my-first-container-app:latest inactive 1/1 false
    1s3749 service stefanl-first-stack/web registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest inactive 1/1 false
    stefanl@cori10:stefanl-first-stack $

### Start your Stack

    stefanl@cori10:stefanl-first-stack $ rancher start stefanl-first-stack
    1st1443
    stefanl@cori10:stefanl-first-stack $ rancher ps
    ID TYPE NAME IMAGE STATE SCALE SYSTEM ENDPOINTS DETAIL
    1s3748 service stefanl-first-stack/app registry.spin.nersc.gov/stefanl/my-first-container-app:latest healthy 1/1 false
    1s3749 service stefanl-first-stack/web registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest healthy 1/1 false
    stefanl@cori10:stefanl-first-stack $

### Obtain a shell into a container

**Remember: Any changes that you make to a running container will be removed** next time the container is restarted. **Containers are ephemeral by design, not persistent.** This is one big difference between Virtual Machines and Containers. To make a permanent change, you must change image or the Docker Compose file. For persistent data, we use Rancher NFS volumes, mounts on the Global Filesystem (shown above) and will provide other options in the future.

    stefanl@cori10:stefanl-first-stack $ rancher exec -it stefanl-first-stack-web-1 /bin/bash
    root@1c442b1d19a3:/# ls
    bin dev home lib64 mnt proc run srv tmp var
    boot etc lib media opt root sbin sys usr
    root@1c442b1d19a3:/# exit
    exit
    stefanl@cori10:stefanl-first-stack $

### Removing your stack

Use 'rancher rm --type stack' to remove your stack.

    stefanl@cori07:stefanl-flask-demo-live $ rancher ps
    ID TYPE NAME IMAGE STATE SCALE SYSTEM ENDPOINTS DETAIL
    1s3971 service stefanl-first-stack/app registry.spin.nersc.gov/stefanl/my-first-container-app:latest healthy 1/1 false
    1s3972 service stefanl-first-stack/web registry.spin.nersc.gov/stefanl/my-first-container-nginx:latest healthy 1/1 false
    stefanl@cori07:stefanl-flask-demo-live $ rancher rm stefanl-first-stack --type stack
    1st1668
    stefanl@cori07:stefanl-flask-demo-live $ rancher ps
    ID TYPE NAME IMAGE STATE SCALE SYSTEM ENDPOINTS DETAIL
    stefanl@cori07:stefanl-flask-demo-live $

If you forget the '--type stack' flag, Rancher may complain with an error like 'You don't own volume your-stack-here'. This is an ordering problem within Rancher which only occurs sometimes. Using the '--type stack forces Rancher to do the correct thing.

## Other useful command line tasks

For more examples of how to use the Rancher CLI, please see [Spin: Tips & Examples with the Rancher CLI & Docker CLI](http://docs.nersc.gov/services/spin/tips_and_examples/).

## Next Steps: Lesson 3

The next lesson, [Spin Getting Started Guide: Lesson 3: Storage, Secrets & Managing your services](lesson-3.md) will show you how to use Rancher NFS for persistant storage, Rancher Secrets to store files and other sensitive data, and will also cover more advanced upgrades.
