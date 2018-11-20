# Spin Getting Started Guide: Lesson 1: Building your first application on your laptop

## About these lessons

As stated in the [Spin Getting Started Guide overview](index.md). Spin is the
container service platform at NERSC. The Spin project is based on Docker
container technology, and aims to follow the Docker motto, "Build, Ship & Run."
This means you will normally **Build** the image on your workstation, **Ship**
it to the Spin Registry, and then **Run** the container on Spin. In these
guides, we'll explain how to do that.

This guide is broken into multiple lessons.

* Lesson 1, presented here, will show you how to run applications
  using Docker on your laptop and can done at home or in a lab with an
  internet connection. If you are already familiar with Docker, we
  encourage you to at least read the section below about running
  containers as a non-root user.
* [Lesson 2: Running Your Application in Spin](lesson-2.md) shows how
  to migrate your application from your laptop to Spin and is intended
  to be run as a NERSC Hands-On lesson with access to the NERSC
  systems.
* [Lesson 3: Storage, Secrets & Managing your services](lesson-3.md)
  shows upgrading, using Rancher NFS for persistent storage, and using
  Rancher Secrets to store sensitive data.

These lessons are written with Mac users in mind. While the guide does
not cover Microsoft Windows, Windows users should be able to use
similar concepts to accomplish the goals. After finishing these
lessons, be sure to consult
the [Spin Best Practices Guide](../best_practices.md) if you have more
questions.

### Docker command syntax

These tutorials make use the Rancher and Docker command line (CLI) tools. The
[Docker 1.13 CLI introduced a newer
syntax](https://blog.docker.com/2017/01/whats-new-in-docker-1-13/#h.yuluxi90h1om)
in the form of **`docker [object] [verb]`**, such as `docker image list`,
`docker image build` and `docker container start`. This guide uses the newer
style of commands for clarity. The legacy style of commands, such as
`docker rmi` and `docker stop`, will also work.

## Prerequisites for the Spin Getting Started Guide

### Read the Overview

Before proceeding with this guide, we recommend that you read [Spin Getting
Started Guide overview](index.md). This will help you learn the requirements for
running in Spin, and will direct you to Docker's own Getting Started Guide to
become familiar with Docker.

### Get account on Spin, and a Project directory for Lesson 2

The next lesson, [Spin Getting Started Guide: Lesson 2: Running Your Image in
Spin](lesson-2.md), requires an account on Spin. To do that, please see the
[Spin Getting Started Guide: How do I get started](index.md)? You will also need
SSH access on a NERSC Login node, such as cori.nersc.gov, and access to a
Project directory. You will need the ability to modify the permissions on that
Project directory (Specifically, you will need to run `chmod o+x` on that
directory as described in Lesson 2.)

If you are waiting for your NERSC account to be set up, note that Lesson 1 can
be completed on your laptop without a NERSC account.

### Install Docker on your laptop

The first step is to install Docker on your laptop. We offer a brief lesson
here. For a more detailed guide, see [Docker's Get Started, Part 1: Orientation
and setup](https://docs.docker.com/get-started/).

Download Docker Community Edition (Docker CE) from https://www.docker.com/community-edition:

* Mac users, please see
  https://docs.docker.com/docker-for-mac/install/ and install "Docker
  for Mac (stable)".
* Windows users, please see
  https://docs.docker.com/docker-for-windows/install/ and install
  "Docker for Windows".

Now that Docker is installed, run a quick test to ensure that it's
working. Run this command:

    docker container run hello-world

You should see output like the following:

    elvis@laptop: $ docker container run hello-world

    Hello from Docker!
    This message shows that your installation appears to be working correctly.

    To generate this message, Docker took the following steps:
     1. The Docker client contacted the Docker daemon.
     2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
     3. The Docker daemon created a new container from that image which runs the
        executable that produces the output you are currently reading.
     4. The Docker daemon streamed that output to the Docker client, which sent it
        to your terminal.

    To try something more ambitious, you can run an Ubuntu container with:
     $ docker run -it ubuntu bash

    Share images, automate workflows, and more with a free Docker ID:
     https://cloud.docker.com/

    For more examples and ideas, visit:
     https://docs.docker.com/engine/userguide/

    elvis@laptop: $

Docker is now installed on your laptop, and you are ready to proceed.

## About Our Application

In the examples below, we will be building a simple application stack
consisting of two services: An **application service** based on Python
Flask to serve dynamic data, and a **web service** based on Nginx
which sits in front of the application service.

Docker recommends building the stack so that each service be stored in it's own
container, following a software design principal called "separation of
concerns". This allows each container to do one thing, and do it well: A Python
Flask container can focus on what it does best, which is to serve an dynamic
application and not worry so much about handling the complexities of web
connections. Likewise, a Nginx webserver can do what it does best, which is to
serve web content securely and scale easily. We recommend placing a web
container in front of a webapp container to improve security.

## Part 1: Build a Docker Image & Run a Container Based on the Image

Now that Docker is installed on your laptop, let's build an image for the first time.

As explained in the [Docker Get Started, Part 1: Orientation and
setup](https://docs.docker.com/get-started/) (Part of Docker's own
documentation), a Docker **image** is a stand-alone, executable package that
includes everything needed to run a piece of software. A **Dockerfile** is a set
of written instructions which is used to assemble the Docker image. Afterwards,
we will use that image to run a **container**.

The steps to build an image include:

* Create the working directory to store your files.
* Create a Dockerfile to define the image. You might download a file
  from GitHub using `git clone`.
* Add any prerequisites & content for your image, such as files,
  images or application code required for your web app.
* Build the image.
* Test the image by running a container based on your image.

For this example, we will build an image named 'my-first-container',
which will install a simple Python Flask app and is based
on
[Docker's own Flask tutorial](https://github.com/docker/labs/blob/master/beginner/chapters/webapps.md).
We will install Flask and add some simple code. Later we will rename
this image using Docker 'tags' and will ship it to the Spin Image
Registry where it can be deployed as a container on Spin. If you have
your own application, you can try to build it by following the general
instructions here. At this point, we encourage you to keep it simple.

### Step 1: Create the parent working directory

First, we need to create a parent working directory to store your
files. These lessons will use '~/docker/my-first-container' in your
home directory, but you can place it whereever you prefer. Open a
terminal and create these directories:


    mkdir ~/docker
    mkdir ~/docker/my-first-container
    cd ~/docker/my-first-container

### Step 2: Add Application Code

Next, we want to add a custom Flask application to this image. Create
a place to store your application code:

    mkdir app
    cd app

Open a file named 'app.py' in your favorite editor and add content
similar to the following. Note that the indenting in this file is
important.

    import os
    from flask import Flask
    app = Flask(__name__)
    @app.route('/')
    def hello_world():
        return 'Hello %s!' % (os.environ.get('WHO', 'WORLD'))

    if __name__ == '__main__':
        app.run(debug=True,host='0.0.0.0')

This simple Flask application will listen on the TCP port 5000 (the
default) and will print 'Hello WORLD!' when called normally. The app
will instead print the contents of the `WHO` environment variable if
it's defined (more on that later). Flask will also print debug
information to the log.

### Step 3: Build the image

In this step, we will create a Dockerfile and use that Dockerfile to
build a Docker image.

#### Create the Dockerfile

You should already be in the 'app' subdirectory, which will be the
working directory for this image. Open a file called "Dockerfile"
using your favorite editor.

    vi Dockerfile

Add the following text to the Dockerfile, and then exit your editor.

    FROM debian:latest
    RUN apt-get update --quiet -y && apt-get install --quiet -y python-flask
    WORKDIR /app
    COPY app.py /app
    ENTRYPOINT ["python"]
    CMD ["app.py"]

When Docker builds the image, Docker will run the commands in this Dockerfile
from top to bottom, and will do the following:

1. Create your image from the Debian:latest base image.
1. Update all packages in the image, in case they are out of date.
    * Note that the output from `apt-get` will be very verbose
      here. The `--quiet` flag reduces the noise a bit.
1. Install python-flask and any dependencies.
    * Note that we run this command on the same line as `apt-get` to
      reduce the number of image layers. Image layers are discussed
      more in the [Spin Best Practices Guide](../best_practices.md).
1. Copy the contents of the ./app/ directory from your laptop into the
   image at the location /app.
1. Instruct the container to run the app.py application, out of the
   /app directory, using Python when the container is started.

You should now have a simple set of files for your application:

    elvis@laptop:app $ tree
    .
    ├── Dockerfile
    └── app.py

    0 directories, 2 files
    elvis@laptop:app $

#### Build the image from the Dockerfile

Now that your application is in place, and the Dockerfile is ready, we
can build the image and save the copy on our laptop. Run the following
command, which will build an image using the Dockerfile in the current
working directory ".", and will 'tag' the container with the name
"app:latest".

    docker image build --tag my-first-container-app .

The command will produce output similar to the following. 'docker
image build' can print a large amount of logs, and we removed some
here for brevity.

    elvis@laptop:app $ docker image build --tag my-first-container-app .
    Sending build context to Docker daemon 4.096kB
    Step 1/6 : FROM debian:latest
    ---> 72ef1cf971d1
    Step 2/6 : RUN apt-get update --quiet -y && apt-get install --quiet -y python-flask
    ---> Running in b64e4f148838
    Ign:1 https://deb.debian.org/debian stretch InRelease
    ...
    ...
    Removing intermediate container b64e4f148838
    ---> 8cb8bf647f22
    Step 3/6 : WORKDIR /app
    Removing intermediate container 4f84c603d6c0
    ---> 3202f9d5feef
    Step 4/6 : COPY app.py /app
    ---> 5483884646ec
    Step 5/6 : ENTRYPOINT ["python"]
    ---> Running in cec8bb2863cb
    Removing intermediate container cec8bb2863cb
    ---> fe439bd53ad5
    Step 6/6 : CMD ["app.py"]
    ---> Running in 7532fc9c99ff
    Removing intermediate container 7532fc9c99ff
    ---> c4f1cd0eb01c
    Successfully built c4f1cd0eb01c
    Successfully tagged my-first-container-app:latest

(If you re-run the command in the future, you may see a less output
and messages showing that Docker is using a cache to rebuild).

Now, list your Docker images using the command `docker image list
my-first-container-app` and you'll see your container. Note how your
image is named "app" (The Docker CLI also calls this a 'Repository'),
followed by a a Docker version 'Tag' of :latest. We'll cover
Repositories & Docker Tags more later. For most projects, we recommend
that you give your image a name. If you don't specify the name, all
future commands must refer to the image by it's Image ID
('c4f1cd0eb01c' below) instead of the name.

    elvis@laptop:app $ docker image list my-first-container-app
    REPOSITORY              TAG     IMAGE         ID  CREATED  SIZE
    my-first-container-app  latest  c4f1cd0eb01c  36  seconds  ago   165MB
    elvis@laptop:app $

### Step 4: Run a container based on the image

Before you use your container in a real environment, such as Spin,
it's always a good idea to test this locally on your laptop.

Run the following command to start a container based on your new
image. The flag `--publish 5000:5000` will map the containers internal
port of '5000' (Flask's default port number) to port '5000' on your
machine for testing from a web browser. The `--rm` flag will remove
the container completely when we stop the container so that there
aren't stale containers left behind during testing.

    docker container run --rm --publish 5000:5000 my-first-container-app

Open up a web browser, and browse
to [http://localhost:5000/](http://localhost:5000/) . The browser
should display the text "Hello World!" from your Flask code
above. Docker will print logs similar the following. The container
startup logs are shown first, followed by the application logs:

    elvis@laptop:app $ docker container run --rm --publish 5000:5000 my-first-container-app
    * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
    * Restarting with stat
    * Debugger is active!
     Debugger PIN: 160-224-640
    172.17.0.1 - - [12/Jan/2018 01:18:10] "GET / HTTP/1.1" 200 -

Hit **"Control C"** to exit & stop this container.

Congratulations, you built your first container using your own custom
content!

## Part 2: Extend the Nginx image to run as a non-root user

The next step is to create a webserver to sit in front of the web
application.  We generally recommend that you place a robust webserver
in front of your application server, as a webserver is generally going
to handle the complexities of web connections better than a custom
application.

For this image, we'll use the official Nginx image from Docker Hub,
which is a widely used, well respected, and robust service.  Since
we're using a preexisting image provided by the community, we only
need to concern ourselves with a few details about the webserver, and
that saves us time. We don't need to build a webserver from scratch.

In Lesson 2, we will be reading from the NERSC Global Filesystem. For
security reasons, any container which uses the NERSC Global Filesystem
must run as a non-root user. The Nginx image by default runs as the
'root' user by default, and we need to change this.

There are a couple methods to modify a container as non-root. We show
one method here, and discuss a few others in
the [Spin Best Practices Guide.](../best_practices.md) The Docker &
Linux communities are working on alternative ways to run stock images
without modification, and we're keeping an eye on that.

Switch back to the parent directory, and create a new working
directory for web container.

    cd ~/docker/my-first-container
    mkdir web
    cd web

Add the following Dockerfile. Notice how we make a few changes to
allow a non-root user to run the service, including listening on port
**8080** (Port 80 is only available to privileged users, such as
'root'. Unprivileged users can listen on any port above 1024),
changing the ownership of '/var/cache/nginx' and writing the Nginx PID
file to a different location.

    FROM nginx:latest

    # Make /var/cache/nginx/ writable by non-root users
    RUN chgrp nginx /var/cache/nginx/
    RUN chmod g+w /var/cache/nginx/

    # Run as port 8080, which is available to non-root users allows us to drop
    # all remaining root capabilities from the container, which improves security.
    RUN sed --regexp-extended --in-place=.bak 's%(^\s+listen\s+)80(;)%\18080\2%' /etc/nginx/conf.d/default.conf
    EXPOSE 8080

    # Write the PID file to a location where regular users have write access.
    RUN sed --regexp-extended --in-place=.bak 's%^pid\s+/var/run/nginx.pid;%pid /var/tmp/nginx.pid;%' /etc/nginx/nginx.conf

Build the image, similar to what we did above with the 'app'
image. We're keeping the 'nginx' in the name here to help distinguish
the kind of web software that we're using.

    docker image build --tag my-first-container-nginx .

#### Create configuration files for the Nginx container

The next step is to customize the Nginx container a little bit more so
it works with our app.

We'll do this by creating a custom Nginx configuration file on the
host which will be used in **Part 3** below.

Create a file named **nginx-proxy.conf**, and add the following
text. This will configure Nginx to behave as a function as a standard
'reverse proxy' listening on the internal port of 8080. The reverse
proxy will forward all traffic to an backend application server named
**"app"** listening on an internal port 5000.  These ports are only
available internally, and won't be exposed to the outside world.

    server {
        listen 8080;
        location / {
          proxy_pass http://app:5000;
          proxy_set_header Host $host:$server_port;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Host $server_name;
          proxy_set_header X-Real-IP $remote_addr;
        }
    }

How does Nginx know about the hostname **app**? We define it in our
Docker Compose files, as you'll see in the next step.

## Part 3: Create an Application Stack using Both Images

To combine multiple container services into a single, cohesive
application, you will use a **Docker Compose** file to assemble the
containers together into an "Application Stack". The Docker Compose
file will define **services** named **web** and **app**, their
published (public) ports, and any files which are mounted from the
host into the container.

Here, we'll combine our custom Nginx image and with the custom Flask
app that we built above into a single application stack.

!!! Note "Terminology: Containers vs. Services"
    Note the change in terminology here. When using `docker container run`
    above, we created a **container**. However, in a Docker Compose file,
    we are defining a **service**. A **service** may have one or more
    instances of itself, called **containers**. **Containers** are
    'instances' of a **service**.

### Define your application stack using docker-compose.yml

Switch back to the working directory:

    cd ~/docker/my-first-container

Create a file named docker-compose.yml and add the following
content. Again, the indenting in this file is important.

    version: '2'
    services:
      app:
        image: my-first-container-app
      web:
        image: my-first-container-nginx
        ports:
        - "80:8080"
        volumes:
        - ./web/nginx-proxy.conf:/etc/nginx/conf.d/default.conf:ro

The configuration will define an application stack with two services:

* An 'app' service, which is based on the 'my-first-container-app'
  image created above. This service has no public ports and is instead
  hidden away behind the Nginx proxy (which improves secuirity).
* A 'web' service, which is based on the 'my-first-container-nginx'
  image created above.
    * The Nginx container will provide a reverse proxy to our backend
      'app' application. The **./web/nginx-proxy.conf** file is
      mounted from the host directory into the container at
      **/etc/nginx/conf.d/default.conf** using a Docker "Bind Mount",
      which is read by Nginx by default.
    * There are other ways to get files into a container, such as
      using the 'COPY' or 'ADD' statements in the Dockerfile. Normally
      you want to build all application code into your image when it's
      complete, as that reduces dependencies on outside
      resources. During development, it can be simpler to mount a file
      into the container to allow quick and easy modifications.
* The hostnames 'app' and 'web' are created within the application
  Stack, and are available within any containers that are a part of
  this stack.
* The web service will listen publicly on port 80 for now, which maps
  to Nginx's internal port of 8080.

!!!Note
    If you are familiar with Docker Compose, note that Rancher
    only supports Docker Compose v2. Do not use Docker Compose v3.

### Validate your Docker Compose file

Typos in a Docker Compose file can cause major headaches. So before
proceeding, let's validate the Docker Compose file using this command,
which by default will check the file 'docker-compose.yml' in your
current directory.

    docker-compose config --quiet

If the Compose file contains no validation errors, the command will
print no output, like so:

    elvis@laptop:my-first-container $ docker-compose config --quiet
    elvis@laptop:my-first-container $

If there are validation errors, this command will print an
error. Here's what happens if my indenting is wrong, for example:

    elvis@laptop:my-first-container $ docker-compose config --quiet
    ERROR: yaml.scanner.ScannerError: mapping values are not allowed here
    in "./docker-compose.yml", line 7, column 10
    elvis@laptop:my-first-container $

## Part 4: Run the Application Stack

You should now have the following files in your working directory:

    elvis@laptop:my-first-container $ tree
    .
    ├── app
    │   ├── Dockerfile
    │   └── app.py
    ├── docker-compose.yml
    └── web
        ├── Dockerfile
        └── nginx-proxy.conf

    2 directories, 5 files
    elvis@laptop:my-first-container $

Now, start the application server with `docker-compose up`, and point
your browser to [http://localhost/](http://localhost/) (port 80). Your
browser should show yo the "Hello World!" screen as it did in the
previous example. Docker should print log output similar to the
following:

    elvis@laptop:my-first-container $ docker-compose up
    Creating myfirstcontainer_web_1 ... done
    Creating myfirstcontainer_app_1 ... done
    Attaching to myfirstcontainer_app_1, myfirstcontainer_web_1
    app_1 | * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
    app_1 | * Restarting with stat
    app_1 | * Debugger is active!
    app_1 | * Debugger PIN: 157-714-645
    app_1 | 172.27.0.2 - - [18/Jan/2018 00:35:15] "GET / HTTP/1.0" 200 -
    web_1 | 172.27.0.1 - - [18/Jan/2018:00:35:15 +0000] "GET / HTTP/1.1" 200 12 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36" "-"
    app_1 | 172.27.0.2 - - [18/Jan/2018 00:35:15] "GET /favicon.ico HTTP/1.0" 404 -
    web_1 | 172.27.0.1 - - [18/Jan/2018:00:35:15 +0000] "GET /favicon.ico HTTP/1.1" 404 233 "http://localhost/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36" "-"

As before, hit **Control-C** to exit.

Congratulations, you built and ran a multi-service application stack
composed of two Docker images.

## Extra credit - Environment variables

Our flask app accepts environment variables. To change the value away
from the default on the shell, open up the Docker Compose file, and
under the `app` definition, add an environment stanza as in the
following example:

    version: '2'
    services:
      app:
        image: my-first-container-app
        environment:
          - WHO=elvis
      web:
        image: my-first-container-nginx
        ports:
        - "80:8080"
        volumes:
        - ./web/nginx-proxy.conf:/etc/nginx/conf.d/default.conf:ro

Now, run `docker-compose up` again, and browse to http://localhost/ . The app
will say 'Hello elvis!' instead of 'Hello WORLD!'

## Next Steps: Lesson 2

The next lesson, [Spin Getting Started Guide: Lesson 2: Running Your
Application in Spin](lesson-2.md) will show you how to push your application
images into the Spin registry, run that application in Spin, as well as how to
do other administrative tasks.
