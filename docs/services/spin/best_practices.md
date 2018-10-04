# Spin Best Practices

!!!info "This guide is under development"
	Spin is in a Pilot phase for NERSC staff and users in early 2018. During this development period, the content of thees page may be updated, and some docs may be incomplete. We are working to convert all Spin documentation to this new system.

## Spin Overview & Steps to Get Started

For an overview of Spin and short tutorials to get started, please see the [Spin Getting Started Guide](getting_started)

## Security Audit

All applications sent to Spin are automatically audited at the API to ensure that they follow our security requirements, which are outlined in the Spin Implementation Guide. The Rancher CLI will print an error if an application breaks one of the security requirements.

## Using Images & Creating Containers

### Choosing the best image source

When building your own image, you will usually be pulling an official image from a major, reputable project.

In general, good images from [Docker Hub](https://hub.docker.com/) tend to be well maintained and have wide community support. We look for images which meet the following guidelines:

* Are part of the [official repositories](https://docs.docker.com/docker-hub/official_repos/), such as the [Docker Hub Official Repositories](https://hub.docker.com/explore/)
* Have a high number of pulls, indicating that the project is well used
* Have a high number of stars, indicating that the software works well
* Are updated as frequently as needed to address security vulnerabilities and to keep up to date with upstream features. Look for images with recent modification times, which indicates that the image is being kept up to date.

If the project page has a de facto image or a recommended image, that's usually the best and simplest option. The goal here is to keep the image simple, and yet still be functional enough to support your application.

There are also many low quality images on Docker Hub, but they tend to be obvious. Consider avoiding images that have a low number of pulls, are poorly rated, or lacks recent updates. Image size is another useful criteria. The appropriate size of an image will obviously vary depending on the application stack, but as a rule of thumb, take a close look at images > 5 GB to see if it contains a lot of unnecessary components. Images that are overly large, besides suggesting that they contain too many unnecessary elements, may be frustratingly slow to push to the image registry during the development cycle (especially over a typical home internet link), and will be slower to deploy.

Popular projects may have multiple images on their project page. The Apache httpd project has `httpd:2.4` and `apache:alpine`, which show that the Apache community is maintaining a mainline application while also experimenting with tiny images based on the Alpine container OS.

Examples of official projects that have provided a selection of images for different use cases are:

* [Apache](https://hub.docker.com/_/httpd/) (httpd)
* [Ubuntu](https://hub.docker.com/_/ubuntu/)
* [Nginx](https://hub.docker.com/_/nginx/)

### Operating Systems

There are official images available on Docker Hub for nearly every popular Linux distribution. If you are basing your image on a prebuilt container, such as Apache httpd, your choice will be made by the original image developers. If you are building your own image, there are a number of criteria that can be helpful in guiding your choice.

OS Considerations:

* Utilizes a package manager and/or is a distribution that is familiar to you
* [Alpine](https://hub.docker.com/_/alpine/) ([Recommended by Docker](https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/#the-dockerfile-instructions), starts as a very small base image)
* [Debian](https://hub.docker.com/_/debian/) (Commonly used in the Docker community. Balances size with usability.)
* [CentOS](https://hub.docker.com/_/centos/) (for an RPM based distribution, and as a replacement for Scientific Linux, which does not currently have an official Docker image)
* [OpenSUSE](https://hub.docker.com/_/opensuse/) (if similarity to the super computer environment is a consideration)

### Dockerfile

When writing a Dockerfile, seek a balance between readability and size. Familiarize yourself with Docker's [Best practices for writing Dockerfiles](https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/). Some general rules:

* Containers should be ephemeral
* Each container should have only one concern
* Avoid installing unnecessary packages
* Keep the image small by reducing the number of layers. This can be accomplished by condensing operations into a single step (e.g. Single yum command with multiple packages vs. multiple yum commands), by chaining commands with ‘&&’. Consider using [Docker multi-stage](https://docs.docker.com/develop/develop-images/multistage-build/) builds with Docker 17.05 or higher.
* Improve caching and shorten build time by ordering statements such that the stable parts of the Dockerfile are at the beginning, and the more frequently changed statements are at the end

## Image Registry

Local images for Spin must be stored in the associated registry service, https://registry.spin.nersc.gov, or come from directly from Docker Hub (preferably only images produced by official projects will come from Docker Hub). No other registries may be used as the image source for Spin.

The local registry is organized along the following lines:

* **Teams** are used to group people that are working on the same project. Although the registry doesn’t yet use LDAP to define these, team names should match a group name in LDAP.
* **Namespaces** are analogous to directories, and contain sets of related repositories. They are owned by teams, and team membership and roles define who can view and write to them. By convention, the default namespace should use the team name.
* **Repositories** are where the docker images are actually stored.

Everyone with access to the Spin interface also has access to registry.spin.nersc.gov. If a Team and Namespace hasn’t yet been set up for your group, make a request via Service Now.

The process of tagging and pushing an image to the registry is described in the [Spin Getting Started Guide](getting_started).

### Be careful using the `:latest` tag

Be careful when using the `:latest` tag, as it confuses many new (and
experienced!) users, and may not work the way you expect.

Contrary to the name, the label `:latest` is ambiguous, and may actually be
'latest' from the last time you downloaded the code, 18 months ago.

For example, if you run a service based on `someimage:latest`, the Docker
daemon will only download that image if it's not present in the local image
cache.

If a new version of `someimage:latest` is uploaded to Dockerhub or the Spin
registry, Docker has no way to know that the image was updated on the remote
registry. As far as the Docker daemon on the node is concerned, it already had
the `:latest` image cached on the node, and therefore doesn’t need to check the
registry.

Note that if you update an image, and re-use a version tag, Docker will still use
the cached image on the node.

Furthermore, remember that `:latest` changes over time. If your service has
replicas on multiple Docker hosts, one replica may be running `:latest` from
September, while a second node may be running `:latest` from July. 

We recommend using explict version numbers, such as `:v1.2.3` or a date format
such as `:v20180809` instead of `:latest`, an that you update the tag for any changes.

If you do use `:latest`: in your service, you can also use the label
`io.rancher.container.pull_image: always` to tell Docker to always pull the
latest version your `:latest` image. Note that a download will add a short
delay to upgrade operations.

## Version Control

Storing the Dockerfile, docker-compose.yml (if it exists), and any other files associated with a container-based application is helpful for all of the same reasons that version control is useful for other projects. In addition, during the pilot phase, container application developers frequently work with the Spin administrators to migrate their projects into this environment. Version control systems facilitate these collaborations. Frequently used systems include:

* https://bitbucket.org academic account - Academic users are upgraded to an account with unlimited public and private repositories. The upgrade is automatic and based on the email address used to register the account.
* https://gitlab.com - Comes with comprehensive CI/CD features; it is a common choice for JGI projects
* https://github.com - Less commonly used as the free account tier doesn’t include private repositories, and the academic accounts are more limited and without the automatic upgrade feature of Bitbucket.
* NERSC Staff can also use the internal Bitbucket service.

Regardless of which system is being used, avoid storing secrets such as database passwords or private keys in files that get pushed to a repo; instead the [Secrets](#secrets) feature should be utilized.

## Services

One or more identical containers providing the same function in Rancher is termed a ‘Service’.

### Naming Convention for Services

* Common services should use a name from the following table of recommended names

  | Name | Description        |
  |------|--------------------|
  | app  | Application server |
  | db   | Database           |
  | lb   | Load Balancer      |
  | web  | Web Server         |
  | kv   | Key Value store    |

* Unique services should use a descriptive name for the service they provide. It should be a general description of the functionality, rather than the specific name of the implementation (e.g. web rather than apache)
* As new service types become common, they will be added to this table of recommended names.
* Standardizing service names has the benefit of clearly communicating the purpose of each service. This is beneficial when collaborating with others, when revisiting a service created in the past, and when enlisting the help of NERSC staff during troubleshooting.

## Stacks

The collection of services that comprise an application within Rancher is termed a ‘Stack’.

### Naming Convention for Stacks

Stacks should be named to match to non-TLD components of public-facing name (eg foo-jgi-doe for foo.jgi.doe.gov and bar-nersc for bar.nersc.gov)

### Stack Tags

Tags are created within the Rancher environment to label stacks with information that can be useful for identifying ownership, support and resource usage. Some tags are optional, while others are required for all stacks.

| Tag | Status | Description | Example |
|-----|--------|-------------|-------- |
| owner: | Required | Developer or primary user of stack. Must be in the form of a NIM username | owner:fred |
| staff-owner: | Required | NERSC staff member that is most familiar with or contact person for application  (similar semantics to staff-owner in Service Now). Must be in the form of a NIM username | staff-owner:wilma |
| group: | Recommended | Group that owns the stack. All members of group have permission to update or restart services in stack. Should be an LDAP group | group:csg |
| staff-group: | Recommended | NERSC group most familiar with or contact for application | staff-group:isg |
| fqdn: | Recommended | Public facing DNS Name | fqdn:foo.nersc.gov |
| requires: | Optional | Specifies dependencies of the stack, for example external file systems | requires:gpfs |

## Storage

A number of different types of storage are available to containers running in the Spin environment. Docker concepts such as Volumes, Bind Mounts and tmpfs volumes are explained at https://docs.docker.com/storage/ . A brief summary of the different types of storage and their properties is presented in the table below, with the following column headings and their meanings:

* **Persistent** - Is the data in the volume preserved when the container is destroyed & recreated
* **Portable** - Is the data in the volume available when the container is restarted on a different node
* **Performance** - A relative measure of the performance category of the storage
* **Auto-created** - Is the source directory auto-created by Spin, or must it pre-exist to be mounted
* **Externally Accessible** - Is the data available outside of the Spin environment

    TODO insert table here

### Container Storage

Storing data to the container file system should only be used for small amounts of ephemeral data. The data is lost whenever the container is restarted with a fresh image, which can happen in a number of scenarios (container restarted on different node, container upgraded with new image, etc.)

### Local Node Storage

Each node within Spin has 1.7 TB of storage available to containers. Because it is provisioned on SSD, this storage is relatively fast. Because it is local to a node, any data previously written will not be available to a container if it’s restarted on a different node. Therefore, this storage is most useful for applications that read/write lots of data that is considered transient or disposable (for example an application cache).

### Rancher NFS

Rancher NFS is a storage class residing on NFS servers and available only from within the Spin environment. It is appropriate when an application needs persistent storage, that will be available to containers even if restarted on a different node. The storage is not available from outside of Spin, so it’s not a good choice when data needs to be part of a pipeline that has components outside of Spin, or when users expect to have direct access to the data files from login nodes. Rancher NFS does have nice built-in lifecycle features that can optionally create/destroy data directories to match the life cycle of an application stack.

### Global File System / GPFS

General guidelines for using GPFS within Spin

* The mount point should be as deep in directory structure as possible
* e.g. /global/project/scratch/username/project/application rather than /global/project/scratch
* The volume should be mounted read-only, unless the container actually writes to the global file system.
* The filesystem must have the execution bit set for ‘other’ (o+x) on parent directories all of the way down to the mount point for the docker daemon to successfully mount the directory. For example permission mode 0741 would work on a parent directory, but 0740 would not.

### Permissions, UID and GID Considerations

Permissions set on the global file systems must be respected and enforced when files are accessed from within a container. This is accomplished through a combination of container configuration and external controls enforced by the docker daemon. This leads to several considerations when using the global file system within Spin.

* The user ‘root’ in a container maps to user ‘nobody’ on the global file systems, which places significant restrictions on the data that can be accessed from a container in a default configuration.
* Setting the container to run as a non-root account with the appropriate file system access is an effective way to address these permission constraints.
* Using a collaboration account with the necessary file system access is an effective way to ensure data access while also avoiding issues that occur when the owner of a personal account leaves a group or project.
* When a container can’t easily be modified to run as a non-root user, the container can often be run with the group set in a manner that provides access. For example, a container running as root:genome will successfully read files in a directory with the following restrictive permissions:

        dino@genepool13:~$ ls -ld scratch/dino
        drwxr-x--- 5 dino genome 512 Sep  8 12:00 scratch/dino

* The Linux setuid and setgid capabilities will be dropped for containers accessing the global file system as discussed in the [Security](#security) section
* Configuring the user or group that the containers will run as, and configuring capabilities will be performed by ISG administrators during the Spin pilot phase as part of the initial stack setup.
* Images that will be run as a different user or group will need RUN statements as shown in the following example to prepare the image with the necessary group and user definitions.

#### Dockerfile Statements for setting a container group using useradd & groupadd

This example illustrates the Dockerfile RUN statement for an image to run with the group ‘genome’ (gid 124).

    # Add a genome group to facilitate access to global file system
    RUN groupadd -g 124 genome

#### Dockerfile Statements for setting a container user and group

This example illustrates the Dockerfile RUN statement for an image run as collaboration account ‘c_flintstones’’(uid 501) and group ‘genome’ (gid 124).

    # Add collab account and group to facilitate access to
    # global file system
    RUN groupadd -g 124 genome && \
      useradd -u 501 -g 124 -c 'Collaboration Account' c_flintstones

The ampersands (&&) in this example minimize the layers created in the docker image. The cases would be configured to run as “root:genome” and “c_flintstones:genome”, respectively, during the initial stacks configuration.

### Read-Only vs. Read-Write

* Public (unauthenticated) services must mount the global file systems read-only
* Authenticated services are allowed to mount the global file systems read-write. The authenticated application must include the capability of tracking the username responsible for creating/modifying/deleting data on the global file system. Writes need to be traceable to the user doing the writing

## Logging

* Logging strategies for container-based services may need to be modified for applications developed in a more traditional environment.  Files written to the container’s file system aren’t easily accessible, and also aren’t persistent across container restarts or upgrades. There are several approaches that have proven useful in Spin:
* Log to stdout and stderr rather than writing to a file in the file system. If the service needs just one log, it can write to stdout. If it needs two logically separate log streams, it can write to stdout and stderr. In cases where more than one log stream, the container should be started without the -i or -t flag so that stdout and stderr are not combined. These logs will be persistent, but as they can only be accessed via Rancher or a docker command on the Spin hosts, access to the logs must be coordinated with ISG staff  during the pilot phase.
* Write to a persistent log volume hosted outside of the Spin environment (e.g. A global project directory). This will facilitate direct access to log information.
* Log to central logging system (future capability)

## Networking

### Ports

Services within a stack can communicate with each other on all network ports without mapping ports or modifying [firewall rules](#firewall-configuration). If a port is to be available outside of a stack, a mapping of a public Spin host port to the private container port must be created. Recommendations on port usage include:

* All use of ports 80 and 443 must go through the [reverse proxy](#httphttps-reverse-proxy)
* Whenever possible, use a port number > 1024 within the container. This allows the NET_BIND_SERVICE capability to be dropped.
* When practical (within the above rules), use the default or well known port for the service, to help convey additional context about your service.
* If the port you need isn’t currently part of the [existing firewall configuration](#firewall-configuration), and it needs to be publicly accessible, request that it be added to the list via Service Now.
* Requests for privileged ports (ports greater than 1024) must be reviewed/approved by the Spin working group before changes to the firewall are made.
* If you don’t care about the port number, and don’t want to request a firewall change, There are several port ranges open and available for use. Ports 50,000 - 50,050 (inclusive) are open to traffic with source addresses within the NERSC network, and ports 60,000 - 60,050 are open to traffic from all source addresses. Try to make a random(ish) selection within this range to minimize the chance of a port scheduling conflict on the Spin nodes. (Basically, your service won’t be able to run if the scheduler can’t find an IP on a Spin node that isn’t already listening on this port, and the chances of this decrease if everyone doesn’t pick the same port).

### HTTP/HTTPS reverse proxy

All access to ports 80 and 443 is achieved via a reverse proxy running in Spin. This simplifies applications, as the SSL certificates and TLS configurations are centrally managed, and it also conserves IP address space.  All incoming HTTPS connections terminate at this proxy rather than at the endpoint service. As part of the initial stack configuration, ISG staff will configure the reverse proxy stack to direct the appropriate traffic to your web service.

### Firewall configuration

The following TCP ports are publicly available from all source addresses:

    80, 443, 8080, and 8443, 60000 - 60050

The following TCP ports are available only from within NERSC networks (128.55.0.0/16) as well as from LBL networks that are secure and authenticated, such as the employee wireless and LBL VPN.

    3128, 3306, 5432, 5672, 8008, 50000 - 50050

The following TCP ports are available only from within NERSC networks:

    4873, 8081

## External DNS

Services that listen on port 80 or port 443 are accessed via a reverse proxy service. Typically a DNS CNAME record would be added to the appropriate domain (nersc.gov, jgi.doe.gov, etc.) pointing to the reverse proxy FQDN for the environment being used:

* Production:  lb.reverse-proxy.prod-cattle.stable.spin.nersc.org
* Development:  lb.reverse-proxy.dev-cattle.stable.spin.nersc.org

Services running on other ports will have a dynamic DNS entry automatically created for them when an external port mappings is created in Spin. The DNS name will be of the form:

    <service name>.<stack name>.<environment>.stable.spin.nersc.org.

For example, a database service in the stack named ‘mystack’ in the production environment would get the name db.mystack.prod-cattle.stable.spin.nersc.org. Similarly to web services, a CNAME record would be added to nersc.gov or jgi.doe.gov domains to point to to the dynamically created FQDN to provide a more convenient or memorable name for accessing the service.

## Security

Docker containers are fairly secure by default. This security is achieved through the use of Linux kernel 'namespaces', isolated network stacks, Control Groups, and whitelisting the Linux kernel 'capabilities' to only those needed. Docker security is a big topic. For a good summary explaining the current security features of Docker, read [Docker security](https://docs.docker.com/engine/security/security/) in the Docker manual.

AppArmor and SELinux security policies on Ubuntu & CentOS will be enabled on Spin in the future.

To enhance security of your containers, we recommend:

* When possible, run services in the container as a non-root user. Many of the reasons that a process would need escalated privileges (direct access to hardware, writing to a particular directory, binding to a low numbered port) don’t apply in a or can be avoided in a containerized environment. For example, a service can bind to a high numbered port, and then let docker map the privileged port on the docker host to the unprivileged port on the container. Similarly, volume mounts to a persistent volume with the desired permissions can avoid some of the permission hurdles.

* Just as with a traditional server, if a container conducts a mix of
  privileged and unprivileged operations, it can implement [privilege
  separation](https://en.wikipedia.org/wiki/Privilege_separation), and drop
  privileges after the privileged operations have been completed.
* If it’s not possible to run as a non-root user, minimize the [Linux
  capabilities](http://man7.org/linux/man-pages/man7/capabilities.7.html)
  granted to the container. In most cases, a container can drop all
  capabilities, and only add back one or two that are actually needed by the
  container. The [initial set of capabilities that Docker
  uses](https://github.com/moby/moby/blob/master/oci/defaults.go#L14-L30) is
  small enough that reviewing the list of what’s needed by a specific
  application isn’t an onerous task. Experience has shown that many containers
  (if not most containers) don’t actually need any of these capabilities.
* If your service uses external file systems (like the global file system), it
  will be required to run as a non-root user, and drop many Kernel capabilities.
  This allows existing ownership and permissions on the filesystem to be
  effectively enforced within Spin.

### Allowed Kernel Capabilities

The following chart shows which capabilities are allowed for Spin containers,
and Spin containers which uses the NERSC Global Filesystem:

| Permission    | No External Filesystem | External Filesystem | Description |
|---------------|------------------------|---------------------|-------------|
| CHOWN         | Yes | No | Make arbitrary changes to file UIDs and GIDs (see chown(2)). |
| DAC_OVERRIDE  | Yes | No | Bypass file read, write, and execute permission checks |
| FOWNER        | Yes | No | Bypass permission checks on operations that normally require the file system UID of the process to match the UID of the file |
| KILL          | Yes | No | Bypass permission checks for sending signals |
| SETGID        | Yes | No | Make arbitrary manipulations of process GIDs and supplementary GID list |
| SETUID        | Yes | No | Make arbitrary manipulations of process UIDs. |
| NET_BIND_SERVICE | Yes | Yes | Bind a socket to internet domain privileged ports (port numbers less than 1024). |

* Detailed desciption of each capabilities can be found at http://man7.org/linux/man-pages/man7/capabilities.7.html

## Secrets

Rancher Secrets are a mechanism for storing encrypted copies of sensitive items such as database passwords and SSH keys that are needed by a container at runtime. Storing them as Rancher secrets obviates the need to store sensitive information as a file in your Docker development directory or as an environment variable (which is exposed in the docker-compose.yml file), and helps prevent the information from ending up in the image registry or in a source code revision control repository.

### Properties of Secrets

* Stored in encrypted form within the Spin infrastructure
* When attached to a container, they are available in unencrypted form in a file mounted as `/run/secrets/secretname`
* Secrets are arbitrary files that can contain anything that is considered sensitive. Examples of secret files: certificates, config files that contains sensitive passwords, environment files with sensitive information. It is up to the application to read and interpret the secret file.
* Must be entered into the Rancher UI by an ISG administrator (during the pilot phase)

If an application requires a specific path to the secret, a symbolic link can be made to the file stored in /run/secrets/. Even if only one component of a configuration file is sensitive, the entire contents of the configuration file can be pasted into a secret to protect the sensitive component.

### Naming Convention for Secrets

Following the Spin naming convention will help identify secrets related to your stack, and also aid in the overall stack lifecycle management.

#### Single Service Secret Naming

Secrets used for a single service should be named:

    <service name>.<stack name>.<filename>

Wherever possible, the filename should indicate how the secret is used. For example, a MySQL password within a stack named ‘My Portal’ would be:

    db.myportal.mysql_password

#### Multi-Service Secret Naming

If the secret is used by a number of services within the stack, the service part of the name can be dropped. Leaving the secret name as:

    <stack name>.<filename>

For example an SSH private key that is used for multiple components within a stack named ‘My Portal’ would be:

    myportal.private_key

### Secret Description

When creating a secret, the description should always indicate the secret’s owner, by adding owner:<nersc username> to the description field.

### Adding Secret to Container

When adding a secret to a container in the ‘Secrets’ tab:

* Set ‘As Name’ to the filename component of the secret name. In the above multi-service secret example, the ‘As Name’ field would be set to ‘mysql_password’, and the secret would be available in the file /run/secrets/mysql_password.
* Customize the file ownership and permissions to restrict read permissions within the running container. In general the file owner should match the UID that the service is running as, and the mode should be set to 400.

## Future Directions

### User Interface

A current limitation of the Spin environment is a lack of access to the Rancher environment for service developers. This is necessary for the time being, because the existing version of Rancher does not provide true multi-tenancy, and therefore would provide root-equivalent privileges to anyone with access to the interface to all containers running in the environment. Until Rancher provides an acceptable multi-tenant, an alternative interface is being provided to that enables the following operations on stacks that they own:

* Stop services
* Restart services
* Upgrade a container
* Access container logs
* Access a container shell

### CI / CD Pipeline

A continuous integration / continuous deployment pipeline that would automate some of the steps involved in building images, pushing them to the registry and deploying them in Spin is in the early planning stages.

### External Service Access

Streamlining and standardizing the interfaces to Slurm, HPSS and other NERSC resources is recognized as a commonly desired feature, and will be added to Spin in the future.

## Rancher Idiosyncrasies

### Differences between Rancher and laptop Docker environment

Although the ‘build, ship, run’ paradigm maps very well to developing a service on your laptop and shipping it to the Spin environment, there are some differences between the environments. These are typically fairly straightforward to address, but it’s helpful to be aware of their existence.

* There is no direct analogy for Rancher Secrets in a laptop Docker environment
* It can be difficult to simulate the global filesystem and/or copy data when prototyping on a laptop
* Rancher only supports up to version 2 of the Docker compose file format

### Pulling an updated image from registry on restart - (Stefan’s container crash example)

When a container is restarted in Spin, it may or may not first pull the image from the registry. If developers aren’t mindful of how they update their images in the registry, they might inadvertently be in a situation where their image is deployed into production before they intended. Observed behavior in the Spin environment:

* If a container is restarted (due to a failed health check or manual operation) and Rancher chooses to schedule it on the same Spin node, a fresh copy of the image is not necessarily pulled from the registry
* If a container is restarted and Rancher chooses to schedule it on a different Spin node, a fresh copy of the image will be pulled from the repository
* If the ‘Upgrade’ operation is performed, Rancher will pull a fresh copy of the image from the registry even if no properties have changed.
