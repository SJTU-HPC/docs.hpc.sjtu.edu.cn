# Science Gateways

## About Science Gateways

A science gateway is a web-based interface to access HPC computers and
storage systems.  Gateways allow science teams to access data, perform
shared computations, and generally interact with NERSC resources over
the web. Common gateway goals are

* to improve ease of use in HPC so that more scientists can benefit
  from NERSC resources
* to create collaborative workspaces around data and computing for
  science teams that use NERSC
* to make your data accessible and useful to the broader scientific
  community.

NERSC encourages its users to create their own science gateways by
using the resources described on this page. The center engages with
science teams interested in using web services, assists with
deployment, accepts feedback, and tries to recycle successful
approaches into methods that other science teams can benefit
from. Below you will find links to current projects and details about
the building blocks available to NERSC users. If you would like to
participate, or if you have questions, please contact
consult@nersc.gov.

## Science Gateway Availability and Support

Developers of science gateway applications hosted at NERSC should be
aware that if their gateways critically depend on NERSC infrastructure
then their gateways will inherit availability from NERSC's underlying
infrastructure to some degree.  Some examples:

* If the /project file system is out of service for multiple days and
  a science gateway uses scripts, HTML templates, or other web content
  stored on /project then the site will not work for the same period
  of time as the /project outage.
* In contrast, applications that only depend on data files stored
  (e.g. on /project) will not have the functionality the data files
  make possible.  In such cases it is up to maintainers of gateways to
  inform their users of any degradation in service.
* Applications that submit jobs to one of the supercomputer system
  queues via NEWT (see below) may be unable to support that
  functionality for the duration of the a system outage; however
  proper use of the API can mean a web site functions just with
  decreased functionality.  It is up to gateway maintainers to handle
  graceful failures in such cases.

Science gateway application developers should keep in mind that
NERSC's goal is to make sharing scientific data and high-performance
computing resources over the web practical, but NERSC is not a web
hosting service.  Developers and users should not anticipate
availability approaching what is available in commercial offerings.
The NEWT API provides one avenue for users who require >99% uptime for
a web presence that exposes NERSC resources like data or job
submission.  This offers a clean separation between the web
application and NERSC infrastructure that can be managed by
developers.

If a science gateway or website does not clearly depend on NERSC
resources or data, we encourage users to pursue other hosting
solutions.  For example, [Google sites](https://sites.google.com) is
an excellent alternative for users seeking to establish merely a web
presence for their project.  Such simple websites are not within the
scope of science gateways at NERSC, and we do not provide support to
users attempting to set them up.
See
[Wikipedia's comparison of free hosting services](https://en.wikipedia.org/wiki/Comparison_of_free_web_hosting_services).

The service level for NERSC science gateway support is formally 8x5
(business hours).  Outside of those hours the NERSC Data and Analytics
Services staff provide support on a best effort basis.

## Gateway Technologies

NERSC provides science teams with the building blocks to create their
own science gateways and web interfaces into NERSC. Many of these
interfaces are built on web and database technologies.

### Web Methods for Data

Science gateways can be configured to provide public unauthenticated
access to data sets and services as well as authenticated access, if
needed. The following features are available to projects that wish to
enable gateway access to their data through the web. Other features
can be made available on request. Direct access to the NGF "project"
filesystem and HPSS tape archives are described in the table below.

There are two science gateway nodes that are available to all NERSC
users. These are portal and portal-auth. They function very similarly
but the former is for port 80 unauthenitcated traffic and the latter
is for https. These two gateway nodes are available for users to do
general development on. Service level agreements are possible along
with dedicated resources for projects that wish to build robustly
monitored web services.


NERSC encourages its users to create their own science gateways by
using the resources described on this page. The center engages with
science teams interested in using web services, assists with
deployment, accepts feedback, and tries to recycle successful
approaches into methods that other science teams can benefit
from. Below you will find links to current projects and details about
the building blocks available to NERSC users. If you would like to
participate, or if you have questions, please contact
consult@nersc.gov.

NERSC Resource | Path On NERSC Resource | URL on the Web
--- | --- | ---
NGF (project filesystem) | /global/project/projectdirs/myproj/www | https://portal.nersc.gov/project/myproj/
NGF (dna filesystem) | /global/dna/projectdirs/myproj/mysubproj/www | https://portal.nersc.gov/dna/myproj/mysubproj/
HPSS archive (home) | /home/m/myuser/www | https://portal.nersc.gov/archive/home/m/myuser/www/
HPSS archive (project) | /home/projects/myproj/www | https://portal.nersc.gov/archive/projects/myproj/www/

### Web Methods for Computing

Science gateways can use a REST-based web API
([NEWT](https://newt.nersc.gov/)) to access the NERSC center,
including authentication, file management, job submission and
accounting interfaces. These interfaces allow you to run large or
small jobs on NERSC machines through the web. The NEWT demos show how
to submit a parallel batch job through a simple HTML form. Other
programming language and web-toolkit-level building blocks include

* Full-featured back-end programming environments in the language of
  your choice (PHP or Python recommended).
* Support for LDAP and Shibboleth authentication.
* Conduits to PostGRESQL/MySQL/NoSQL Databases.
* Modern Web 2.0 interfaces with AJAX front-ends such as Google maps
  and visualization kits.
* OpenDAP access to large data sets (netCDF and HDF5)
* Access to NERSC filesystems and HPSS through the NEWT API, grid
  tools, or other custom interfaces

### Database Methods

Science gateways can also access data from NERSC's science database
nodes. These are specially configured nodes which support MySQL,
Postgres, and MongoDB for high-performance access. More detail on the
science gateway database services is provided on
the
[Databases page](https://www.nersc.gov/users/data-analytics/data-management/databases/#Databases). Some
examples of database methods used by gateways are

* Access file catalogs and other persistently stored collections from
  your batch jobs
* Connect a web-based gateway to datasets stored in a database (read
  and read-write)
* Store, search, and analyze data objects (e.g., job output) through
  map/reduce-like MongoDB methods
* Expose public read-only data collections through database protocols

For more information on databases for user science data, please submit
a question or request via
the
[science database request form](https://www.nersc.gov/users/data-analytics/data-management/databases/science-database-request-form/).

## Science Gateways in Production

Science gateways that have moved from development to providing
services to broader communities are listed on
the [Science Gateways index page](https://portal.nersc.gov/).

Nagios monitoring and service level checks of gateway functions are
available.

## Getting Started

A
[project directory](https://www.nersc.gov/users/storage-and-file-systems/file-systems/project-file-system/) is
a good place to host a science gateway. Both NGF and HPSS allow users
to create a special web directory within a project directory. You can
publish data through a publicly accessible URL by simply making an
appropriate subdirectory called "www". The procedure differs slightly
depending on which file system you choose, as detailed below. You can
also use the NEWT API to make web applications that use NERSC
resources.

#### How to publish your data on NGF to the web:

    ssh portal-auth.nersc.gov

In the above example, you can replace portal-auth with any other NERSC
compute platform that has access to /project. Create a www directory
in your project directory:

    mkdir /global/project/projectdirs/yourproject/www

Make sure your project directory and the www directory are world
executable and that the www directory is also world readable. If not,
the owner of each of them will need to change its permissions:

    chmod 751 /global/project/projectdirs/yourproject/
    chmod 755 /global/project/projectdirs/yourproject/www

Copy your data to this www directory. Any public data will need to be
world readable. Add PHP and HTML files to this directory to build
custom gateway interfaces to the data. Any data under
`/global/project/projectdirs/yourproject/www` will be publicly
accessible through `https://portal.nersc.gov/project/yourproject/`.

#### How to publish data in HPSS to the web:

You can also publish data in the archive HPSS system directly to a
public URL on the web. Note that this is not intended to be a
high-performance interface; it is just a quick way to make data
publicly available.

Generally we recommend that users share data from the "project" file
system when creating a science gateway. Sharing data from the HPSS
tape archives via a science gateway should only be reserved for
infrequent accesses from a data pool that is too large to be
practically kept on the "project" filesystem. If you need to serve
very large files very frequently via a science gateway, please
contact
[NERSC consulting](https://www.nersc.gov/users/getting-help/online-help-desk/) for
assistance.

Retrieving data from HPSS via a science gateway can be very slow. If
files have not been accessed in some time they will have to be
retrieved from tape. If you are accessing multiple files, multiple
tapes may need to be read and special care will need to be taken to
retrieve the data files in the most optimal way. Finally, the number
of concurrent connections per IP address is limited to two. All of
these factors can combine for long delays in file retrieval from HPSS
via a NERSC web portal.

Login to archive via hsi:

    hsi -h archive.nersc.gov

Create a www directory

    mkdir /home/projects/DIRNAME/www

Make sure the parent directory and the www directory are world
executable and that the www directory is also world readable. If not,
the owner of each of them will need to change its permissions:

    chmod 751 /home/projects/DIRNAME
    chmod 755 /home/projects/DIRNAME/www

The data in the www directory will now be available at a URL of the
form
https://portal.nersc.gov/archive/home/projects/DIRNAME/www/{FILE|DIR}
where DIRNAME is the project directory and FILE|DIR is the name of a
file.

Files will be downloaded directly, while directories will give you a
listing. Note that all files and directories in the path must be world
readable.

Here is an example:
https://portal.nersc.gov/archive/home/projects/incite11/www/1935

!!! note
    The time to download files from tape may take some time to
    start as the tape robot finds and mounts the correct tape.

#### How to get started with NEWT

To build more sophisticated web apps, we recommend using
the [NEWT API](https://newt.nersc.gov/), which allows you to build
rich, interactive JavaScript applications that can communicate
directly with NERSC HPC resources via a RESTful Web API. This includes
access to authentication, jobs, files, interactive commands, system
information, NIM accounting information and object storage.

To get started, insert the following in your HTML files to give you
access to all NERSC compute and data resources through NEWT:

    <script src="[https://newt.nersc.gov/js/jquery-1.7.2.js](https://newt.nersc.gov/js/jquery-1.7.2.js)" />
    <script src="[https://newt.nersc.gov/js/newt.js](https://newt.nersc.gov/js/newt.js)" />

Follow the "Hello World" example
at [https://newt.nersc.gov/](https://newt.nersc.gov/), or work through
some of the fuller examples
at
[https://newt.nersc.gov/examples/](https://newt.nersc.gov/examples/)
to get a feel for how NEWT works. The complete NEWT API docs can be
found at [https://newt.nersc.gov/api](https://newt.nersc.gov/api).

## Moving Beyond Simple Gateway Functions

If you are building a web gateway to your science at NERSC, please
contact us at consult@nersc.gov. We are interested in engagaing
directly with science teams so that you can build a gateway that meets
your specific needs.
