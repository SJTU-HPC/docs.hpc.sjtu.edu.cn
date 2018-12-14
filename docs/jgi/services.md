# Databases and Web Services

## JGI Services at NERSC

Any long-lived process or process listening to a network socket to communicate
requires some special attention and care to ensure that the service is
accessible to the appropriate audience, is properly monitored, and maintained
correctly. The rules and guidlines presented here are intended to strike a
balance between convenience of running and developing services with thee need
for system maintenance, distribution of resources, appropriate security, and
monitoring of each service.

All services, even development services, should be registered with NERSC.
Please [file a help ticket](https://help.nersc.gov/) to begin the registration
process.

## Web (HTTP/HTTPS) Services

Any service transporting data via HTTP or other related protocols (HTTPS, AJP)
needs very careful handling because of the heightened risk of a security
incident. For this reason, we have a special focus on providing some
additional assistance with providing web services.  Each interactive
system will operate an system-level httpd binding all the common http and
https ports. This system-level httpd is used to provide proxy/reverse proxy
service to user-level web services.  It is not intended to directly serve
content. A web service is any service using HTTP or related protocol for
transport, or delivering any form of HTML, XHTML, JSON or other common formats
understood by generic client software.  Any webservice which is used to
directly launch processes on compute systems will require special scrutiny.

### Guidelines for Running Web Services

* All services should endeavor to support https (encrypted http) as the default
  method of communication. NERSC will provide appropriate certificates for
  nersc.gov addresses on the system-level proxying httpd.
* All services must log all connections and requests to the central logging
  facility using a common format; this is achieved simply by using the proxying
  httpd.
* All production services must be centrally monitored via a NERSC-provided
  services nagios instance.  The form of the monitoring is determined based on
  the needs and requirements of the service.
* Web services may only bind to ports on the loopback device (127.0.0.1), and
  only on ports assigned to the service by NERSC staff. The assigned ports will
  be unique across the entire system to enable multiple services to operate on
  the same host simultaneously.
* Each service will require a fully qualified hostname in the jgi.doe.gov domain
* Any service passing authentication credentials must use https to secure the
  communication
* Any service starting processes must require some kind of authentication or
  pre-shared key to mitigate risk of unauthorized access
* Only the NERSC-operated proxy httpd will provide access to services on ports
  80, 443, 8000, 8080, 8443

### Public Web Services

JGI users may host web services for public access. NERSC offers two web hosting
environments, [Spin](https://docs.nersc.gov/services/spin/) and virtual
machines. The requirements of your web service will determine which of these
environments is appropriate. Please [contact NERSC](https://help.nersc.gov/)
with inquiries regarding public web services.

### Getting Started

When you’re ready, [submit a request](https://help.nersc.gov) with the details
of your planned web service. A NERSC consultant or other staff member will
respond to let you know when the hosting environment is ready for you to
deploy the development or test version (including an assigned TCP port), or to
ask additional questions.

Production web services must be reviewed before they are launched. In
preparation for this, please give advance consideration to:

* look and feel: is the service and its behavior consistent with other related
  services?
* on-going maintenance: what individual or group will be responsible for the
  service now, and what individual or group will maintain the service in the future?
* application security: does the service follow the secure coding standards
  provided by
  [OWASP](https://www.owasp.org/index.php/Projects/OWASP_Secure_Coding_Practices_-_Quick_Reference_Guide)
  and [CERT](http://www.cert.org/secure-coding/research/secure-coding-standards.cfm),
  and are any software dependencies bundled with the application (e.g. Perl or
  Python modules or Ruby GEMs) regularly updated?

## Service Security

Maintaining service security is an ongoing responsibility that the service
operator will need to pay close attention to. NERSC Security will
periodically scan the system and may provide some guidance, however, the
service operator is ultimately responsible. The responsibilities include
configuring your service to minimize risk of exposure, discovering potential
security vulnerabilities, and ensuring the latest security patches are applied
to the software you operate.

### Security Guidelines

* Any service directly launching processes should require some kind of
  authentication or pre-shared key to mitigate risk of unauthorized access
* Communcations should be encrypted if possible
* Communications of authentication credentials or pre-shared keys should be
  encrypted
* Connections or at least communications should be restricted to computers in
  the NERSC (128.55.\*.\*) and possibly LBL (128.3.\*.\* or 131.243.\*.\*)
  address space
* Any discovered security issues should be addressed very quickly; it may be
  necessary to temporarily stop the service or disable associated accounts
  while addressing a security issue

### Backups

The service operator (you!) is responsible for ensuring appropriate backups of
the service are made.  This may be as simple as ensuring the primary data
store for the service is in an already backed up portion of the filesystem. If
you choose to localize the service to local disk, you will then be responsible
for both backup and restore within your space.
