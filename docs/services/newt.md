NEWT is bringing High Performance Computing (HPC) to the web through easy to write web applications. We think that writing applications for the web should be a simple task. We want to make HPC resources accessible and useful to scientists who are more comfortable with the web than they are with compilers, batch queues, or punched cards. Thanks to Web 2.0 APIs, we believe HPC can speak the same language as the web.

NEWT is a web service that allows you to access computing resources at NERSC through a simple RESTful API. The NEWT API and web service will let you interact with NERSC through simple HTTP urls and commands. NEWT responds to client requests using JSON. This makes it very easy to build powerful web applications using nothing but HTML and Javascript.

NEWT currently exposes the following services over the web (some services require authentication):
* Authentication
* System Status
* File Upload/Download
* Directory Listings
* Running Commands
* Batch Queue Jobs
* Accounting Information
* Persistent object storage

NEWT consists of two parts:

1. A web service that resides at https://newt.nersc.gov/newt.
2. A javascript library called newt.js which provides you with helper functions to access the NEWT service through simple AJAX calls.

You can learn more about the NEWT API and get a copy of the newt.js library at [https://newt.nersc.gov/](https://newt.nersc.gov/)