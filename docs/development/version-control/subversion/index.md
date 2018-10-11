# Subversion

_Note: This functionality is not formally supported by NERSC. These features
should be considered self-serve. We cannot offer any guarantees in terms of
functionality and availability moving forward._

## Prerequisites

In order to create Subversion (SVN) repositories on the NERSC Global File
System (NGF) a user must first have a project directory. After the project
directory is setup then access to the directory can be controlled in the NERSC
Information Management system (NIM) by the project's pricipal investigator
(PI). Users who create repositories must have write access to the project
directory. Once the user has a project directory, then the SVN repository can
be created under that project directory. For example, a valid repository path
would be something like:
```
/project/projectdirs/MyProjectDirectory/MySVNRepo.
```
Subversion and CVS repositories can be made available via the web using a
software package called ViewVC. The recommended version of ViewVC can be found
in the section below (Viewing Repositories Over the Web: Using ViewVC).

The following instructions assume you have setup a project directory on NGF and
have write access. References to the project path are identified with the
`<project>` tag and the repository type identified with the
`<cvs_repository_name>`, `<svn_repository_name>`, or `<git_repository_name>`
tag below.

## Creating a Subversion Repository on NGF

Create the repository in your project directory or subdirectory. You may find
keeping your repos in a specific directory is a useful way to keep track of
them. Add the option `--pre-1.5-compatible` when creating your repo to ensure
cross compatibility between all of the NERSC hosts.
```
svnadmin create --pre-1.5-compatible <svn_repository_name>
```
Note that the `--pre-1.5-compatible` flag makes your repository compatible with
older svn clients but may prevent you from using newer svn merge features. Any
directories and files created in the repository directory should inherit the
group membership of the directory. Perhaps your project directory already has
this, in which case the command below is not necessary (though it is harmless,
if so).
```
chmod -R g+srwx <svn_repository_name>
```
Add your project directories and files to the repository. Directions for adding
directories or files to Subversion can be found below. Be sure to commit your
changes after adding directories and files to your repository.

## Adding files to SVN from NGF

Any files added to SVN from NGF should be done using `portal-auth.nersc.gov`.
After you login then you can follow the basic instructions below for adding
files to your SVN repo. If you have an existing project then the simplest way
to get started with SVN is to import your local project directory into the SVN
repo you previosly created. An example has been provided to you below.
```
svn import /project/projectdirs/<project>/<project_name> file:///project/projectdirs/<project>/<svn_repository_path>/<svn_repository_name>
```
Files and directories can be added to the SVN repo using the following set of
commands and options provided in the following link: [Adding files and
directories to
SVN](http://svnbook.red-bean.com/nightly/en/svn.ref.svn.c.add.html)

## Adding files to SVN remotely

Setting the `REPO_PATH` variable on your local machine is required in order for
authentication to work properly.

bash:
```
REPO_PATH=/project/projectdirs/<project>/<svn_repository_name>; export REPO_PATH
```
csh:
```
setenv REPO_PATH /project/projectdirs/<project>/<svn_repository_name>
```

Importing your initial version can be done using authenticated access to svn.
An example has been provided below.
```
svn import /<local_project_directory> svn+ssh://<nim_user_name>@portal-auth.nersc.gov/project/projectdirs/<svn_repository_path>/<svn_repository_name> -m "initial import"
```
You will need to checkout a new copy for your revisions to be saved in your SVN
repo. After you have a local copy of the repo then you can add/commit files as
usual.
```
svn co svn+ssh://<nim_user_name>@portal-auth.nersc.gov/project/projectdirs/<project>/<svn_repository_path>/<svn_repository_name>  <local_name>
cd /<local_project_directory>/<local_name>
svn add myfile
svn commit myfile -m "initial version"
```

## Receiving notifications when commits are made

If the "post-commit" file in your project's SVN repository "hooks" directory
has not already been updated then you will first have to copy the template to
the "hooks" directory.
```
cp /project/projectdirs/sgn/software/usg/subversion/scripts/post-commit /project/projectdirs/<project_name>/<svn_repository_path>/<svn_repository_name>/hooks/.
cd /project/projectdirs/<project_name>/<svn_repository_path>/<svn_repository_name>/hooks
```
Grant read and execute permissions for the script to run when commits are made to the SVN repository:
```
chmod o+rx post-commit
```
Edit the `post-commit` file in the `hooks` directory and change the sample list
of email addresses associated with the `SENDTO` variable to those who should
receive notifications. Make sure the email addresses are comma separated. None
of the other information should be changed. Save the file and test it out by
commiting changes to an existing file or adding a new file to the repo and
committing the changes.

## Read-only public access to SVN

Both read and execute permissions have to be granted to others in order for the
files to be read via http.
```
chmod -R o+rx /project/projectdirs/<project_name>/<svn_repository_path>/<svn_repository_name>
```
A symlink will have to be created by NERSC staff in order for your svn repo to
be accessible via http. You can request for a symlink to be created by sending
an email to `consult@nersc.gov`. Please include the full path to the SVN repo
in your request to expedite the process. A NERSC staff member will have to
create a symlink in `/var/www/svn` and will have to point to the location of
the svn repo (i.e.,
`/project/projectdirs/<project_name>/<svn_repository_name>`).
```
ln -s /project/projectdirs/<project>/<svn_location> /var/www/svn/<symlink_name>
```
Once the symlink is setup then checking out a project from the svn repo can be
completed by using the Subversion client:
```
svn co https://portal.nersc.gov/svn/<project>
```

## Authenticated access to SVN

Setting the `REPO_PATH` variable may be necessary if it's not already set.

bash:
```
REPO_PATH=/project/projectdirs/<project>/<svn_repository_name>; export REPO_PATH
```
csh:
```
setenv REPO_PATH /project/projectdirs/<project>/<svn_repository_name>
```

Accessing an SVN repo on NGF via SSH can be done using the svn client:
```
svn co  svn+ssh://<nim_user_name>@portal-auth.nersc.gov/project/projectdirs/<project>/<svn_repository_path>/<svn_repository_name>  <local_name>
```
