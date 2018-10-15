# git

_Note: This functionality is not formally supported by NERSC. These features
should be considered self-serve. We cannot offer any guarantees in terms of
functionality and availability moving forward._

## Prerequisites

In order to create CVS, Subversion, or git repositories on the NERSC Global
File System (NGF) a user must first have a project directory. After the project
directory is set up, access to the directory can be controlled in the NERSC
Information Management system (NIM) by the project's pricipal investigator
(PI). Users who create repositories must have write access to the project
directory. Once the user has a project directory, then the repository can be
created under that project directory. For example, a valid repository path
would be something like
```
/project/projectdirs/MyProjectDirectory/MygitRepo
```
The following instructions assume you have setup a project directory on NGF and
have write access. References to the project path are identified with the
`<project>` tag and the repository type is identified with the
`<git_repository_name>` tag below.

## Creating a git Repository

git uses a peer-to-peer model, so we need to distinguish between the repository
you are about to create and any other "clones" of it you have, perhaps on your
laptop. Call the repository you are about to create the SG-repository, for
"Science Gateway repository." Other repositories will get some other prefix, as
in L-repository for "laptop repository".

Anticipating our use of GitWeb below we'll put the repository in the
web-visible region of your project. The web visible region of your project is
```
/project/projectdirs/<project>/www
```
which must have permissions readable and executable by all. Additionally, your
project directory itself must be executabe by all. These instruction also
assume you may want multiple repositories so they will all be collected under
the `git` subdirectory.
```
cd /project/projectdirs/<project>/www
mkdir git
```
Any directories and files created in the repository directory should inherit
the group membership of the directory.
```
chmod -R g+srwx git
```
Similarly, the server process needs to be able to see anything you plan to put
in a publicly available repository.
```
chmod -R o+rx git
cd git
mkdir <git_repository_name>.git
```
Always use the `.git` extension so the gitweb (see below) interface works
correctly.
```
chmod -R ug+rwx <git_repository_name>.git
chmod -R o+rx <git_repository_name>.git
```
See above. You can defer this until you've set everything up and do a final
pair of `chmod -R` commands as indicated below.
```
cd <git_repository_name>.git
git --bare init
Initialized empty Git repository in /project/projectdirs/<project>/www/git/<git_repository_name>.git/
```
This step initializes an empty "bare" SG-repository. The "bare" is important
because you do not want a working copy in this SG-repository, ever. (This is a
widely commented upon and controversial subject, and I won't go into it here.
Ask a Git expert.)
```
git update-server-info --force
```
The above generates some auxiliary files to assist in decoding some references.
I have always found it necessary, though the [git
documentation](https://git-scm.com/documentation)  suggests it might not always
be required.

Since the Git SG-repository is updated with each `push` there can be new files
and even new directories created. They all need the same `chmod` and
`update-server-info` treatment, potentially after each `push`. The
`post-receive` hook is the one invoked after the SG-repository receives a
`push`. You can create a new `post-receive` hook or just edit the sample
provided:
```
cp hooks/post-receive.sample hooks/post-receive
echo '/usr/common/usg/git/1.6.5.6/bin/git update-server-info --force' >> hooks/post-receive
echo 'chmod -R ug+rwx /project/projectdirs/<project>/www/git/<git_repository_name>.git' >> hooks/post-receive
echo 'chmod -R o+rx /project/projectdirs/<project>/www/git/<git_repository_name>.git' >> hooks/post-receive
```
Now that the repository has been initialized the `chmod` can be applied recursively.
```
chmod -R ug+rwx /project/projectdirs/<project>/www/git
chmod -R o+rx /project/projectdirs/<project>/www/git
```
You now have an empty Git SG-repository that is visible on the web. Remember,
never access this SG-repository from within your `/project` space. A working
copy of the SG-repository in this location can subsequently corrupt anything
that is sent to the SG-repository via a remote `push` operation. You will only
put new content into this SG-repository via remote `push`.

## Read-only access to a Git Repository

Public (unauthenticated) access to your SG-repository is possible on a
read-only basis. The following command will allow anyone to get a `clone` of
your SG-repository:
```
HTTP_REPO=portal.nersc.gov/project/<project>/git/<git_repository_name>.git
git clone https://${HTTP_REPO} <git_repository_name>
Initialized empty Git repository in .../<git_repository_name>/.git/
```
You now have an L-repository that is a local, read-only clone of the
SG-repository. That repository can be used in every way the same as any other
with the one exception that it will not `push` back to the SG-repository. That
is because the `portal.nersc.gov` URL does not (cannot) authenticate, and the
`push` will simply fail.

The command will always report `Initialized empty Git repository in
.../<git_repository_name>`, since it creates the empty one before cloning the
remote contents. If you get error or warning messages like:
```
warning: You appear to have cloned an empty repository.
```
```
bash: git-upload-pack: command not found fatal: The remote end hung up unexpectedly
```
```
Total 13 (delta 0), reused 0 (delta 0) chmod: cannot access `<some path>': No
such file or directory
```
See the troubleshooting section.

## Authenticated access to a Git Repository

You put content into a Git SG-repository remotely via the git push command.
That operation will require authentication via NERSC's LDAP service.
Authenticated access is via the host `portal-auth.nersc.gov`. Your
SG-repository is in the same relative location on `portal-auth` as on `portal`.
```
https://portal-auth.nersc.gov/project/<project>/git/<git_repository_name>.git
```
The following puts content into a previously empty SG-repository. Updating the
SG-repository with subsequent new content proceeds in the same way. In order to
get some initial content into your SG-repository do the following (on a remote
system: eg. your laptop):
```
REPO_PATH=/project/projectdirs/<project>/www/git/<git_repository_name>.git
```
See the troubleshooting section if you get the error: module: command not found.
```
git clone ssh://portal-auth.nersc.gov${REPO_PATH} <git_repository_name>
Initialized empty Git repository in .../<get_repository_name>/.git/

...

Password:

Warning: No xauth data; using fake authentication data for X11 forwarding.

warning: You appear to have cloned an empty repository.
```
```
cd <git_repository_name>
```
Put some content in place, say `README.txt`.
```
git add README.txt
```
Repeat for whatever other content you have in mind.
```
git commit -m "Initialize local repository content"
[master (root-commit) 5ee963c] Initialize repository content

1 files changed, 480 insertions(+), 0 deletions(-)

create mode 100644 README.txt
```
Content is now in your local L-repository.

This synchronizes your L-repository with the SG-repository you created on the
NERSC Science Gateway:
```
git push origin master
...

Password:

Counting objects: 3, done.

Delta compression using up to 2 threads.

Compressing objects: 100% (2/2), done.

Writing objects: 100% (3/3), 4.32 KiB, done.

Total 3 (delta 0), reused 0 (delta 0)

To ssh://portal-auth.nersc.gov/project/projectdirs/pma/www/git/git_test.git

* [new branch] master -> master
```
Getting content from the SG-repository proceeds in the same way when you don't
have a cloned L-repository already in place, but the SG-repository does have
content. Refer to the Git documentation for further details on interacting with
Git.

##  Private Git repositories

 If you do not want your SG-repository to be publicly visible, even read-only,
 then don't do the `chmod o+rx` on that SG-repository and its contents. Also
 leave that line out of the `post-receive` script. Otherwise proceed as above.
 So long as the `other` UNIX permission bits are not set the Apache server will
 not be able to see the SG-repository. Therefore that SG-repository will not be
 visible via `portal.nersc.gov`. Keep this in mind when/if you run the
 recursive `chmod -R` to set permissions. You can have some private and some
 public SG-repositories in your project's `www/git` directory if you manage the
 permissions accordingly.

## Configuring GitWeb and public access to your Git repository

GitWeb is a part of the standard Git distribution. The way GitWeb is organized,
it needs to be told about the location of your SG-repository (or repositories).
Since your repository is in a distinct location from other projects on the
Science Gateway, you have to have your own instance of GitWeb. Take the
`gitweb` directory from the Git distribution and put a copy of it in your
project web directory.
```
cp -r /usr/common/usg/git/gitweb /project/projectdirs/<project>/www/
cd /project/projectdirs/<project>/www/gitweb
```
Find the line in `gitweb.cgi` that sets the `$GIT` variable and point it at the
Git executable (where it is on `portal.nersc.gov`):
```
our $GIT = "/usr/common/usg/git/1.6.5.6/bin/git";
```
Find the line in `gitweb.cgi` that sets the `$projectroot` variable and point
it at your repositories directory (the directory above
`<git_repository_name>.git`):
```
our $projectroot = "/project/projectdirs/<project>/www/git";
```
There are numerous other variables in `gitweb.cgi` that you may want to
customize. For example the `site header`:
```
our $site_header = "Project Name";
```
To add a description to your git repository, so that the text shows up in
gitweb. Add the description text to a new file called description:
```
/project/projectdirs/<project>/www/git/<repository>.git/description
chmod a+r /project/projectdirs/<project>/www/git/<repository>.git/description
```
Once GitWeb has been localized you will want to make it visible on the web as
above:
```
cd /project/projectdirs/<project>/www/
chmod -R ug+rwx gitweb
chmod -R o+rx gitweb
```
See above for a note about public versus private repositories. If you do not
have any publically visible SG-repositories, then you do not need to do the
second chmod line above. If you have some public and some private, then go
ahead and do the `chmod -R g+rx`, since anonymous visitors will only see the
public ones anyway.

You can then access your repositories anonymously via the web at
```
https://portal.nersc.gov/project/<project>/gitweb/gitweb.cgi
```

## Troubleshooting

### module: command not found
If you see the above message on your local system (your laptop) when you try to
clone the git SG-repository it means that your (non-terminal) login environment
on `portal.nersc.gov` does not have the `module` command.

My `bash` account on the global home file system has a system generated
`${HOME}/".bashrc` that includes `${HOME}/.modules`, which has the following
content:
```
if [ -z $MODULEPATH ] ; then
export MODULEPATH=/usr/share/Modules/modulefiles:/global/common/datatran/dsg/Modules/modulefiles:/global/common/datatran/usg/Modules/modulefiles:/global/common/datatran/mss/Modules/modulefiles:/usr/common/Modules/modulefiles:/usr/common/usg/Modules/modulefiles:/etc/modulefiles:
module () { eval `/usr/bin/modulecmd bash $*` }
fi
module load null git/1.6.5.6
```
Put something similar in your environment and remote git commands should start
working.


### warning: You appear to have cloned an empty repository
If you see the above message on your local system (your laptop) when you try to
anonymously clone the git SG-repository (via HTTP) it probably means that you
have a permissions issue in the SG-repository. Visit that repository and issue
both the `update-server-info` command and the `chmod` commands from the
post-receive script. Verify that there aren't any typos. Then try the anonymous
clone command again. If it works this time then it was a permissions issue.
Review the set up details and verify it works going forward.
