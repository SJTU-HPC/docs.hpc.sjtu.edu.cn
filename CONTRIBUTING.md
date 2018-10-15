# Contribution guide

This guide outlines how to contribute to this project and standards
which should be observed when adding to this repositiory.

## About

This repository contains NERSC technical documentation written in 
Markdown which is converted to html/css/js with the 
[mkdocs](http://www.mkdocs.org) static site generator. The 
[theme](https://gitlab.com/NERSC/mkdocs-material) is a fork of 
[mkdocs-material](https://github.com/squidfunk/mkdocs-material) with 
NERSC customizations such as the colors.

## Rules

1.  Follow this [Markdown styleguide](https://github.com/google/styleguide/blob/3591b2e540cbcb07423e02d20eee482165776603/docguide/style.md).
1.  Do not commit large files 
	(e.g. very high-res images, binary data, executables)
	* [Image optimization](https://developers.google.com/web/fundamentals/performance/optimizing-content-efficiency/image-optimization)
1.  No commits directly to the master branch

## Setup 

### Prerequisites

1. Anaconda or Python virtual env
2. git
3. gitlab account

### Clone the repo and install dependencies

```shell
git clone git@gitlab.com:NERSC/nersc.gitlab.io.git documentation
cd documentation
conda create -n docs pip
source activate docs
pip install -r requirements.txt
```

## How to

### Edit with live preview

Open a terminal session with the appropriate conda environment 
activated, navigate to the root directory of the repository (where 
`mkdocs.yml` is located) and run the command `mkdocs serve`. This will
start a live-reloading local web server. You can then open 
[http://127.0.0.1:8000](http://127.0.0.1:8000) in a web browser to 
see your local copy of the site.

In another terminal (or with a GUI editor) edit existing files, add 
new pages to `mkdocs.yml`, etc. As you save your changes the local 
web serve will automatically rerender and reload the site.

### Output a static site

To build a self-contained directory containing the full html/css/js 
of the site:

```
mkdocs build
```

### Contribute to the repo

#### Option 1

Work with a branch of the main repo.

1.  Make a new branch and call it something descriptive.

    ```shell
    git checkout -b username/what_you_are_doing
    ```

2.  Create/edit your content
3.  Commit your changes

    ```
    git commit -m 'describe what I did'
    ```

4.  Push your changes to gitlab

    ```shell
    git push
    ```

    Or if the branch doesn't exist on the gitlab repository yet

    ```shell
    git push --set-upstream origin username/my-new-feature
    ```

5.  Check if the continuous integration of your changes was successful
6.  Submit a merge request to the master branch with your changes

#### Option 2

Make a fork of the repository and do all of your work on the fork. 
Submit a merge request through gitlab when you have made your changes.

#### Option 3

For some changes you do not need the full environment. It is possible
to edit Markdown files directly on gitlab. This work should be in a 
private fork or branch and submitted as a merge request. A new branch
can be created by clicking the "+" button next to the repository name.

### Add a new page

For a newly added page to appear in the navigation edit the top-level
`mkdocs.yml` file.

### Review a Merge Request from a private fork

1.  Modify `.git/config` so merge requests are visible

    ```text
    ...
    [remote "origin"]
	        url = git@gitlab.com:NERSC/documentation.git
	        fetch = +refs/heads/*:refs/remotes/origin/*
	        fetch = +refs/merge-requests/*/head:refs/remotes/origin/pr/*
	...
	```
	
2.  Check for any Merge Requests

    ```shell
    git fetch
    ```
    
3.  Checkout a specific Merge Request for review (merge request `N` 
	in this example)

    ```shell
    git checkout origin/pr/N
    ```

# Content standards

## Command prompts

1. when showing a command and sample result, include a prompt 
   indicating where the command is run, eg for a command valid on any
   NERSC system, use `nersc$`:

    ```console
    nersc$ sqs
    JOBID   ST  USER   NAME         NODES REQUESTED USED  SUBMIT               PARTITION SCHEDULED_START      REASON
    864933  PD  elvis  first-job.*  2     10:00     0:00  2018-01-06T14:14:23  regular   avail_in_~48.0_days  None
    ```

    But if the command is cori-specific, use `cori$`:
    ```console
    cori$ sbatch -Cknl ./first-job.sh
    Submitted batch job 864933
    ```

2. Where possible, replace the username with `elvis` 
(i.e. a clearly-arbitrary-but-fake user name)

3. If pasting a snippet of a long output, indicate cuts with `[snip]`:
    ```console
    nersc$ ls -l
    total 23
    drwxrwx--- 2 elvis elvis  512 Jan  5 13:56 accounts
    drwxrwx--- 3 elvis elvis  512 Jan  5 13:56 data
    drwxrwx--- 3 elvis elvis  512 Jan  9 15:35 demo
    drwxrwx--- 2 elvis elvis  512 Jan  5 13:56 img
    -rw-rw---- 1 elvis elvis  407 Jan  9 15:35 index.md
    [snip]
    ```
    
## Writing Style

When adding a page think about your audience. 

* Are they new or advanced expert users?
* What is the goal of this content?

* [Grammatical Person](https://en.wikiversity.org/wiki/Technical_writing_style#Grammatical_person)
* [Active Voice](https://en.wikiversity.org/wiki/Technical_writing_style#Use_active_voice)

## Shell code should be `bash` not `csh`

## Definitions

* I/O not IO
* Slurm allocation
* NERSC allocation

## Slurm options

* Show both long and short option when introducing an option in text
* Use the long version (where possible) in scripts

## Markdown lint

Install the markdown linter (requires node/npm) locally
```shell
npm install markdownlint-cli
```

Run the linter from the base directory of the repository

```shell
./node_modules/markdownlint-cli/markdownlint.js docs
```