# NERSC Technical Documentation

## About

This repository contains NERSC technical documentation written in Markdown which is converted to html/css/js with the [mkdocs](http://www.mkdocs.org) static site generator. The [theme](https://gitlab.com/NERSC/mkdocs-material) is a fork of [mkdocs-material](https://github.com/squidfunk/mkdocs-material) with NERSC customizations such as the colors.

## Rules

1.  Follow this [Markdown styleguide](https://github.com/google/styleguide/blob/3591b2e540cbcb07423e02d20eee482165776603/docguide/style.md).
1.  Do not commit large files (e.g. very high-res images, binary data, executables)
	* [Image optimization](https://developers.google.com/web/fundamentals/performance/optimizing-content-efficiency/image-optimization)
1.  Do not commit directly to master branch

## Setup 

### Prerequisites

1. Anaconda or Python virtual env
2. git
3. gitlab account

### Clone the repo and install dependencies

```shell
git clone git@gitlab.com:NERSC/documentation.git
cd documentation
conda create -n docs pip
source activate docs
pip install -r requirements.txt
```

## How to

### Edit with live preview

Open a terminal session with the appropriate conda environment activated, navigate to the root directory of the repository (where `mkdocs.yml` is located) and run the command `mkdocs serve`. This will start a live-reloading local web server. You can then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a web browser to see your local copy of the site.

In another terminal (or with a GUI editor) edit existing files, add new pages to `mkdocs.yml`, etc. As you save your changes the local web serve will automatically rerender and reload the site.

### Output a static site

To build a self-contained directory containing the full html/css/js of the site:

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

Make a fork of the repository and do all of your work on the fork. Submit a merge request through gitlab when you have made your changes.

#### Option 3

For some changes you do not need the full environment. It is possible to edit Markdown files directly on gitlab. This work should be in a private fork or branch and submitted as a merge request.
