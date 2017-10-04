# NERSC Technical Documentation

## About

This repository contains NERSC technical documentation written in Markdown which is converted to html/css/js with the [mkdocs](http://www.mkdocs.org) static site generator. This file mainly describes how to contribute.

## Rules

1.  Follow this [Markdown styleguide](https://github.com/google/styleguide/blob/3591b2e540cbcb07423e02d20eee482165776603/docguide/style.md).
1.  Do not commit large files (e.g. very high-res images, binary data, executables)
	* [Image optimization](https://developers.google.com/web/fundamentals/performance/optimizing-content-efficiency/image-optimization)
1.  Do not commit directly to master branch

## Setup 

Instructions to setup your local environment to allow for a live-reloading local server for development.

1.  Clone this repo
1.  Create a new conda env/ virtualenv
1.  `cd nersc-docs`
1.  `pip install -r requirements.txt`


### Run a local server for a live preview of changes

In the root directory of the repository run `mkdocs serve` and navigate your browser to the displayed address. Edits to source files will automatically trigger rebuilds of the site as needed.

### Output a static site

To build a self-contained directory containing the full html/css/js of the site:

`mkdocs build`

## How to make a new page

1.  Create a new branch `git checkout master && git branch my-new-page`
1.  Create/edit your content
1.  Edit the `mkdocs.yml` file to include your content in the site
1.  Check that your new content is built properly and without errors
    ```shell
	mkdocs build --clean --strict
	```
1.  Commit your changes
1.  Submit a pull request for review

