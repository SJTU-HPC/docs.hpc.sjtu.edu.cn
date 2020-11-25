# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going -n
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Download code samples from https://github.com/SJTU-HPC/reframe-tests.git
download:
	wget --no-check-certificate --content-disposition https://codeload.github.com/SJTU-HPC/reframe-tests/zip/master -O _static/reframe-tests-master.zip
	unzip -o _static/reframe-tests-master.zip -d _static/
	rsync -avr --progress --delete _static/reframe-tests-master/sjtuhpc-checks _static/

.PHONY: help Makefile download

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile download
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
