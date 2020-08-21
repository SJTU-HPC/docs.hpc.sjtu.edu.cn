.PHONY : update server

git:
	git pull
	git push

clean:
	rm -rf docs/_build

build:
	mkdocs build --clean
	make -C docs html SPHINXOPTS="-W --keep-going"
