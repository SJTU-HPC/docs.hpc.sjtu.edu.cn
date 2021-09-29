clean:
	rm -rf docs/_build
	rm -rf docs/reframe-tests-master*
	rm -rf docs/_static/sjtuhpc-checks

%:
	make -C docs $@
