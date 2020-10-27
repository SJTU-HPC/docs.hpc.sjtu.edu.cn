clean:
	rm -rf docs/_build

%:
	make -C docs $@ 
