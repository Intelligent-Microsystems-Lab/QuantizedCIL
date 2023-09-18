run:
	python setup.py install
	rm clean.done
clean.done:
	rm -r build
	rm -r dist
	rm -r *.egg-info
	touch clean.done
clean: clean.done
