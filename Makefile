# install  - installation in system directories
# local    - installation for local user only
# examples - run examples and tutorials from examples/
# test     - run test suite
#
# Helper targets for developers
# 	_tags - building tags file for PyDSTool
# 	dev   - installation using 'python setup.py develop'
# 	undev - uninstall after 'make dev'

.PHONY: install local
	
install:
	@python setup.py install

local:
	@python setup.py install --user


.PHONY: examples test

examples:
	@cd examples && python run_all_tests.py

test: pyclean clean
	@python setup.py test


.PHONY: _tags dev undev

_tags:
	@find PyDSTool -name "*.py" | ctags -L -


dev:
	@python setup.py develop

undev:
	@python setup.py develop --uninstall

.PHONY: pyclean clean distclean 

pyclean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete

clean:
	-@find . -maxdepth 2 -type d \( -name "auto_temp" -o -name "radau5_temp" -o -name "dopri853_temp" \) -exec rm -rf {} \;
	-@find . -type f \( -name "dop853_*_vf.py" -o -name "radau5_*_vf.py" -o -name "auto_*_vf.py" \) -delete
	-@find . -type f -name "*.so" -delete
	-@find . -type f -name "*module.c" -delete
	-@find . \( -name "temp*.pkl" -o -name "fort.*" -o -name "tvals.dat" -o -name "varvals.dat" -o -name "vanderPol.dat" \) -delete

distclean: pyclean clean
	@python setup.py clean --all
	@rm -f PyDSTool/__config__.py

.PHONY: unixify
# Change newlines to unix format
# Endings: .txt, .py, .c, .f, .cc, .h, .hh, .html .csh .sh .c.dev .dat .py.lib .py.works .out .i 
# Plain files: README, Makefile, makefile
unixify:
	chmod u+x convert2unix.sh ; \
	find . \( -name '*.txt' -o -name '*.py' -o -name '*.c' -o -name '*.f' -o -name '*.cc' -o -name '*.h' -o -name '*.hh' -o -name '*.html' -o -name '*.csh' -o -name '*.sh' -o -name '*.c.dev' -o -name '*.dat' -o -name '*.py.lib' -o -name '*.py.works' -o -name '*.out' -o -name '*.i' -o -name 'Makefile' -o -name 'makefile' -o -name 'README' \) -exec ./convert2unix.sh '{}' \; 
