# Makefile to perform housekeeping operations for PyDSTool distributions
# - Change newlines in files to unix format
# - Remove .o, .so, ~ files, etc.
#
# Erik Sherwood (sherwood@cam) 16OCT06

.PHONY: clean cleanup unixify

# Remove object files, bytecode
clean:
	rm -rf *.so *.o *.pyc *~

cleanup:
	rm -rf *.pyc *~ *.so dop853_* dopri853_temp radau5_temp radau5_* auto_temp auto_* fort.*


# Change newlines to unix format
# Endings: .txt, .py, .c, .f, .cc, .h, .hh, .html .csh .sh .c.dev .dat .py.lib .py.works .out .i 
# Plain files: README, Makefile, makefile

unixify:
	chmod u+x convert2unix.sh ; \
	find . \( -name '*.txt' -o -name '*.py' -o -name '*.c' -o -name '*.f' -o -name '*.cc' -o -name '*.h' -o -name '*.hh' -o -name '*.html' -o -name '*.csh' -o -name '*.sh' -o -name '*.c.dev' -o -name '*.dat' -o -name '*.py.lib' -o -name '*.py.works' -o -name '*.out' -o -name '*.i' -o -name 'Makefile' -o -name 'makefile' -o -name 'README' \) -exec ./convert2unix.sh '{}' \; 

