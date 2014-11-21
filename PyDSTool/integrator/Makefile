# Makefile to perform housekeeping operations for PyDSTool distributions
# - Change newlines in files to unix format
# - Remove .o, .so, ~ files, etc. and any errant .svn directories (for release)
#
# Erik Sherwood (sherwood@cam) 16OCT06

.PHONY: clean cleanup unixify

# Remove object files, bytecode
clean:
	rm -rf *.so *.o *.pyc *~

cleanup:
	rm -rf *.pyc *~ *.so dop853_* dopri853_temp radau5_temp radau5_* auto_temp auto_* fort.*
	rm -rf .svn
	rm -f .DS_Stor*


# Change newlines to unix format
# Endings: .txt, .py, .c, .f, .cc, .h, .hh, .html .csh .sh .c.dev .dat .py.lib .py.works .out .i 
# Plain files: README, Makefile, makefile

unixify:
	chmod u+x convert2unix.sh ; \
	find . \( -name '*.txt' -o -name '*.py' -o -name '*.c' -o -name '*.f' -o -name '*.cc' -o -name '*.h' -o -name '*.hh' -o -name '*.html' -o -name '*.csh' -o -name '*.sh' -o -name '*.c.dev' -o -name '*.dat' -o -name '*.py.lib' -o -name '*.py.works' -o -name '*.out' -o -name '*.i' -o -name 'Makefile' -o -name 'makefile' -o -name 'README' \) -exec ./convert2unix.sh '{}' \; 


NUMPY_INCLUDE=$(shell python -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE=$(shell python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")

CFLAGS += -std=c99

# not yet true, just to suppress warning
CFLAGS += -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

ifdef USE_CLANG
	CC=/usr/bin/clang
	CFLAGS += -Wno-return-type
else
	CC=cc
endif

COMMON_SRC = \
	integration.c \
	interface.c \
	eventFinding.c \
	memory.c

dopri:
	$(CC) $(CFLAGS) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -D__DOPRI__ -c dop853mod.c dop853.c $(COMMON_SRC)

radau:
	$(CC) $(CFLAGS) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -D__RADAU__ -c radau5mod.c $(COMMON_SRC)
