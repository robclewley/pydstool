# Generated automatically from Makefile.in by configure.
#
#   Makefile		3/1994		Xianjun WANG
#
PGM    = lor
CC     = gcc
OPT    = -O -g
#AUTO_DIR = /home/mlamar/PyDSTool/PyCont/auto
CFLAGS =  $(OPT) -I$(AUTO_DIR)/src/include
SRC    = $(PGM).c
EXE    = $(PGM).exe
LIBS   = -L$(AUTO_DIR)/lib -lauto2000 -lm
RM     = rm -f
#
all: clean $(EXE) run
#
$(EXE): $(SRC:.c=.o)
	$(CC) $(CFLAGS) $(SRC:.c=.o) -o $@ $(LIBS)
#
run: first
#
first:	$(EXE)
	@echo " "
	@echo "Demo $(PGM) is started"
	@echo " "
	@echo "$(PGM) : first run : stationary solutions"
	@$(AUTO_COMMAND_PREFIX) $(EXE) $(AUTO_COMMAND_ARGS)
	@echo " Saved as *.$(PGM)"
#
clean:
	@echo "Cleaning $(PGM) ..."
	@cp $(SRC) $(EXE)
	@$(RM) *.exe *.o b.* s.* d.* fort.* *~ core
	@echo "Cleaning ... done"
 
