#!/bin/bash

# source /opt/intel/mkl/bin/mklvars.sh intel64
# source /opt/intel/compilers_and_libraries_2017.4.181/mac/bin/compilervars.sh intel64

TARGETS=learn_curve_krls

CROSS_TOOL=
CC_CPP=$(CROSS_TOOL)g++
CC_C=$(CROSS_TOOL)gcc

CFLAGS = -O3 -pg -g -m64
#O3 -Werror -fopenmp -m64


INCLUDEPATH = -I./ -I${MKLROOT}/include 

LIBPATH = -L./ -L/usr/local/lib -L${MKLROOT}/lib/intel64/
LIBS =-lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lmkl_intel_thread -lmkl_intel_lp64 -lmkl_core -lpthread -liomp5 -lm

all: clean $(TARGETS)

$(TARGETS):
	$(CC_CPP) $(CFLAGS) $(INCLUDEPATH) $@.cpp -o $@ $(LIBPATH) $(LIBS)	

clean:
	rm -f $(TARGETS)

train:
	$(CC_CPP) $(CFLAGS) $(INCLUDEPATH) tb_train_krls.cpp -o $@ $(LIBPATH) $(LIBS)	

learn_krls_A:
	$(CC_CPP) $(CFLAGS) $(INCLUDEPATH) learn_curve_krls.cpp -o learn_krls_A  $(LIBPATH) $(LIBS)   

learn_krls_B:
	$(CC_CPP) $(CFLAGS) $(INCLUDEPATH) learn_curve_krls.cpp -o learn_krls_B $(LIBPATH) $(LIBS)

mixed:
	$(CC_CPP) $(CFLAGS) $(INCLUDEPATH) mixed_learn_curve_krls.cpp -o $@ $(LIBPATH) $(LIBS)  

mg:
	$(CC_CPP) $(CFLAGS) $(INCLUDEPATH) mg_learn_curve_krls.cpp -o $@ $(LIBPATH) $(LIBS) 

init: 
	source ../intel/mkl/bin/mklvars.sh intel64
	source ../intel/bin/compilervars.sh intel64


