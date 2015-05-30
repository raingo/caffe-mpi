INCLUDE_DIRS = caffe/include caffe/build/src /u/yli/.local/include /usr/local/include /home/qyou/bin/include /u/qyou/lib/imdb/include/

CFLAGS = -std=c++11 -ggdb
CFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))

LDFLAGS=-lgflags -lglog -lcaffe -Lcaffe/build/lib -Wl,-rpath,caffe/build/lib/ -ggdb

all: sgd sgd-mpi

.sgd.o: sgd.cpp common.hpp evaluator.hpp flags.hpp
	g++ sgd.cpp ${CFLAGS} -c -o .sgd.o

sgd:.sgd.o ./caffe/build/lib/libcaffe.so
	g++ .sgd.o -o sgd ${LDFLAGS}

.sgd-mpi.o: sgd-mpi.cpp common.hpp evaluator.hpp flags.hpp mpi.hpp
	mpicxx sgd-mpi.cpp ${CFLAGS} -c -o .sgd-mpi.o

sgd-mpi:.sgd-mpi.o
	mpicxx .sgd-mpi.o -o sgd-mpi ${LDFLAGS}
