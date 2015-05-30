INCLUDE_DIRS = caffe/include caffe/build/src /u/yli/.local/include /usr/local/include /home/qyou/bin/include /u/qyou/lib/imdb/include/

CFLAGS = -std=c++11
CFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))

LDFLAGS=-lgflags -lglog -lcaffe -Lcaffe/build/lib -Wl,-rpath,caffe/build/lib/

all: sgd sgd-mpi

.sgd.o: sgd.cpp
	g++ sgd.cpp ${CFLAGS} -c -o .sgd.o

sgd:.sgd.o ./caffe/build/lib/libcaffe.so
	g++ .sgd.o -o sgd ${LDFLAGS}

.sgd-mpi.o: sgd-mpi.cpp
	mpicxx sgd-mpi.cpp ${CFLAGS} -c -o .sgd-mpi.o

sgd-mpi:.sgd-mpi.o
	mpicxx .sgd-mpi.o -o sgd-mpi ${LDFLAGS}
