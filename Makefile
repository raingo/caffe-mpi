INCLUDE_DIRS = caffe/include caffe/build/src /u/yli/.local/include /usr/local/include /home/qyou/bin/include /u/qyou/lib/imdb/include/

CFLAGS = -std=c++11
CFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))

COMMON_LD=-lprotobuf -lgflags -lglog -lcaffe

LDFLAGS += ${COMMON_LD} -Lcaffe/build/lib -Wl,-rpath,caffe/build/lib/
EVAL_LDFLAGS += ${COMMON_LD} -Lcaffe/build-cu/lib -Wl,-rpath,caffe/build-cu/lib/ -lcudart -lcublas -lcurand -L/u/qyou/lib/cuda-6.5/lib64/

all: sgd sgd-mpi evaluator count

.evaluator.o: evaluator.cpp common.hpp evaluator.hpp flags.hpp mpi.hpp snapshot.pb.h caffe/build-cu/lib/libcaffe.so
	g++ evaluator.cpp ${CFLAGS} -c -o .evaluator.o

evaluator:.evaluator.o .snapshot.pb.o caffe/build-cu/lib/libcaffe.so
	g++ .evaluator.o .snapshot.pb.o -o evaluator ${EVAL_LDFLAGS}

.sgd.o: sgd.cpp common.hpp evaluator.hpp flags.hpp snapshot.pb.h caffe/build/lib/libcaffe.so
	g++ sgd.cpp ${CFLAGS} -c -o .sgd.o

sgd:.sgd.o .snapshot.pb.o caffe/build/lib/libcaffe.so
	g++ .sgd.o .snapshot.pb.o -o sgd ${LDFLAGS}

.count.o: count.cpp common.hpp evaluator.hpp flags.hpp snapshot.pb.h caffe/build/lib/libcaffe.so
	g++ count.cpp ${CFLAGS} -c -o .count.o

count:.count.o .snapshot.pb.o caffe/build/lib/libcaffe.so
	g++ .count.o .snapshot.pb.o -o count ${LDFLAGS}


# build caffe, both cpu version and gpu version
# cpu version can run on cycle machines, together with release.sh

caffe/build-cu/lib/libcaffe.so:
	$(MAKE) -C caffe dist DISTRIBUTE_DIR=build-cu

caffe/build/lib/libcaffe.so: caffe/build-cu/lib/libcaffe.so
	$(MAKE) -C caffe clean
	$(MAKE) -C caffe CPU_ONLY=1

.sgd-mpi.o: sgd-mpi.cpp common.hpp evaluator.hpp flags.hpp mpi.hpp snapshot.pb.h caffe/build/lib/libcaffe.so
	mpicxx sgd-mpi.cpp ${CFLAGS} -c -o .sgd-mpi.o

sgd-mpi:.sgd-mpi.o .snapshot.pb.o caffe/build/lib/libcaffe.so
	mpicxx .sgd-mpi.o .snapshot.pb.o -o sgd-mpi ${LDFLAGS}

snapshot.pb.cc snapshot.pb.h: snapshot.proto
	protoc snapshot.proto --cpp_out=.

.snapshot.pb.o: snapshot.pb.cc snapshot.pb.h
	g++ snapshot.pb.cc -c ${CFLAGS} -o .snapshot.pb.o

dist: sgd sgd-mpi
	./release.sh

.PHONY: dist

clean:
	rm -Rf .*.o sgd sgd-mpi evaluator
	rm caffe/build-cu -Rf
	$(MAKE) -C caffe clean
