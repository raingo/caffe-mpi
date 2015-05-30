#ifndef HEADER_FLAGS
#define HEADER_FLAGS

DEFINE_int32(gpu, -1,
        "Run in GPU mode on given device ID.");
DEFINE_int32(snapshot_intv, 50,
        "Number of snapshots to evaluate");
DEFINE_int32(eval_iter, 1000,
        "number of iterations to cover all samples, cifar=500, lenet=1000");
DEFINE_double(lr, 0.01,
        "Learning Rate");
DEFINE_string(model, "example/lenet.prototxt",
        "The network definition protocol buffer text file.");
DEFINE_int32(iterations, 100,
        "The number of iterations to run.");


#endif
