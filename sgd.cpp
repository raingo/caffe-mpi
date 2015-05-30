
#include "common.hpp"
#include "flags.hpp"
#include "evaluator.hpp"

void ApplyUpdate(shared_ptr<Net<Dtype> > net, Dtype lr)
{
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net -> params();

    for (int i = 0; i < net_params.size(); i++) {
        auto param = net_params[i];
        //grad_mag += sumsq(param -> cpu_diff(), param -> count());
        caffe_scal(param -> count(), lr, param -> mutable_cpu_diff());
    }

    net -> Update();
}

int main(int argc, char** argv) {
    // Print output to stderr (while still logging).
    // FLAGS_alsologtostderr = 0;
    FLAGS_minloglevel = 3;

    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);

    int iters = FLAGS_iterations;
    string netdefs = FLAGS_model;
    Dtype lr = Dtype(FLAGS_lr);

    auto net = init_net(netdefs);
    std::vector<Blob<Dtype>*> bottom_vec;
    Dtype loss = net -> ForwardBackward(bottom_vec);
    init_buffer(iters, FLAGS_snapshot_intv, net);


    boost::posix_time::ptime timer = boost::posix_time::microsec_clock::local_time();

    for (int i = 0; i < iters; i++) {
        Dtype loss = net -> ForwardBackward(bottom_vec);
        ApplyUpdate(net, lr);

        std::cout << i << std::endl;

        if (i % FLAGS_snapshot_intv == 0)
            snapshot(net, i, (boost::posix_time::microsec_clock::local_time() - timer).total_milliseconds());
    }

    save_snapshot(FLAGS_snap_path);
    return 0;
}
