
#include "common.hpp"
#include "flags.hpp"
#include "evaluator.hpp"
#include "mpi.hpp"

void ApplyUpdate(shared_ptr<Net<Dtype> > net, Dtype lr, int node_id)
{
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net -> params();

    for (int i = 0; i < net_params.size(); i++) {
        auto param = net_params[i];
        recv_diff(param.get(), node_id);
        caffe_scal(param -> count(), lr, param -> mutable_cpu_diff());
        param -> Update();
        send_data(param.get(), node_id);
    }
}

void root_node(shared_ptr<Net<Dtype> > net, int iters, Dtype lr)
{
    const std::vector<Blob<Dtype>*>& result = net -> output_blobs();

    boost::posix_time::ptime timer = boost::posix_time::microsec_clock::local_time();

    MPI_Status status;
    std::vector<Blob<Dtype>*> bottom_vec;
    float loss;

    init_buffer(iters, FLAGS_snapshot_intv, net);

    for (int i = 0; i < iters; i++) {

        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        ApplyUpdate(net, lr, status.MPI_SOURCE);
        std::cout << i << std::endl;

        if (i % FLAGS_snapshot_intv == 0)
            snapshot(net, i, (boost::posix_time::microsec_clock::local_time() - timer).total_milliseconds());
    }

    evaluate(net, FLAGS_eval_iter);
}

void worker_node(shared_ptr<Net<Dtype> > net, int iters)
{
    std::vector<Blob<Dtype>*> bottom_vec;
    auto net_params = net -> params();
    const std::vector<Blob<Dtype>*>& result = net -> output_blobs();

    for (int i = 0; i < iters; i++) {
        Dtype loss = net -> ForwardBackward(bottom_vec);

        CHECK(!isnan(loss));

        for (int j = 0; j < net_params.size(); j++) {
            auto param = net_params[j];
            send_diff(param.get(), 0);
        }

        for (int j = 0; j < net_params.size(); j++) {
            auto param = net_params[j];
            recv_data(param.get(), 0);
        }
    }
}

int main(int argc, char** argv) {
    init_mpi(&argc, &argv);

    // disable log from caffe (disable glog entirely)
    FLAGS_minloglevel = 3;

    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);

    int iters = FLAGS_iterations;
    string netdefs = FLAGS_model;
    Dtype lr = Dtype(FLAGS_lr);

    // at least iters in total
    // actuall number of workers is np - 1
    iters = (iters + np - 2) / (np - 1);

    if (pid == 0)
        iters *= (np - 1);

    auto net = init_net(netdefs);
    std::vector<Blob<Dtype>*> bottom_vec;

    // sync parameters
    broadcast_params(net);

    if (pid == 0)
        root_node(net, iters, lr);
    else
        worker_node(net, iters);

    return finish_mpi();
}
