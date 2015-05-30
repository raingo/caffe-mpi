#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <mpi.h>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

using namespace caffe;

DEFINE_int32(gpu, -1,
        "Run in GPU mode on given device ID.");
DEFINE_double(lr, 0.01,
        "Learning Rate");
DEFINE_string(model, "",
        "The network definition protocol buffer text file.");
DEFINE_int32(iterations, 50,
        "The number of iterations to run.");

typedef float Dtype;
MPI_Datatype MY_MPI_TYPE = MPI_FLOAT;

void send_diff(const Blob<Dtype>* param, int dst)
{
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Isend(param -> cpu_diff(), param -> count(), MY_MPI_TYPE, dst, 0, MPI_COMM_WORLD, &request);
}
void send_data(const Blob<Dtype>* param, int dst)
{
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Isend(param -> cpu_data(), param -> count(), MY_MPI_TYPE, dst, 0, MPI_COMM_WORLD, &request);
}
void recv_diff(Blob<Dtype>* param, int dst)
{
    MPI_Status status;
    MPI_Recv(param -> mutable_cpu_diff(), param -> count(), MY_MPI_TYPE, dst, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
}
void recv_data(Blob<Dtype>* param, int dst)
{
    MPI_Status status;
    MPI_Recv(param -> mutable_cpu_data(), param -> count(), MY_MPI_TYPE, dst, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
}

shared_ptr<Net<Dtype> > init_net(const string &net_def)
{
    NetState net_state;
    NetParameter net_param;

    ReadNetParamsFromTextFileOrDie(net_def, &net_param);
    net_state.set_phase(TRAIN);
    net_state.MergeFrom(net_param.state());
    net_param.mutable_state() -> CopyFrom(net_state);

    return shared_ptr<Net<Dtype> >(new Net<Dtype>(net_param));
}

double sumsq(const Dtype *vec, int cnt)
{
    double res = 0.0;
    for (int i = 0; i < cnt; i++) {
        res += vec[i] * vec[i];
    }
    return res;
}

double ApplyUpdate(shared_ptr<Net<Dtype> > net, Dtype lr, int node_id)
{
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net -> params();

    double grad_mag = 0;

    for (int i = 0; i < net_params.size(); i++) {
        auto param = net_params[i];
        recv_diff(param.get(), node_id);
        grad_mag += sumsq(param -> cpu_diff(), param -> count());
        caffe_scal(param -> count(), lr, param -> mutable_cpu_diff());
        param -> Update();
        send_data(param.get(), node_id);
    }
    return std::log(grad_mag);
}

int pid = -1;
int np = -1;

void init_mpi(int *argc, char ***argv)
{
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
}

int finish_mpi()
{
    return MPI_Finalize();
}

// for initialization
void broadcast_params(shared_ptr<Net<Dtype> > net)
{
    auto net_params = net -> params();

    for (int i = 0; i < net_params.size(); i++) {
        auto param = net_params[i];
        MPI_Bcast(param -> mutable_cpu_data(), param -> count(), MY_MPI_TYPE, 0, MPI_COMM_WORLD);
    }
}


void root_node(shared_ptr<Net<Dtype> > net, int iters, Dtype lr)
{
    const std::vector<Blob<Dtype>*>& result = net -> output_blobs();
    std::vector<string> res_names;
    std::vector<const Dtype *> res;
    int Nres = result.size();
    for (int j = 0; j < Nres; j++)
    {
        res_names.push_back(net ->blob_names()[net -> output_blob_indices()[j]]);
        const Dtype* result_vec = result[j]->cpu_data();
        res.push_back(result_vec);
    }

    Dtype grad;
    res_names.push_back("log(|g|^2)");
    res.push_back(&grad);
    ++Nres;

    boost::posix_time::ptime timer = boost::posix_time::microsec_clock::local_time();

    MPI_Status status;
    std::vector<Blob<Dtype>*> bottom_vec;
    float loss;

    for (int i = 0; i < iters; i++) {

        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        grad = ApplyUpdate(net, lr, status.MPI_SOURCE);
        for (int j = 0; j < result.size(); j++) {
            recv_data(result[j], status.MPI_SOURCE);
        }

        // net -> Forward(bottom_vec, &loss);

        std::cout << "iter: " << i << " ts(ms): " << (boost::posix_time::microsec_clock::local_time() - timer).total_milliseconds();

        for (int j = 0; j < Nres; j++) {
            CHECK(!std::isnan(res[j][0])) << "nan loss";
            std::cout << " " << res_names[j] << ": " << res[j][0];
        }

        std::cout << std::endl;
    }

}

void worker_node(shared_ptr<Net<Dtype> > net, int iters)
{
    std::vector<Blob<Dtype>*> bottom_vec;
    auto net_params = net -> params();
    const std::vector<Blob<Dtype>*>& result = net -> output_blobs();

    for (int i = 0; i < iters; i++) {
        Dtype loss = net -> ForwardBackward(bottom_vec);

        for (int j = 0; j < net_params.size(); j++) {
            auto param = net_params[j];
            send_diff(param.get(), 0);
        }
        for (int j = 0; j < result.size(); j++) {
            send_data(result[j], 0);
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
