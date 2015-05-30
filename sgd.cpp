#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

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

double ApplyUpdate(shared_ptr<Net<Dtype> > net, Dtype lr)
{
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net -> params();

    double grad_mag = 0;

    for (int i = 0; i < net_params.size(); i++) {
        auto param = net_params[i];
        grad_mag += sumsq(param -> cpu_diff(), param -> count());
        caffe_scal(param -> count(), lr, param -> mutable_cpu_diff());
    }

    net -> Update();
    return std::log(grad_mag);
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

    for (int i = 0; i < iters; i++) {
        Dtype loss = net -> ForwardBackward(bottom_vec);
        grad = ApplyUpdate(net, lr);
        std::cout << "iter: " << i << " ts(ms): " << (boost::posix_time::microsec_clock::local_time() - timer).total_milliseconds();
        for (int j = 0; j < Nres; j++) {
            std::cout << " " << res_names[j] << ": " << res[j][0];
        }

        std::cout << std::endl;
    }
    return 0;
}
