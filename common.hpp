#ifndef HEADER_COMMON
#define HEADER_COMMON

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

typedef float Dtype;
using namespace caffe;


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

int num_of_params(shared_ptr<Net<Dtype> > &net)
{
    auto net_params = net -> params();
    int size = 0;
    for (int i = 0; i < net_params.size(); i++) {
        auto param = net_params[i];
        size += param -> count();
    }
    return size;
}

double sumsq(const Dtype *vec, int cnt)
{
    double res = 0.0;
    for (int i = 0; i < cnt; i++) {
        res += vec[i] * vec[i];
    }
    return res;
}

#endif
