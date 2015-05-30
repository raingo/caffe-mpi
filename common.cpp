/**
 * @author Yuncheng Li (raingomm[AT]gmail.com)
 * @version 2015/05/29
 */

#include <iostream>
#include "common.hpp"

using namespace std;

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

typedef float Dtype;

double sumsq(const Dtype *vec, int cnt)
{
    double res = 0.0;
    for (int i = 0; i < cnt; i++) {
        res += vec[i] * vec[i];
    }
    return res;
}
