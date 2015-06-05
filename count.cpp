/**
 * @author Yuncheng Li (raingomm[AT]gmail.com)
 * @version 2015/06/05
 */

#include "common.hpp"
#include "flags.hpp"
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{

    FLAGS_minloglevel = 3;
    caffe::GlobalInit(&argc, &argv);
    string netdefs = FLAGS_model;
    auto net = init_net(netdefs);
    cout << "size of net: " << num_of_params(net) << endl;
    return 0;
}
