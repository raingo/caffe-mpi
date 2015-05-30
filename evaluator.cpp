/**
 * @author Yuncheng Li (raingomm[AT]gmail.com)
 * @version 2015/05/30
 */

#include "common.hpp"
#include "flags.hpp"
#include "evaluator.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    FLAGS_minloglevel = 3;
    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);

    // Set device id and mode
    if (FLAGS_gpu >= 0) {
        Caffe::SetDevice(FLAGS_gpu);
        Caffe::set_mode(Caffe::GPU);
    } else {
        Caffe::set_mode(Caffe::CPU);
    }

    load_snapshot(FLAGS_snap_path);
    string netdefs = FLAGS_model;
    auto net = init_net(netdefs);

    evaluate(net, FLAGS_eval_iter);

    return 0;
}
