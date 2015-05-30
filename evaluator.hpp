#ifndef HEADER_EVALUATOR
#define HEADER_EVALUATOR

#include "snapshot.pb.h"

BufferProto buffer;
int cur = 0;

void init_buffer(int iters, int snap_intv, shared_ptr<Net<Dtype> > net)
{
   int Nsnapshots = (iters + snap_intv - 1) / snap_intv;

   auto net_params = net -> params();

   for (int j = 0; j < Nsnapshots; j++) {
       SnapShotProto *snap = buffer.add_snap();
       snap -> set_iter(0);
       snap -> set_ts(0);

       for (int i = 0; i < net_params.size(); i++) {
           ParamProto *param = snap -> add_param();
           for (int k = 0; k < net_params[i] -> count(); k++) {
               param -> add_data(0.0);
           }
       }
   }
}

void snapshot(shared_ptr<Net<Dtype> > net, int iter, long ts)
{
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net->params();
    SnapShotProto *snap = buffer.mutable_snap(cur);
    snap -> set_iter(iter);
    snap -> set_ts(ts);

    for (int i = 0; i < net_params.size(); i++) {
        ParamProto *param = snap -> mutable_param(i);
        memcpy(param -> mutable_data() -> mutable_data(), net_params[i] -> cpu_data(), net_params[i] -> count() * sizeof(Dtype));
    }
    ++cur;
}

void save_snapshot(const string path)
{
    WriteProtoToBinaryFile(buffer, path.c_str());
}

void load_snapshot(const string path)
{
  ReadProtoFromBinaryFile(path.c_str(), &buffer);
}

void evaluate(shared_ptr<Net<Dtype> > net, int iters)
{

    const vector<shared_ptr<Blob<Dtype> > >& net_params = net->params();
    std::vector<Blob<Dtype>*> bottom_vec;

    const std::vector<Blob<Dtype>*>& result = net -> output_blobs();
    std::vector<string> res_names;
    std::vector<const Dtype *> res_ptr;
    int Nres = result.size();
    for (int j = 0; j < Nres; j++)
    {
        res_names.push_back(net ->blob_names()[net -> output_blob_indices()[j]]);
        const Dtype* result_vec = result[j]->cpu_data();
        res_ptr.push_back(result_vec);
    }

    int Nsnapshots = buffer.snap_size();
    for (int nn = 0; nn < Nsnapshots; nn++) {
        SnapShotProto snap = buffer.snap(nn);
        long iter = snap.iter();
        long ts = snap.ts();

        for (int i = 0; i < net_params.size(); i++) {
            ParamProto param = snap.param(i);
            memcpy(net_params[i] -> mutable_cpu_data(), param.data().data(), net_params[i] -> count() * sizeof(Dtype));
            // ready to cumulate diff
            memset(param.mutable_data()->mutable_data(), 0, net_params[i] -> count() * sizeof(Dtype));
        }

        vector<double> res;
        for (int k = 0; k < Nres; k++) {
            res.push_back(0.0);
        }

        for (int i = 0; i < iters; i++) {
            net -> ForwardBackward(bottom_vec);
            for (int j = 0; j < net_params.size(); j++) {
                ParamProto param = snap.param(j);
                caffe_axpy(net_params[j] -> count(), Dtype(1.0 / iters), net_params[j] -> cpu_diff(), param.mutable_data()->mutable_data());
            }

            for (int k = 0; k < Nres; k++) {
                res[k] += res_ptr[k][0] / iters;
            }
        }

        double grad = 0.0;

        for (int j = 0; j < net_params.size(); j++) {
            ParamProto param = snap.param(j);
            grad += sumsq(param.data().data(), net_params[j] -> count());
        }
        std::cout << "iter: " << iter << " ts(ms): " << ts << " log(|g|^2): " << std::log(grad);
        for (int k = 0; k < Nres; k++) {
            std::cout << " " << res_names[k] << ": " << res[k];
        }
        std::cout << std::endl;
    }
}

#endif
