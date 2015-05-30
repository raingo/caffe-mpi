#ifndef HEADER_EVALUATOR
#define HEADER_EVALUATOR

struct snapshot_t {
    vector<vector<Dtype> > params;
    int iter;
    long ts;
};

vector<snapshot_t> buffer;
int cur = 0;

void init_buffer(int iters, int snap_intv, shared_ptr<Net<Dtype> > net)
{
   int Nsnapshots = (iters + snap_intv - 1) / snap_intv;

   auto net_params = net -> params();

   for (int j = 0; j < Nsnapshots; j++) {
       snapshot_t snap;
       for (int i = 0; i < net_params.size(); i++) {
           vector<Dtype> param(net_params[i] -> count());
           snap.params.push_back(param);
       }
       buffer.push_back(snap);
   }
}

void snapshot(shared_ptr<Net<Dtype> > net, int iter, long ts)
{
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net->params();
    for (int i = 0; i < net_params.size(); i++) {
        memcpy(&buffer[cur].params[i][0], net_params[i] -> cpu_data(), net_params[i] -> count() * sizeof(Dtype));
        buffer[cur].iter = iter;
        buffer[cur].ts = ts;
    }
    ++cur;
}

void evaluate(shared_ptr<Net<Dtype> > net, int iters)
{
    auto begin = buffer.begin();
    auto end = buffer.end();

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


    for (; begin != end; begin++) {
        snapshot_t snap = *begin;

        for (int i = 0; i < net_params.size(); i++) {
            memcpy(net_params[i] -> mutable_cpu_data(), &snap.params[i][0], net_params[i] -> count() * sizeof(Dtype));
            // ready to cumulate diff
            fill(snap.params[i].begin(), snap.params[i].end(), Dtype(0.0));
        }

        vector<double> res;
        for (int k = 0; k < Nres; k++) {
            res.push_back(0.0);
        }

        for (int i = 0; i < iters; i++) {
            std::cout << snap.iter << " " << i << std::endl;
            net -> ForwardBackward(bottom_vec);
            for (int j = 0; j < net_params.size(); j++) {
                caffe_axpy(net_params[j] -> count(), Dtype(1.0), net_params[j] -> cpu_diff(), &snap.params[j][0]);
            }

            for (int k = 0; k < Nres; k++) {
                res[k] += res_ptr[k][0];
            }
        }

        double grad = 0.0;

        for (int j = 0; j < net_params.size(); j++) {
            caffe_scal(net_params[j] -> count(), Dtype(1.0 / iters), &snap.params[j][0]);
            grad += sumsq(&snap.params[j][0], net_params[j] -> count());
        }
        std::cout << "iter: " << snap.iter << " ts(ms): " << snap.ts << " log(|g|^2): " << std::log(grad);
        for (int k = 0; k < Nres; k++) {
            std::cout << " " << res_names[k] << ": " << res[k] / iters;
        }
        std::cout << std::endl;
    }
}


#endif
