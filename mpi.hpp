#ifndef HEADER_MPI
#define HEADER_MPI

#include <mpi.h>
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
// for initialization
void broadcast_params(shared_ptr<Net<Dtype> > net)
{
    auto net_params = net -> params();

    for (int i = 0; i < net_params.size(); i++) {
        auto param = net_params[i];
        MPI_Bcast(param -> mutable_cpu_data(), param -> count(), MY_MPI_TYPE, 0, MPI_COMM_WORLD);
    }
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




#endif
